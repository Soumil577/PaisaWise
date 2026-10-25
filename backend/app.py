"""
app.py - PaisaWise web server: serves the front end and exposes the ML backend as a JSON API.

    python backend/app.py                       # local use: http://127.0.0.1:8000
    gunicorn -w 1 --threads 8 -b 0.0.0.0:$PORT --chdir backend app:app    # deployment

Design
  * Same origin: the UI (frontend/index.html) and the API (/api/*) come from one server, so there is no CORS to configure.
  * Stateless analytics: /api/analyze receives the user's transactions with each request and stores nothing.
    The browser (IndexedDB) stays the system of record; the server only categorises and analyses.
  * The only server-side state is the category-correction memory (data/corrections.json) that personalises the model.
  * One process holds the model, so run a single gunicorn worker (threads are fine).

Environment variables
  PAISAWISE_TOKEN          if set, every /api/* call except /api/health must send header  X-API-Key: <token>
  PAISAWISE_DATA_DIR       where corrections.json lives (default ./data) - mount a volume here when containerised
  PAISAWISE_STRUCTURED_MODEL path of the structured category model (default model/structured_category_model.joblib)
  PAISAWISE_CORS_ORIGINS   comma-separated origins allowed to call the API from another site (default: none)
  PAISAWISE_TRUST_PROXY    "1" when behind a reverse proxy that sets X-Forwarded-* (Render, Railway, nginx, ...)
  PAISAWISE_RATE_LIMIT     requests per minute per client for /api/* (default 240, 0 disables)
  PAISAWISE_MAX_TX         max transactions accepted by /api/analyze (default 20000)
  HOST / PORT              bind address for `python backend/app.py` (default 127.0.0.1 / 8000)
"""
from __future__ import annotations

import copy
import hmac
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from upi_backend import (  # noqa: E402
    analyze, fit_baselines, normalize_description, preprocess,
    remaining_capacity, suggest_budgets, to_jsonable,
)
from description_rules import DescriptionRuleClassifier  # noqa: E402

log = logging.getLogger("paisawise")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VERSION = "1.0.0"
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = Path(os.environ.get("PAISAWISE_DATA_DIR", ROOT / "data"))
STRUCTURED_MODEL = Path(os.environ.get("PAISAWISE_STRUCTURED_MODEL", ROOT / "model" / "structured_category_model.joblib"))
# Backwards-compatible name used by existing tests/scripts; it now points to the structured model.
BASE_MODEL = STRUCTURED_MODEL
TRAINING_CSV = Path(os.environ.get("PAISAWISE_TRAINING_CSV", ROOT / "data" / "upi_transactions_2024.csv"))
TOKEN = os.environ.get("PAISAWISE_TOKEN", "").strip()
CORS = [o.strip() for o in os.environ.get("PAISAWISE_CORS_ORIGINS", "").split(",") if o.strip()]
RATE = int(os.environ.get("PAISAWISE_RATE_LIMIT", "240"))
MAX_TX = int(os.environ.get("PAISAWISE_MAX_TX", "20000"))
MAX_CLASSIFY = 1000
MAX_CORRECTIONS = 5000


# ======================================================================================
# model store: base model + user corrections (exact-match memory now, retrain in background)
# ======================================================================================
class ModelStore:
    """Holds the structured ML model plus transparent rules for description-only entries.

    The structured model is trained from the supplied CSV. The browser's existing
    description/amount API does not receive the structured fields needed by that model,
    so free-text descriptions use clear keyword rules and user correction memory instead.
    """
    def __init__(self, base_path: Path, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        if not base_path.exists():
            if not TRAINING_CSV.exists():
                raise FileNotFoundError(
                    f"Structured model missing ({base_path}) and training CSV missing ({TRAINING_CSV}). "
                    "Include data/upi_transactions_2024.csv or set PAISAWISE_TRAINING_CSV."
                )
            log.warning("no structured model at %s - training from %s", base_path, TRAINING_CSV)
            subprocess.run([
                sys.executable, str(HERE / "train_model.py"),
                "--csv", str(TRAINING_CSV), "--out", str(base_path),
            ], check=True, cwd=str(ROOT))
        self.structured_model = joblib.load(base_path)
        self.base = DescriptionRuleClassifier()
        self.corr_path = data_dir / "corrections.json"
        self.corrections: list[list] = self._load_corrections()   # [normalized description, amount, category]
        self.lock = threading.Lock()
        self._dirty = threading.Event()
        self._worker: threading.Thread | None = None
        self.retrains = 0
        mp = base_path.parent / "metrics.json"
        self.metrics = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        self.clf = self._build(retrain=False)
        if self.corrections:
            self._kick()

    def _load_corrections(self) -> list:
        try:
            return json.loads(self.corr_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _build(self, retrain: bool) -> DescriptionRuleClassifier:
        """Rebuild the lightweight rule wrapper and preserve user corrections."""
        clf = copy.deepcopy(self.base)
        for d, a, c in list(self.corrections):
            clf.overrides[d] = c
            clf._corr[0].append(d)
            clf._corr[1].append(float(a))
            clf._corr[2].append(c)
        if retrain and self.corrections:
            clf._refit()
        return clf

    def _kick(self):
        self._dirty.set()
        with self.lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def _run(self):
        while self._dirty.is_set():
            time.sleep(0.25)  # debounce a burst of user corrections
            self._dirty.clear()
            try:
                new = self._build(retrain=True)
                new.overrides.update({d: c for d, _, c in self.corrections})
                self.clf = new
                self.retrains += 1
                log.info("refreshed keyword classifier with %d user correction(s)", len(self.corrections))
            except Exception:
                log.exception("could not refresh keyword classifier; exact-match corrections still apply")

    def classify(self, descriptions, amounts) -> list[dict]:
        """Categorize description-only inputs with transparent rules and saved corrections."""
        norm = [normalize_description(d) for d in descriptions]
        p = self.clf.predict(norm, amounts)
        return [dict(category=str(r.category), confidence=round(float(r.confidence), 4), source=str(r.source),
                     needs_review=bool(r.needs_review)) for r in p.itertuples()]

    def classify_structured(self, items: list[dict]) -> list[dict]:
        """Predict from the structured columns the training dataset actually contains."""
        rows = []
        for item in items:
            item = item or {}
            timestamp = pd.to_datetime(item.get("timestamp"), errors="coerce")
            day = str(item.get("day_of_week") or (timestamp.day_name() if not pd.isna(timestamp) else "Unknown"))
            hour = item.get("hour_of_day", timestamp.hour if not pd.isna(timestamp) else np.nan)
            weekend = item.get("is_weekend", int(timestamp.dayofweek >= 5) if not pd.isna(timestamp) else np.nan)
            rows.append({
                "transaction_type": str(item.get("transaction_type") or item.get("transaction type") or "Unknown"),
                "transaction_status": str(item.get("transaction_status") or item.get("transaction status") or "SUCCESS"),
                "day_of_week": day,
                "device_type": str(item.get("device_type") or item.get("device type") or "Unknown"),
                "network_type": str(item.get("network_type") or item.get("network type") or "Unknown"),
                "amount": float(np.log1p(max(_num(item.get("amount", item.get("amount (INR)", 0)) or 0, "amount"), 0.0))),
                "hour_of_day": _num(hour, "hour_of_day") if hour is not None and not pd.isna(hour) else np.nan,
                "is_weekend": _num(weekend, "is_weekend") if weekend is not None and not pd.isna(weekend) else np.nan,
            })
        frame = pd.DataFrame(rows)
        probabilities = self.structured_model.predict_proba(frame)
        classes = list(self.structured_model.classes_)
        results = []
        for probs in probabilities:
            index = int(np.argmax(probs))
            confidence = float(probs[index])
            results.append({
                "category": str(classes[index]),
                "confidence": round(confidence, 4),
                "source": "structured_model",
                "needs_review": confidence < 0.60,
            })
        return results

    def feedback(self, description: str, amount: float, category: str) -> dict:
        """Save user correction for exact descriptions without uploading it elsewhere."""
        d = normalize_description(description)
        if not d:
            raise ValueError("description has no usable text")
        with self.lock:
            self.corrections = [c for c in self.corrections if c[0] != d][-(MAX_CORRECTIONS - 1):]
            self.corrections.append([d, float(amount), category])
            tmp = self.corr_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.corrections), encoding="utf-8")
            os.replace(tmp, self.corr_path)
            self.clf.overrides[d] = category
            self.clf.classes_ = sorted(set(self.clf.classes_) | {category})
        self._kick()
        return dict(stored=True, corrections=len(self.corrections), will_retrain=True)


# ======================================================================================
# helpers
# ======================================================================================
def _bad(msg: str, code: int = 400):
    return jsonify(error=msg), code


def _num(v, name):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _ts(series: pd.Series) -> pd.Series:
    """Parse ISO strings (naive local wall-clock from the browser, or with Z/offset) into naive timestamps."""
    return pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601").dt.tz_localize(None)


def _clean(o):
    """NaN/inf are not valid JSON (browsers' JSON.parse rejects them) - turn them into null."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def run_analysis(store: ModelStore, payload: dict) -> dict:
    txs = payload.get("transactions")
    if not isinstance(txs, list):
        raise ValueError("transactions must be a list")
    if len(txs) > MAX_TX:
        raise ValueError(f"too many transactions (max {MAX_TX})")
    budgets = {str(k): float(v) for k, v in (payload.get("budgets") or {}).items() if _num(v, "budget") > 0}
    empty = dict(recurring=[], anomalies=[], capacity=None, suggested_budgets={}, n=0)
    rows = []
    for t in txs:
        if not isinstance(t, dict):
            continue
        rows.append(dict(txn_id=str(t.get("id", "")), timestamp=t.get("timestamp"), amount=t.get("amount"),
                         description=str(t.get("description", ""))[:200], category=str(t.get("category") or "Other"),
                         merchant=str(t.get("merchant") or "")[:80]))
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df["timestamp"] = _ts(df["timestamp"])
    d = preprocess(df)
    if d.empty:
        return empty
    enriched, rec = analyze(d)
    base = fit_baselines(enriched)

    recurring = []
    for r in rec.itertuples():
        recurring.append(dict(
            key=r.key, category=r.category, cadence=r.cadence, occurrences=int(r.occurrences),
            median_amount=float(r.median_amount), amount_type=r.amount_type, confidence=float(r.confidence),
            first_date=str(r.first_date.date()), last_date=str(r.last_date.date()),
            next_expected=str(pd.Timestamp(r.next_expected).date()), ids=d.loc[r.idx, "txn_id"].tolist()))

    anomalies = []
    for r in enriched[enriched["is_anomaly"] == 1].itertuples():
        b = base.get(("rec::" + r.rec_key) if r.is_recurring else r.category)
        typical = float(np.expm1(b["med"])) if b else float("nan")
        anomalies.append(dict(
            id=r.txn_id, amount=float(r.amount), z=round(float(r.anom_z), 2),
            votes=int(r.flag_z + r.flag_iqr + r.flag_if), typical=None if not b else round(typical, 2),
            baseline_n=int(b["n"]) if b else 0,
            reason=(f"{r.amount / max(typical, 1):.1f}x your usual spend for this category/payment" if b else "")))
    anomalies.sort(key=lambda a: -a["z"])

    capacity = None
    if budgets.get("TOTAL"):
        today = _ts(pd.Series([payload.get("today")])).iloc[0] if payload.get("today") else pd.Timestamp.now()
        capacity = remaining_capacity(enriched, {"TOTAL": budgets["TOTAL"]}, rec,
                                      today if pd.notna(today) else pd.Timestamp.now())
    try:
        suggested = suggest_budgets(enriched)
    except Exception:
        suggested = {}
    return _clean(to_jsonable(dict(recurring=recurring, anomalies=anomalies, capacity=capacity,
                                   suggested_budgets=suggested, n=int(len(d)),
                                   baseline_categories=[k for k in base if not k.startswith("rec::")])))


# ======================================================================================
# app
# ======================================================================================
def create_app(store: ModelStore | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app.json.sort_keys = False
    if os.environ.get("PAISAWISE_TRUST_PROXY") == "1":
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.store = store or ModelStore(BASE_MODEL, DATA_DIR)
    hits: dict[str, deque] = defaultdict(deque)

    @app.before_request
    def guard():
        if not request.path.startswith("/api/"):
            return None
        if request.method == "OPTIONS":      # CORS preflight; headers are added in after_request
            return ("", 204)
        if RATE:
            q, now = hits[request.remote_addr], time.time()
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE:
                return _bad("rate limit exceeded, slow down", 429)
            q.append(now)
        if TOKEN and request.path != "/api/health":
            if not hmac.compare_digest(request.headers.get("X-API-Key", ""), TOKEN):
                return _bad("missing or invalid API key", 401)
        return None

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
            origin = request.headers.get("Origin")
            if origin and origin in CORS:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
                resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                resp.headers["Vary"] = "Origin"
        else:
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' "
                "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: blob:; "
                "connect-src 'self'" + (" " + " ".join(CORS) if CORS else "") + "; frame-ancestors 'none'")
        return resp

    @app.errorhandler(404)
    def _404(e):
        return _bad("not found", 404)

    @app.errorhandler(413)
    def _413(e):
        return _bad("request too large", 413)

    @app.errorhandler(405)
    def _405(e):
        return _bad("method not allowed", 405)

    @app.errorhandler(Exception)
    def _500(e):
        log.exception("unhandled error")
        return _bad("internal error", 500)

    # ---------------- static front end ----------------
    @app.get("/")
    def index():
        resp = send_from_directory(FRONTEND_DIR, "index.html")
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    # ---------------- API ----------------
    @app.get("/api/health")
    def health():
        s = app.store
        m = s.metrics
        return jsonify(
            status="ok", version=VERSION, auth_required=bool(TOKEN),
            model=dict(kind=m.get("model", "logistic_regression_structured"), classes=list(s.structured_model.classes_),
                       n_train=m.get("n_train"), source=m.get("source"),
                       holdout_accuracy=(m.get("holdout") or {}).get("accuracy"),
                       holdout_f1=(m.get("holdout") or {}).get("f1_macro"),
                       description_categorizer=s.clf.kind, review_threshold=s.clf.review_threshold,
                       corrections=len(s.corrections), retrains=s.retrains))

    @app.post("/api/classify")
    def classify():
        b = request.get_json(silent=True)
        if not isinstance(b, dict):
            return _bad("JSON body required")
        items = b.get("items")
        if items is None and "description" in b:
            items = [dict(description=b["description"], amount=b.get("amount", 0))]
        if not isinstance(items, list) or not items:
            return _bad("items must be a non-empty list of {description, amount}")
        if len(items) > MAX_CLASSIFY:
            return _bad(f"at most {MAX_CLASSIFY} items per call")
        try:
            desc = [str((i or {}).get("description", ""))[:200] for i in items]
            amt = [max(_num((i or {}).get("amount", 0) or 0, "amount"), 0.0) for i in items]
        except ValueError as e:
            return _bad(str(e))
        return jsonify(results=app.store.classify(desc, amt))

    @app.post("/api/classify-structured")
    def classify_structured():
        """Predict categories from structured transaction attributes present in the training CSV."""
        b = request.get_json(silent=True)
        if not isinstance(b, dict):
            return _bad("JSON body required")
        items = b.get("items")
        if not isinstance(items, list) or not items:
            return _bad("items must be a non-empty list of structured transactions")
        if len(items) > MAX_CLASSIFY:
            return _bad(f"at most {MAX_CLASSIFY} items per call")
        try:
            return jsonify(results=app.store.classify_structured(items))
        except (ValueError, TypeError) as e:
            return _bad(str(e))

    @app.post("/api/feedback")
    def feedback():
        b = request.get_json(silent=True)
        if not isinstance(b, dict):
            return _bad("JSON body required")
        cat = str(b.get("category", "")).strip()
        if not (1 <= len(cat) <= 24):
            return _bad("category must be 1-24 characters")
        try:
            return jsonify(app.store.feedback(str(b.get("description", ""))[:200], _num(b.get("amount", 0) or 0, "amount"), cat)), 201
        except ValueError as e:
            return _bad(str(e))

    @app.post("/api/analyze")
    def analyze_route():
        b = request.get_json(silent=True)
        if not isinstance(b, dict):
            return _bad("JSON body required")
        t0 = time.perf_counter()
        try:
            out = run_analysis(app.store, b)
        except ValueError as e:
            return _bad(str(e))
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify(out)

    return app


app = create_app() if os.environ.get("PAISAWISE_NO_AUTOLOAD") != "1" else None


def main():
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    if host not in ("127.0.0.1", "localhost", "::1") and not TOKEN:
        log.warning("listening on %s without PAISAWISE_TOKEN - anyone who can reach this port can use the API", host)
    print(f"PaisaWise running at http://{host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
