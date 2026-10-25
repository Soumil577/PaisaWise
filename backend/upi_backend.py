"""
upi_backend.py - Privacy-preserving, local-first UPI expense analytics backend.

Phase 1  Data & ingestion      : schema, PII scrubbing, cleaning, generic CSV loader
Phase 2  NLP categorisation    : TF-IDF (word + char) + Logistic Regression / calibrated Linear SVM,
                                 user corrections, user-defined categories
Phase 3  Financial analytics   : budgets, remaining capacity, recurring detection, anomaly detection
Phase 4  Local backend         : SQLite store, LocalExpenseEngine, loopback-only HTTP API, dashboard
Phase 5  Evaluation utilities  : NoNetworkGuard (privacy check), metric helpers

Everything runs on-device. Nothing in this module opens an outbound network connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

# =====================================================================================
# PHASE 1 - DATA & INGESTION (generic runtime preprocessing; no synthetic catalog)
# =====================================================================================
# ---------- PII scrubbing & text normalisation ----------
_VPA_RE = re.compile(r"[\w.]{2,}@[A-Za-z]{2,}")  # no "-": statement separators like UPI-SWIGGY-handle@bank must not swallow the merchant
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\-\s]?)?[6-9]\d{9}(?!\d)")
_LONGNUM_RE = re.compile(r"\b\d{6,}\b")
_REF_RE = re.compile(r"\bref\w*\d+\b", re.I)
_NOISE_RE = re.compile(r"\b(upi|paid to|payment to|paid|txn|neft|imps)\b")


def scrub_pii(text) -> str:
    """Replace UPI IDs, phone numbers and long reference numbers with neutral tokens."""
    t = str(text)
    t = _VPA_RE.sub(" upiid ", t)
    t = _PHONE_RE.sub(" phoneno ", t)
    t = _REF_RE.sub(" ", t)
    return _LONGNUM_RE.sub(" ", t)


def normalize_description(text) -> str:
    t = scrub_pii(text).lower()
    t = _NOISE_RE.sub(" ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def derive_merchant(desc_norm: str) -> str:
    """No merchant-name dictionary is embedded in the runtime.

    Merchant extraction is intentionally left blank unless the source CSV/API
    supplies a merchant column. Categorisation uses the full normalized narration.
    """
    return ""


# ---------- cleaning / feature extraction ----------
def add_time_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = d["timestamp"].dt.normalize()
    d["year_month"] = d["timestamp"].dt.strftime("%Y-%m")
    d["dow"] = d["timestamp"].dt.dayofweek
    d["hour"] = d["timestamp"].dt.hour
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["log_amount"] = np.log1p(d["amount"])
    return d


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, de-duplicate, scrub PII, normalise text, derive merchant and time features."""
    d = df.copy()
    missing = [c for c in ("timestamp", "amount", "description") if c not in d.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d["description"] = d["description"].fillna("").astype(str)
    if "user_id" not in d.columns:
        d["user_id"] = "local"
    d = d.dropna(subset=["timestamp", "amount"])
    d = d[d["amount"] > 0]
    subset = ["txn_id"] if "txn_id" in d.columns else ["user_id", "timestamp", "amount", "description"]
    d = d.drop_duplicates(subset=subset)
    d["desc_norm"] = d["description"].map(normalize_description)
    if "merchant" in d.columns:
        d["merchant"] = d["merchant"].fillna("").astype(str).map(normalize_description)
    else:
        d["merchant"] = ""
    d = add_time_features(d)
    return d.sort_values("timestamp").reset_index(drop=True)


_ALIASES = {
    "timestamp": ["timestamp", "date", "datetime", "time", "txn date", "transaction date", "value date"],
    "amount": ["amount", "amount (inr)", "amt", "value", "debit", "withdrawal", "txn amount"],
    "description": ["description", "reason", "remarks", "narration", "note", "details", "particulars"],
    "category": ["category", "label", "type"],
    "user_id": ["user_id", "user", "account"],
}


def load_transactions(path, colmap: dict | None = None) -> pd.DataFrame:
    """Load a CSV of UPI transactions (auto-maps common column names) and preprocess it."""
    raw = pd.read_csv(path)
    if colmap is None:
        low = {c.lower().strip(): c for c in raw.columns}
        colmap = {}
        for target, names in _ALIASES.items():
            for n in names:
                if n in low:
                    colmap[low[n]] = target
                    break
    return preprocess(raw.rename(columns=colmap))


# =====================================================================================
# PHASE 2 - NLP CATEGORISATION
# =====================================================================================
def amount_bucket(a):
    return np.clip(np.floor(np.log2(np.asarray(a, dtype=float) + 1)), 0, 20).astype(int)


def model_text(desc_norm, amounts=None, use_amount=True):
    desc_norm = list(desc_norm)
    if not use_amount or amounts is None:
        return desc_norm
    return [f"{d} amtb{b}" for d, b in zip(desc_norm, amount_bucket(amounts))]


def build_text_pipeline(kind="logreg") -> Pipeline:
    feats = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    if kind == "svm":
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
    elif kind == "linearsvc":  # no probabilities; for benchmarking only
        clf = LinearSVC(C=1.0, class_weight="balanced")
    else:
        clf = LogisticRegression(C=20, max_iter=3000, class_weight="balanced")
    return Pipeline([("tfidf", feats), ("clf", clf)])


class TransactionClassifier:
    """Lightweight on-device categoriser with user-correction memory."""

    def __init__(self, kind="logreg", use_amount=True, review_threshold=0.45, correction_weight=5):
        self.kind, self.use_amount = kind, use_amount
        self.review_threshold, self.correction_weight = review_threshold, correction_weight
        self.pipe = None
        self.classes_: list[str] = []
        self.overrides: dict[str, str] = {}   # desc_norm -> user category (exact-match memory)
        self._train = ([], [], [])            # desc_norm, amount, label
        self._corr = ([], [], [])

    def fit(self, desc_norm, amounts, labels):
        self._train = (list(desc_norm), list(amounts), list(labels))
        return self._refit()

    def _refit(self):
        d = self._train[0] + self._corr[0] * self.correction_weight
        a = self._train[1] + self._corr[1] * self.correction_weight
        y = self._train[2] + self._corr[2] * self.correction_weight
        self.pipe = build_text_pipeline(self.kind).fit(model_text(d, a, self.use_amount), y)
        self.classes_ = list(self.pipe.classes_)
        return self

    def predict(self, desc_norm, amounts) -> pd.DataFrame:
        desc_norm = list(desc_norm)
        if self.kind == "linearsvc":
            labels = self.pipe.predict(model_text(desc_norm, amounts, self.use_amount))
            conf = np.full(len(desc_norm), np.nan)
        else:
            proba = self.pipe.predict_proba(model_text(desc_norm, amounts, self.use_amount))
            labels = np.array(self.classes_)[proba.argmax(1)]
            conf = proba.max(1)
        out = pd.DataFrame({"category": labels, "confidence": conf, "source": "model"})
        for i, d in enumerate(desc_norm):
            if d in self.overrides:
                out.loc[i, ["category", "confidence", "source"]] = [self.overrides[d], 1.0, "user"]
        out["needs_review"] = (out["confidence"] < self.review_threshold) & (out["source"] == "model")
        return out

    def correct(self, desc_norm, amount, category, retrain=True):
        """Record a user correction (also how user-defined categories are introduced)."""
        self.overrides[desc_norm] = category
        self._corr[0].append(desc_norm); self._corr[1].append(float(amount)); self._corr[2].append(category)
        if retrain:
            self._refit()
        return self

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "TransactionClassifier":
        return joblib.load(path)


# =====================================================================================
# PHASE 3 - FINANCIAL ANALYTICS
# =====================================================================================
def _month_end(month: str) -> pd.Timestamp:
    return pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(0)


def suggest_budgets(df: pd.DataFrame, headroom=1.10) -> dict:
    """Starter budgets = median full-month spend per category x headroom."""
    last_full = df["year_month"].max()
    if df["timestamp"].max().normalize() < _month_end(last_full):
        full = df[df["year_month"] != last_full]
    else:
        full = df
    m = full.groupby(["year_month", "category"])["amount"].sum().unstack(fill_value=0)
    out = {"TOTAL": float(np.round(m.sum(axis=1).median() * headroom, -2))}
    for c in m.columns:
        out[c] = float(np.round(m[c].median() * headroom, -1))
    return out


def budget_report(df: pd.DataFrame, budgets: dict, month: str) -> pd.DataFrame:
    d = df[df["year_month"] == month]
    spent = d.groupby("category")["amount"].sum()
    rows = []
    for cat, b in budgets.items():
        s = float(d["amount"].sum()) if cat == "TOTAL" else float(spent.get(cat, 0.0))
        rows.append(dict(category=cat, budget=b, spent=round(s, 2), remaining=round(b - s, 2),
                         utilization=round(100 * s / b, 1) if b else np.nan))
    r = pd.DataFrame(rows, columns=["category", "budget", "spent", "remaining", "utilization"])
    r["status"] = np.select([r["utilization"] >= 100, r["utilization"] >= 80], ["OVER", "WARNING"], "OK")
    return r


# ---------- recurring-payment detection ----------
CADENCES = {"weekly": (7, 1.5, 5), "monthly": (30.4, 5, 4), "quarterly": (91, 10, 4), "yearly": (365, 15, 2)}


def _cadence(median_gap):
    for name, lo, hi in (("weekly", 5.5, 8.5), ("monthly", 25, 36), ("quarterly", 80, 100), ("yearly", 350, 380)):
        if lo <= median_gap <= hi:
            return name
    return None


def _eval_series(g: pd.DataFrame, amt_tol: float):
    if len(g) < 3:
        return None
    ts = g["timestamp"].sort_values()
    gaps = ts.diff().dt.total_seconds().div(86400).dropna().values
    cad = _cadence(float(np.median(gaps)))
    if cad is None:
        return None
    period, tol, min_occ = CADENCES[cad]
    if len(g) < min_occ:
        return None
    regularity = float(np.mean(np.abs(gaps - period) <= tol))
    if regularity < 0.7:
        return None
    amts = g["amount"].values
    med = float(np.median(amts))
    stab = float(np.mean(np.abs(amts - med) <= amt_tol * med))
    return dict(cadence=cad, occurrences=len(g), median_amount=med, regularity=round(regularity, 2),
                amount_stability=round(stab, 2), amount_type="fixed" if stab >= 0.8 else "variable",
                confidence=round(0.5 * regularity + 0.3 * stab + 0.2 * min(len(g) / 6, 1), 2),
                last_date=ts.iloc[-1], first_date=ts.iloc[0], idx=list(g.index))


def _amount_clusters(g: pd.DataFrame, tol: float):
    g = g.sort_values("amount")
    cluster, prev = [], None
    for i, a in zip(g.index, g["amount"].values):
        if prev is not None and a > prev * (1 + tol):
            yield g.loc[cluster]
            cluster = []
        cluster.append(i)
        prev = a
    if cluster:
        yield g.loc[cluster]


def detect_recurring(df: pd.DataFrame, amt_tol=0.15) -> pd.DataFrame:
    """Find subscription-like payments from description/merchant, amount and temporal similarity.

    Pass A groups by normalised description, pass B by merchant on the rows still unclaimed.
    Within a group: (1) the whole series must show a regular cadence (fixed or variable amount, e.g. an
    electricity bill); otherwise (2) rows are split into amount clusters and each fixed-amount cluster
    must show a regular cadence (e.g. a fixed recurring plan mixed with other purchases at the same merchant).
    """
    d = df.copy()
    if "user_id" not in d:
        d["user_id"] = "local"
    steps = {"weekly": pd.Timedelta(days=7), "monthly": pd.DateOffset(months=1),
             "quarterly": pd.DateOffset(months=3), "yearly": pd.DateOffset(years=1)}
    out = []
    for uid, u in d.groupby("user_id"):
        claimed = set()
        for key_col in ("desc_norm", "merchant"):
            pool = u.loc[~u.index.isin(claimed)]
            for key, g in pool.groupby(key_col):
                if len(g) < 3 or not key:
                    continue
                res = _eval_series(g, amt_tol=0.35)
                series = [res] if res else []
                if not res:
                    for c in _amount_clusters(g, amt_tol):
                        r = _eval_series(c, amt_tol)
                        if r and r["amount_type"] == "fixed":
                            series.append(r)
                for r in series:
                    claimed.update(r["idx"])
                    r.update(user_id=uid, key=key, next_expected=r["last_date"] + steps[r["cadence"]],
                             category=g.loc[r["idx"], "category"].mode().iloc[0] if "category" in g else "")
                    out.append(r)
    cols = ["user_id", "key", "category", "cadence", "occurrences", "median_amount", "amount_type",
            "regularity", "amount_stability", "confidence", "first_date", "last_date", "next_expected", "idx"]
    return pd.DataFrame(out, columns=cols).sort_values("confidence", ascending=False).reset_index(drop=True)


def tag_recurring(df: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["is_recurring"] = 0
    d["rec_key"] = ""
    for r in rec.itertuples():
        d.loc[r.idx, "is_recurring"] = 1
        d.loc[r.idx, "rec_key"] = r.key
    return d


# ---------- anomaly detection (personal baselines) ----------
def fit_baselines(d: pd.DataFrame, min_hist=15, min_hist_rec=3) -> dict:
    """Robust per-category (and per recurring series) statistics on log-amounts."""
    d = d.copy()
    d["_bkey"] = np.where(d["is_recurring"] == 1, "rec::" + d["rec_key"], d["category"])
    out = {}
    for k, g in d.groupby("_bkey"):
        if len(g) < (min_hist_rec if k.startswith("rec::") else min_hist):
            continue
        x = np.log1p(g["amount"].values)
        med = float(np.median(x))
        mad = float(max(np.median(np.abs(x - med)), 0.05))
        q1, q3 = np.percentile(x, [25, 75])
        out[k] = dict(med=med, mad=mad, q1=float(q1), q3=float(q3), iqr=float(max(q3 - q1, 0.10)), n=len(g))
    return out


def robust_z(b: dict, amount: float) -> float:
    return float(0.6745 * (np.log1p(amount) - b["med"]) / b["mad"])


def detect_anomalies(df: pd.DataFrame, method="ensemble", z_thresh=3.0, iqr_k=1.5, contamination=0.02,
                     random_state=0) -> pd.DataFrame:
    """Flag unusual spending relative to each user's own history.
    method: 'z' | 'iqr' | 'iforest' | 'ensemble' (>=2 of 3 votes)."""
    d = df.copy()
    if "is_recurring" not in d:
        d = tag_recurring(d, detect_recurring(d))
    if "user_id" not in d:
        d["user_id"] = "local"
    d["anom_z"], d["flag_z"], d["flag_iqr"], d["flag_if"] = 0.0, 0, 0, 0
    for uid, g in d.groupby("user_id"):
        base = fit_baselines(g)
        bkey = np.where(g["is_recurring"] == 1, "rec::" + g["rec_key"], g["category"])
        z = np.zeros(len(g)); iq = np.zeros(len(g), dtype=int)
        for j, (k, a) in enumerate(zip(bkey, g["amount"].values)):
            b = base.get(k)
            if b is None:
                continue
            z[j] = robust_z(b, a)
            iq[j] = int(np.log1p(a) > b["q3"] + iqr_k * b["iqr"])
        d.loc[g.index, "anom_z"] = z
        d.loc[g.index, "flag_z"] = (z >= z_thresh).astype(int)
        d.loc[g.index, "flag_iqr"] = iq
        feats = np.column_stack([g["log_amount"], z, g["hour"], g["dow"]])
        mask = (g["is_recurring"] == 0).values
        if mask.sum() >= 50:
            iso = IsolationForest(contamination=contamination, random_state=random_state).fit(feats[mask])
            pred = np.zeros(len(g), dtype=int)
            pred[mask] = (iso.predict(feats[mask]) == -1).astype(int)
            d.loc[g.index, "flag_if"] = pred
    votes = d["flag_z"] + d["flag_iqr"] + d["flag_if"]
    d["is_anomaly"] = {"z": d["flag_z"], "iqr": d["flag_iqr"], "iforest": d["flag_if"],
                       "ensemble": (votes >= 2).astype(int)}[method]
    return d


def analyze(df: pd.DataFrame, method="ensemble"):
    """Recurring + anomaly enrichment. Returns (enriched_df, recurring_df)."""
    rec = detect_recurring(df)
    return detect_anomalies(tag_recurring(df, rec), method=method), rec


def remaining_capacity(d: pd.DataFrame, budgets: dict, rec: pd.DataFrame, today=None) -> dict:
    """How much can still be spent this month, after committed recurring payments."""
    today = pd.Timestamp(today if today is not None else d["timestamp"].max()).normalize()
    month = today.strftime("%Y-%m")
    end = _month_end(month)
    elapsed, days_left = today.day, max((end - today).days, 0)
    m = d[(d["year_month"] == month) & (d["timestamp"] < today + pd.Timedelta(days=1))]
    spent = float(m["amount"].sum())
    disc = float(m.loc[m["is_recurring"] == 0, "amount"].sum()) if "is_recurring" in m else spent
    committed = 0.0
    if len(rec):
        due = rec[(rec["next_expected"] > today) & (rec["next_expected"] <= end)]
        committed = float(due["median_amount"].sum())
    total_budget = float(budgets.get("TOTAL", 0))
    remaining = total_budget - spent
    avail = remaining - committed
    return dict(month=month, as_of=str(today.date()), days_left=days_left, total_budget=total_budget,
                spent_so_far=round(spent, 2), remaining_budget=round(remaining, 2),
                committed_recurring=round(committed, 2), available_after_commitments=round(avail, 2),
                safe_daily_spend=round(max(avail, 0) / max(days_left, 1), 2),
                projected_month_end=round(spent + committed + disc / max(elapsed, 1) * days_left, 2))


# =====================================================================================
# PHASE 4 - LOCAL-FIRST BACKEND (SQLite + engine + loopback HTTP API + dashboard)
# =====================================================================================
def _json_default(o):
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (list, tuple, set)):
        return list(o)
    return str(o)


def to_jsonable(obj):
    return json.loads(json.dumps(obj, default=_json_default))


class LocalExpenseEngine:
    """Single-user, on-device expense engine. All state lives in a local SQLite file."""

    def __init__(self, db_path="upi_local.db", model_path="upi_model.joblib", user_id="local", kind="logreg"):
        self.db_path, self.model_path, self.user_id, self.kind = str(db_path), str(model_path), user_id, kind
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS transactions(
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts TEXT NOT NULL, amount REAL NOT NULL,
                description TEXT, desc_norm TEXT, merchant TEXT, category TEXT, category_source TEXT, confidence REAL);
            CREATE TABLE IF NOT EXISTS budgets(category TEXT PRIMARY KEY, amount REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS corrections(desc_norm TEXT PRIMARY KEY, category TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_ts ON transactions(ts);""")
        self.clf = TransactionClassifier.load(self.model_path) if Path(self.model_path).exists() else None
        self._cache = None
        self._base = None   # (recurring keys, baselines) - survives writes so live inserts stay fast

    # ---------------- storage ----------------
    def _insert(self, rows):
        self.db.executemany(
            "INSERT INTO transactions(user_id,ts,amount,description,desc_norm,merchant,category,category_source,confidence)"
            " VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self.db.commit()
        self._cache = None

    def import_dataframe(self, df: pd.DataFrame, user_id=None, train=True) -> int:
        """Bulk-load labelled history from a user-provided dataset or bank/UPI CSV export."""
        with self.lock:
            d = preprocess(df)
            if "user_id" in d and d["user_id"].nunique() > 1:
                if user_id is None:
                    raise ValueError("dataframe has several users; pass user_id=...")
                d = d[d["user_id"] == user_id]
            has_cat = "category" in d.columns
            rows = [(self.user_id, r.timestamp.isoformat(), float(r.amount), scrub_pii(r.description), r.desc_norm,
                     r.merchant, (r.category if has_cat and isinstance(r.category, str) else None),
                     "import" if has_cat else None, None) for r in d.itertuples()]
            self._insert(rows)
            if train and has_cat:
                self.train()
            return len(rows)

    def train(self):
        with self.lock:
            t = pd.read_sql_query("SELECT desc_norm, amount, category FROM transactions WHERE category IS NOT NULL", self.db)
            if t["category"].nunique() < 2:
                raise ValueError("need labelled data with >=2 categories to train")
            self.clf = TransactionClassifier(kind=self.kind).fit(t["desc_norm"], t["amount"], t["category"])
            for r in self.db.execute("SELECT desc_norm, category FROM corrections"):
                self.clf.overrides[r[0]] = r[1]
            self.clf.save(self.model_path)
            return self.clf

    # ---------------- analytics frame ----------------
    def frame(self, refresh=False):
        with self.lock:
            if self._cache is None or refresh:
                d = pd.read_sql_query("SELECT * FROM transactions ORDER BY ts", self.db)
                if d.empty:
                    return pd.DataFrame(), pd.DataFrame(), {}
                d = d.rename(columns={"ts": "timestamp"})
                d["timestamp"] = pd.to_datetime(d["timestamp"])
                d["category"] = d["category"].fillna("Uncategorized")
                d["merchant"] = d["merchant"].fillna("")
                d = add_time_features(d)
                enriched, rec = analyze(d)
                self._cache = (enriched, rec, fit_baselines(enriched))
                self._base = (set(rec["key"]) if len(rec) else set(), self._cache[2])
            return self._cache

    # ---------------- write operations ----------------
    def add_transaction(self, amount, description, timestamp=None, category=None) -> dict:
        """Categorise on-device, run anomaly check against personal history, store locally."""
        with self.lock:
            t0 = time.perf_counter()
            ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.now().floor("s")
            desc_norm = normalize_description(description)
            merchant = derive_merchant(desc_norm)
            conf, src, needs_review = None, "user", False
            if category is None:
                if self.clf is None:
                    category, src = "Uncategorized", "none"
                else:
                    p = self.clf.predict([desc_norm], [amount]).iloc[0]
                    category, conf, src, needs_review = p["category"], float(p["confidence"]), p["source"], bool(p["needs_review"])
            alert = self._live_anomaly(float(amount), category, merchant or desc_norm)
            self._insert([(self.user_id, ts.isoformat(), float(amount), scrub_pii(description), desc_norm,
                           merchant, category, src, conf)])
            tid = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
            return dict(id=tid, timestamp=str(ts), amount=float(amount), category=category, confidence=conf,
                        source=src, needs_review=needs_review, anomaly=alert,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def _live_anomaly(self, amount, category, key, z_thresh=3.0, iqr_k=1.5):
        if self._base is None:
            self.frame()
        if self._base is None:
            return None
        rec_keys, base = self._base
        if not base:
            return None
        b = base.get("rec::" + key) if key in rec_keys else base.get(category)
        if b is None:
            return None
        z = robust_z(b, amount)
        upper = float(np.expm1(b["q3"] + iqr_k * b["iqr"]))
        if z >= z_thresh or amount > upper:
            return dict(flag=True, robust_z=round(z, 2), typical=round(float(np.expm1(b["med"])), 2),
                        reason=f"{amount / max(np.expm1(b['med']), 1):.1f}x your usual spend for this category/payment")
        return dict(flag=False, robust_z=round(z, 2))

    def correct_category(self, txn_id: int, category: str, retrain=True) -> dict:
        """User correction: relabel one transaction, remember it, optionally retrain."""
        with self.lock:
            row = self.db.execute("SELECT desc_norm, amount FROM transactions WHERE id=?", (txn_id,)).fetchone()
            if row is None:
                raise KeyError(f"no transaction with id {txn_id}")
            self.db.execute("UPDATE transactions SET category=?, category_source='user', confidence=1 WHERE id=?",
                            (category, txn_id))
            self.db.execute("INSERT OR REPLACE INTO corrections VALUES (?,?)", (row[0], category))
            self.db.commit()
            self._cache = None
            if self.clf is not None:
                self.clf.correct(row[0], row[1], category, retrain=retrain)
                self.clf.save(self.model_path)
            return dict(id=txn_id, category=category, retrained=bool(retrain and self.clf is not None))

    def set_budget(self, category: str, amount: float):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO budgets VALUES (?,?)", (category, float(amount)))
            self.db.commit()

    def get_budgets(self) -> dict:
        with self.lock:
            return {c: a for c, a in self.db.execute("SELECT category, amount FROM budgets")}

    # ---------------- read operations ----------------
    def transactions(self, month=None, limit=200) -> pd.DataFrame:
        d, _, _ = self.frame()
        if month:
            d = d[d["year_month"] == month]
        return d.tail(limit)

    def default_month(self):
        d, _, _ = self.frame()
        return d["year_month"].max() if len(d) else None

    def budget_report(self, month=None) -> pd.DataFrame:
        d, _, _ = self.frame()
        return budget_report(d, self.get_budgets(), month or self.default_month())

    def capacity(self, today=None) -> dict:
        d, rec, _ = self.frame()
        return remaining_capacity(d, self.get_budgets(), rec, today)

    def recurring(self) -> pd.DataFrame:
        return self.frame()[1].drop(columns=["idx"], errors="ignore")

    def anomalies(self, month=None) -> pd.DataFrame:
        d, _, _ = self.frame()
        a = d[d["is_anomaly"] == 1]
        if month:
            a = a[a["year_month"] == month]
        return a[["id", "timestamp", "amount", "category", "description", "anom_z"]]

    def summary(self, month=None) -> dict:
        d, rec, _ = self.frame()
        month = month or self.default_month()
        m = d[d["year_month"] == month]
        wk = m.groupby("is_weekend")["amount"].sum()
        return to_jsonable(dict(
            month=month, total_spent=float(m["amount"].sum()), n_transactions=int(len(m)),
            category_spend=m.groupby("category")["amount"].sum().sort_values(ascending=False).round(2).to_dict(),
            weekday_vs_weekend=dict(weekday=float(wk.get(0, 0)), weekend=float(wk.get(1, 0))),
            budget_report=self.budget_report(month).to_dict("records"),
            capacity=self.capacity(),
            recurring=self.recurring().to_dict("records"),
            anomalies=self.anomalies(month).to_dict("records")))

    def close(self):
        self.db.close()


# ---------- loopback-only HTTP API (stdlib; no third-party server, no external exposure) ----------
def make_handler(engine: LocalExpenseEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence default stderr logging
            pass

        def _send(self, obj, code=200):
            body = json.dumps(to_jsonable(obj)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > 65536:
                raise ValueError("payload too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/health":
                    return self._send(dict(status="ok", local_only=True))
                if u.path == "/summary":
                    return self._send(engine.summary(q.get("month")))
                if u.path == "/transactions":
                    return self._send(engine.transactions(q.get("month"), int(q.get("limit", 50))).to_dict("records"))
                if u.path == "/budgets":
                    return self._send(engine.budget_report(q.get("month")).to_dict("records"))
                if u.path == "/recurring":
                    return self._send(engine.recurring().to_dict("records"))
                if u.path == "/anomalies":
                    return self._send(engine.anomalies(q.get("month")).to_dict("records"))
                if u.path == "/capacity":
                    return self._send(engine.capacity())
                self._send(dict(error="not found"), 404)
            except Exception as e:
                self._send(dict(error=str(e)), 400)

        def do_POST(self):
            try:
                b = self._body()
                if self.path == "/transactions":
                    return self._send(engine.add_transaction(b["amount"], b["description"], b.get("timestamp"), b.get("category")), 201)
                if self.path == "/correct":
                    return self._send(engine.correct_category(int(b["id"]), b["category"]))
                if self.path == "/budgets":
                    engine.set_budget(b["category"], b["amount"])
                    return self._send(dict(ok=True))
                self._send(dict(error="not found"), 404)
            except Exception as e:
                self._send(dict(error=str(e)), 400)
    return Handler


def serve(engine: LocalExpenseEngine, port=8000, background=False):
    """Serve the API on 127.0.0.1 only. Returns the server (call .shutdown() when background=True)."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        srv.serve_forever()
    return srv


# ---------- dashboard ----------
def render_dashboard(d: pd.DataFrame, budgets: dict, rec: pd.DataFrame, month: str, capacity: dict | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    m = d[d["year_month"] == month]
    cs = m.groupby("category")["amount"].sum().sort_values()
    ax[0, 0].barh(cs.index, cs.values, color="#3b6ea5")
    ax[0, 0].set_title(f"Category-wise spend · {month}")
    mt = d.groupby("year_month")["amount"].sum()
    ax[0, 1].plot(mt.index, mt.values, marker="o", color="#2a9d8f")
    if "TOTAL" in budgets:
        ax[0, 1].axhline(budgets["TOTAL"], ls="--", color="gray", label="monthly budget"); ax[0, 1].legend()
    ax[0, 1].set_title("Monthly spending trend"); ax[0, 1].tick_params(axis="x", rotation=60)
    br = budget_report(d, {k: v for k, v in budgets.items() if k != "TOTAL"}, month)
    if len(br):
        x = np.arange(len(br))
        ax[0, 2].bar(x - 0.2, br["budget"], 0.4, label="budget", color="#bbbbbb")
        ax[0, 2].bar(x + 0.2, br["spent"], 0.4, label="actual",
                     color=np.where(br["utilization"] >= 100, "#d62728", "#3b6ea5"))
        ax[0, 2].set_xticks(x); ax[0, 2].set_xticklabels(br["category"], rotation=60, ha="right")
        ax[0, 2].legend()
    ax[0, 2].set_title("Budget vs actual")
    cum = m.groupby(m["timestamp"].dt.day)["amount"].sum().cumsum()
    ax[1, 0].plot(cum.index, cum.values, marker=".", label="cumulative spend", color="#3b6ea5")
    if "TOTAL" in budgets:
        days = _month_end(month).day
        ax[1, 0].plot([1, days], [budgets["TOTAL"] / days, budgets["TOTAL"]], "--", color="gray", label="budget pace")
    if capacity:
        ax[1, 0].axhline(capacity["projected_month_end"], ls=":", color="#d62728", label="projected month-end")
    ax[1, 0].set_title("Month-to-date vs budget pace"); ax[1, 0].set_xlabel("day of month"); ax[1, 0].legend()
    if len(rec):
        r = rec.sort_values("median_amount")
        ax[1, 1].barh(r["key"], r["median_amount"], color="#e9c46a")
        ax[1, 1].set_title("Recurring payments (median amount)")
    else:
        ax[1, 1].text(.5, .5, "no recurring payments", ha="center")
    nm, an = d[d["is_anomaly"] == 0], d[d["is_anomaly"] == 1]
    ax[1, 2].scatter(nm["timestamp"], nm["amount"], s=4, alpha=.3, color="#999999", label="normal")
    ax[1, 2].scatter(an["timestamp"], an["amount"], s=22, color="#d62728", label="flagged")
    ax[1, 2].set_yscale("log"); ax[1, 2].set_title("Anomaly alerts"); ax[1, 2].legend()
    ax[1, 2].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# =====================================================================================
# PHASE 5 - EVALUATION UTILITIES (privacy verification)
# =====================================================================================
class NoNetworkGuard:
    """Context manager that blocks and records any non-loopback network activity."""
    LOOPBACK = {"127.0.0.1", "::1", "localhost", "", None}

    def __init__(self):
        self.attempts: list[str] = []

    def __enter__(self):
        g = self
        self._orig = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo)
        o_connect, o_connect_ex, o_gai = self._orig

        def _remote(addr):
            return isinstance(addr, tuple) and addr[0] not in g.LOOPBACK

        def connect(s, addr):
            if _remote(addr):
                g.attempts.append(f"connect {addr}")
                raise ConnectionRefusedError("blocked by NoNetworkGuard")
            return o_connect(s, addr)

        def connect_ex(s, addr):
            if _remote(addr):
                g.attempts.append(f"connect_ex {addr}")
                return 111
            return o_connect_ex(s, addr)

        def getaddrinfo(host, *a, **k):
            if host not in g.LOOPBACK:
                g.attempts.append(f"dns {host}")
                raise socket.gaierror("blocked by NoNetworkGuard")
            return o_gai(host, *a, **k)

        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
        return self

    def __exit__(self, *exc):
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = self._orig
        return False


def main():
    ap = argparse.ArgumentParser(description="Local-first UPI expense analytics backend")
    ap.add_argument("--db", default="upi_local.db")
    ap.add_argument("--model", default="upi_model.joblib")
    ap.add_argument("--csv", help="optional labelled CSV to import on first run")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    eng = LocalExpenseEngine(a.db, a.model)
    if a.csv:
        print("imported", eng.import_dataframe(load_transactions(a.csv)), "rows")
    print(f"Serving on http://127.0.0.1:{a.port}  (loopback only) - Ctrl+C to stop")
    serve(eng, a.port)


if __name__ == "__main__":
    main()
