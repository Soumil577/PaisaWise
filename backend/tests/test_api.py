import os
import sys
import time
from pathlib import Path

os.environ["PAISAWISE_NO_AUTOLOAD"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import app as appmod


@pytest.fixture()
def client(tmp_path):
    store = appmod.ModelStore(appmod.BASE_MODEL, tmp_path)
    a = appmod.create_app(store)
    a.config["TESTING"] = True
    return a.test_client(), store


def demo_transactions(today="2026-10-11"):
    """Synthetic history shaped like the UI's demo data: monthly subscriptions + everyday spend + 1 outlier."""
    rng = np.random.default_rng(1)
    end = pd.Timestamp(today)
    tx, i = [], 0

    def add(desc, amt, cat, days_ago, merchant=""):
        nonlocal i
        i += 1
        ts = (end - pd.Timedelta(days=days_ago)).replace(hour=int(rng.integers(8, 22)), minute=int(rng.integers(0, 60)))
        tx.append(dict(id=f"t{i}", timestamp=ts.strftime("%Y-%m-%dT%H:%M"), amount=float(round(amt)),
                       description=desc, category=cat, merchant=merchant))

    for k in range(6):
        add("Netflix subscription", 649, "Entertainment", 3 + 30 * k, "Netflix")
        add("Jio mobile recharge", 299, "Bills", 10 + 30 * k, "Jio")
    for _ in range(90):
        add("Swiggy dinner", rng.uniform(150, 600), "Food", int(rng.integers(0, 180)), "Swiggy")
    for _ in range(40):
        add("Uber ride", rng.uniform(90, 400), "Transport", int(rng.integers(0, 180)), "Uber")
    add("Swiggy dinner", 9800, "Food", 2, "Swiggy")   # clear outlier
    return tx


def test_health(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    j = r.get_json()
    assert j["status"] == "ok" and "Food" in j["model"]["classes"]
    assert j["model"]["source"].startswith("user-provided CSV:")
    assert j["model"]["n_train"] >= 100_000
    assert 0.0 <= j["model"]["holdout_accuracy"] <= 1.0
    assert j["model"]["kind"] == "logistic_regression_structured"


def test_index_served_with_security_headers(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and b"PaisaWise" in r.data
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in r.headers


def test_classify_batch(client):
    c, _ = client
    items = [dict(description="Swiggy dinner", amount=350), dict(description="Uber ride to office", amount=180),
             dict(description="BESCOM electricity bill", amount=1200), dict(description="Apollo pharmacy medicines", amount=450)]
    r = c.post("/api/classify", json=dict(items=items))
    assert r.status_code == 200
    got = [x["category"] for x in r.get_json()["results"]]
    assert got == ["Food", "Transport", "Bills", "Healthcare"]
    assert all(0 <= x["confidence"] <= 1 for x in r.get_json()["results"])


def test_structured_category_model(client):
    c, _ = client
    r = c.post("/api/classify-structured", json={"items": [
        {"transaction_type": "P2M", "amount": 350, "timestamp": "2024-10-08T12:30:00", "transaction_status": "SUCCESS", "device_type": "Android", "network_type": "4G"}
    ]})
    assert r.status_code == 200, r.get_data(as_text=True)
    item = r.get_json()["results"][0]
    assert item["source"] == "structured_model"
    assert item["category"] in {"Food", "Transport", "Shopping", "Entertainment", "Healthcare", "Education", "Bills", "Other"}
    assert 0 <= item["confidence"] <= 1


def test_classify_single_shortcut_and_validation(client):
    c, _ = client
    assert c.post("/api/classify", json=dict(description="zomato order", amount=200)).status_code == 200
    assert c.post("/api/classify", json=dict(items=[])).status_code == 400
    assert c.post("/api/classify", data="nope").status_code == 400
    assert c.post("/api/classify", json=dict(items=[dict(description="x", amount="abc")])).status_code == 400


def test_pii_not_needed_for_prediction(client):
    c, _ = client
    r = c.post("/api/classify", json=dict(items=[dict(description="UPI-SWIGGY-swiggy.stores@icici-402938-Payment", amount=340)]))
    assert r.get_json()["results"][0]["category"] == "Food"


def test_feedback_applies_immediately_and_persists(client, tmp_path):
    c, store = client
    desc = "petco dog food"
    before = c.post("/api/classify", json=dict(description=desc, amount=450)).get_json()["results"][0]
    r = c.post("/api/feedback", json=dict(description=desc, amount=450, category="Pets"))
    assert r.status_code == 201
    after = c.post("/api/classify", json=dict(description=desc, amount=450)).get_json()["results"][0]
    assert after["category"] == "Pets" and after["source"] == "user" and after["confidence"] == 1.0
    assert (store.corr_path).exists()
    # new instance reloads the correction
    s2 = appmod.ModelStore(appmod.BASE_MODEL, store.corr_path.parent)
    assert s2.classify([desc], [450])[0]["category"] == "Pets"
    # background retrain eventually learns the new class
    for _ in range(60):
        if store.retrains:
            break
        time.sleep(0.5)
    assert store.retrains >= 1
    assert "Pets" in store.clf.classes_
    assert c.post("/api/feedback", json=dict(description="!!!", amount=1, category="X")).status_code == 400
    assert c.post("/api/feedback", json=dict(description="a b", amount=1, category="")).status_code == 400


def test_analyze_recurring_anomaly_capacity(client):
    c, _ = client
    body = dict(transactions=demo_transactions(), budgets={"TOTAL": 25000, "Food": 8000}, today="2026-10-11T12:00")
    r = c.post("/api/analyze", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    keys = {x["key"]: x for x in j["recurring"]}
    assert any("netflix" in k for k in keys), keys.keys()
    assert any("jio" in k for k in keys)
    nf = next(v for k, v in keys.items() if "netflix" in k)
    assert nf["cadence"] == "monthly" and nf["amount_type"] == "fixed" and len(nf["ids"]) >= 5
    big = [a for a in j["anomalies"] if a["amount"] == 9800]
    assert big and big[0]["typical"] < 1000 and big[0]["votes"] >= 2
    cap = j["capacity"]
    assert cap["total_budget"] == 25000 and cap["month"] == "2026-10" and cap["days_left"] == 20
    assert "Food" in j["suggested_budgets"] or "TOTAL" in j["suggested_budgets"]


def test_analyze_handles_empty_garbage_and_timezones(client):
    c, _ = client
    assert c.post("/api/analyze", json=dict(transactions=[])).get_json()["n"] == 0
    junk = [dict(id="a", timestamp="not a date", amount=5, description="x"), dict(id="b", timestamp="2026-10-01T10:00:00Z", amount=-3, description="y"), "str"]
    assert c.post("/api/analyze", json=dict(transactions=junk)).get_json()["n"] == 0
    z = [dict(id=str(i), timestamp=f"2026-09-{i + 1:02d}T10:00:00.000Z", amount=100 + i, description="chai", category="Food") for i in range(20)]
    assert c.post("/api/analyze", json=dict(transactions=z, budgets={"TOTAL": 1000})).status_code == 200
    assert c.post("/api/analyze", json=dict(transactions="x")).status_code == 400
    assert c.post("/api/analyze", json=dict(transactions=z, budgets={"TOTAL": "abc"})).status_code == 400


def test_auth_token(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "TOKEN", "s3cret")
    a = appmod.create_app(appmod.ModelStore(appmod.BASE_MODEL, tmp_path))
    c = a.test_client()
    assert c.get("/api/health").status_code == 200            # health stays open so the UI can ask for the key
    assert c.get("/api/health").get_json()["auth_required"] is True
    assert c.post("/api/classify", json=dict(description="x y", amount=1)).status_code == 401
    ok = c.post("/api/classify", json=dict(description="swiggy", amount=1), headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "RATE", 5)
    a = appmod.create_app(appmod.ModelStore(appmod.BASE_MODEL, tmp_path))
    c = a.test_client()
    codes = [c.get("/api/health").status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]


def test_unknown_routes_json(client):
    c, _ = client
    r = c.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["error"]


def test_output_is_strict_json_even_with_one_month_of_data(client):
    """suggest_budgets yields NaN when there is no earlier full month; that must not leak as invalid JSON."""
    import json
    c, _ = client
    tx = [dict(id=str(i), timestamp=f"2026-10-{i % 9 + 1:02d}T10:00", amount=100 + i, description="chai", category="Food") for i in range(30)]
    r = c.post("/api/analyze", json=dict(transactions=tx, budgets={"TOTAL": 5000}, today="2026-10-10T09:00"))
    assert r.status_code == 200
    json.loads(r.get_data(as_text=True), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite " + x)))
