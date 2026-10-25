"""Train PaisaWise's structured transaction-category model from the supplied CSV.

This script deliberately does NOT generate or import synthetic training records.

Usage:
    python backend/train_model.py
    python backend/train_model.py --csv data/upi_transactions_2024.csv
    python backend/train_model.py --csv path/to/transactions.csv --out model/structured_category_model.joblib

Expected source columns (column names are matched case-insensitively):
    transaction type, merchant_category, amount (INR), timestamp,
    transaction_status, hour_of_day, day_of_week, is_weekend

The model predicts the dataset's merchant_category mapped into PaisaWise UI labels.
It is a structured-feature model; it does not learn from free-text transaction descriptions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "upi_transactions_2024.csv"
DEFAULT_OUT = ROOT / "model" / "structured_category_model.joblib"

# Preserve the app's existing eight-category vocabulary by combining related labels.
CATEGORY_MAP = {
    "grocery": "Food",
    "food": "Food",
    "fuel": "Transport",
    "transport": "Transport",
    "shopping": "Shopping",
    "entertainment": "Entertainment",
    "healthcare": "Healthcare",
    "education": "Education",
    "utilities": "Bills",
    "other": "Other",
}

# These are non-identifying transaction features that can be available at classification time.
CATEGORICAL_FEATURES = ["transaction_type", "transaction_status", "day_of_week", "device_type", "network_type"]
NUMERIC_FEATURES = ["amount", "hour_of_day", "is_weekend"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _find_column(frame: pd.DataFrame, *names: str) -> str | None:
    """Return the actual column matching any candidate name, ignoring case/space."""
    lookup = {str(col).strip().lower(): col for col in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def load_training_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Read, validate and normalize the supplied structured transaction dataset."""
    raw = pd.read_csv(csv_path, low_memory=False)
    aliases = {
        "transaction_type": ("transaction type", "transaction_type", "type"),
        "category": ("merchant_category", "merchant category", "category"),
        "amount": ("amount (inr)", "amount", "amt"),
        "timestamp": ("timestamp", "datetime", "transaction date"),
        "transaction_status": ("transaction_status", "transaction status", "status"),
        "hour_of_day": ("hour_of_day", "hour"),
        "day_of_week": ("day_of_week", "day of week", "weekday"),
        "is_weekend": ("is_weekend", "is weekend", "weekend"),
        "device_type": ("device_type", "device type"),
        "network_type": ("network_type", "network type"),
    }
    found = {key: _find_column(raw, *names) for key, names in aliases.items()}
    for required in ("transaction_type", "category", "amount"):
        if not found[required]:
            raise ValueError(f"CSV is missing required column for {required}: expected one of {aliases[required]}")

    data = pd.DataFrame(index=raw.index)
    for feature in CATEGORICAL_FEATURES:
        source = found.get(feature)
        data[feature] = raw[source].fillna("Unknown").astype(str).str.strip() if source else "Unknown"
    data["amount"] = pd.to_numeric(raw[found["amount"]], errors="coerce")

    # Derive missing time fields from the timestamp when possible; never invent timestamps.
    timestamp_col = found.get("timestamp")
    timestamps = pd.to_datetime(raw[timestamp_col], errors="coerce") if timestamp_col else pd.Series(pd.NaT, index=raw.index)
    hour_col, day_col, weekend_col = found.get("hour_of_day"), found.get("day_of_week"), found.get("is_weekend")
    data["hour_of_day"] = pd.to_numeric(raw[hour_col], errors="coerce") if hour_col else timestamps.dt.hour
    if day_col:
        data["day_of_week"] = raw[day_col].fillna("Unknown").astype(str).str.strip()
    else:
        data["day_of_week"] = timestamps.dt.day_name().fillna("Unknown")
    # day_of_week is categorical and already part of CATEGORICAL_FEATURES.
    data["is_weekend"] = (
        pd.to_numeric(raw[weekend_col], errors="coerce") if weekend_col
        else pd.Series(np.where(timestamps.notna(), timestamps.dt.dayofweek.ge(5).astype(float), np.nan), index=raw.index)
    )

    # Keep the categories defined in the source, folded into the app's labels.
    source_labels = raw[found["category"]].fillna("").astype(str).str.strip()
    labels = source_labels.str.lower().map(CATEGORY_MAP)
    valid = data["amount"].notna() & (data["amount"] >= 0) & labels.notna()
    data = data.loc[valid, FEATURES].copy()
    labels = labels.loc[valid].astype(str)
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} valid labelled rows found; at least 100 are required.")
    data["amount"] = np.log1p(data["amount"].astype(float))  # log scale handles the highly skewed transaction amounts
    data["hour_of_day"] = pd.to_numeric(data["hour_of_day"], errors="coerce")
    data["is_weekend"] = pd.to_numeric(data["is_weekend"], errors="coerce")

    info = {
        "source_file": csv_path.name,
        "source_rows": int(len(raw)),
        "usable_rows": int(len(data)),
        "discarded_rows": int(len(raw) - len(data)),
        "source_category_counts": {str(k): int(v) for k, v in source_labels[valid].value_counts().items()},
        "training_category_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "feature_columns": FEATURES,
        "target": "merchant_category mapped to PaisaWise category labels",
    }
    return data, labels, info


def build_pipeline() -> Pipeline:
    """Build an explainable baseline: imputation + one-hot categoricals + logistic regression."""
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    features = ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1200, random_state=42)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaisaWise structured category model from labelled transaction CSV.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV file to use as the main training dataset")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output path for the trained joblib model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for a reproducible hold-out split")
    args = parser.parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Training CSV not found: {csv_path}\nPlace it at {DEFAULT_CSV} or pass --csv.")

    started = time.perf_counter()
    X, y, source_info = load_training_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y,
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "model": "logistic_regression_structured",
        "source": f"user-provided CSV: {csv_path.name}",
        "dataset": source_info,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": sorted(y.unique().tolist()),
        "holdout": {
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "precision_macro": round(float(precision_score(y_test, predicted, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, predicted, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 4),
            "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        },
        "features": FEATURES,
        "feature_transformations": {"amount": "log1p(amount) to reduce right-skew"},
        "excluded_fields": ["transaction id", "sender/receiver age groups", "sender state", "sender/receiver bank", "fraud_flag"],
        "limitations": [
            "This model predicts categories from structured transaction fields, not free-text descriptions.",
            "The hold-out split is drawn from the supplied dataset; it does not establish performance on all real-world users or banks.",
            "The dataset's provenance and real-world authenticity have not been independently verified by this training script.",
            "Category labels are mapped from merchant_category; Grocery is grouped as Food, Fuel as Transport, and Utilities as Bills.",
        ],
        "training_seconds": round(time.perf_counter() - started, 2),
        "random_seed": args.seed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Trained on {len(X_train):,} rows; held out {len(X_test):,} rows.")
    print(f"Accuracy: {metrics['holdout']['accuracy']:.4f} | Macro F1: {metrics['holdout']['f1_macro']:.4f}")
    print(f"Saved model: {out_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
