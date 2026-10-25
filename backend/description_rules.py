"""Transparent description categorizer used for free-text entries in the app.

This rule-based helper intentionally replaces the old synthetic-text model for the
free-text workflow. It is not trained on generated examples. Users' explicit category
corrections take priority over every rule.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# Match specific terms before broad terms (e.g. "grocery" before generic "store").
RULES = [
    ("Food", ("swiggy", "zomato", "restaurant", "cafe", "coffee", "tea stall", "mess ", "pizza", "food", "dining", "bakery", "blinkit", "bigbasket", "grocery", "groceries", "zepto", "instamart")),
    ("Transport", ("uber", "ola", "rapido", "petrol", "fuel", "diesel", "indian oil", "bharat petroleum", "fastag", "irctc", "train ticket", "flight", "redbus", "metro", "parking", "transport", "cab ride")),
    ("Shopping", ("amazon", "flipkart", "myntra", "shopping", "croma", "decathlon", "retail", "store purchase", "clothing", "electronics")),
    ("Entertainment", ("netflix", "spotify", "hotstar", "disney", "bookmyshow", "pvr", "cinema", "movie", "steam", "gaming", "entertainment", "prime video")),
    ("Healthcare", ("pharmacy", "apollo", "medicine", "medical", "hospital", "doctor", "clinic", "practo", "pharmeasy", "diagnostic", "healthcare", "dental")),
    ("Education", ("udemy", "coursera", "tuition", "college fee", "exam fee", "school fee", "stationery", "textbook", "education", "course fee")),
    ("Bills", ("electricity", "bescom", "tneb", "water bill", "gas bill", "broadband", "wifi", "internet bill", "recharge", "airtel", "jio", "vodafone", "vi recharge", "rent", "utility", "utilities", "dth", "tata play")),
    ("Other", ("transfer to", "sent to", "paid to friend", "pocket money", "cash transfer", "donation", "gift")),
]

class DescriptionRuleClassifier:
    """Small deterministic classifier with the same interface the API expects."""
    def __init__(self, review_threshold: float = 0.45):
        self.kind = "transparent_keyword_rules"
        self.review_threshold = review_threshold
        self.classes_ = sorted({category for category, _ in RULES} | {"Other"})
        self.overrides: dict[str, str] = {}
        self._corr = ([], [], [])

    @staticmethod
    def _match(description: str) -> tuple[str, float]:
        text = re.sub(r"\s+", " ", str(description).lower()).strip()
        if not text:
            return "Other", 0.20
        for category, keywords in RULES:
            if any(keyword in text for keyword in keywords):
                return category, 0.90
        # The rule system cannot infer a reliable category from arbitrary text.
        return "Other", 0.25

    def predict(self, desc_norm, amounts) -> pd.DataFrame:
        rows = []
        for description in desc_norm:
            if description in self.overrides:
                category, confidence, source = self.overrides[description], 1.0, "user"
            else:
                category, confidence = self._match(description)
                source = "keyword_rule" if confidence >= 0.9 else "fallback"
            rows.append({
                "category": category,
                "confidence": confidence,
                "source": source,
                "needs_review": confidence < self.review_threshold,
            })
        return pd.DataFrame(rows, columns=["category", "confidence", "source", "needs_review"])

    def _refit(self):
        """There are no learned text weights to refit; retain corrected labels in metadata."""
        self.classes_ = sorted(set(self.classes_) | set(self._corr[2]))
        return self
