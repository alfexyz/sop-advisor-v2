"""
Payment Risk Engine
-------------------
Trains an XGBoost classifier to predict which invoices will be paid late (>30 days
past due_date). Uses SHAP to produce per-invoice feature explanations.

Public API
----------
    results = run(invoices: list[dict]) -> list[dict]

Each result dict:
    invoice_id          str
    buyer_name          str
    supplier_name       str
    amount              float
    currency            str
    due_date            str
    actual_payment_date str | None
    is_late             bool | None   (ground truth, None if unpaid)
    late_prob           float         (0–1, model prediction)
    confidence_pct      int           (0–100)
    top_factors         list[dict]    [{feature, value, shap_value, direction}, ...]
    days_late_estimate  int           (estimated days past due if late)
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent / "data" / "invoices.json"

# Feature display names for the advisory layer
FEATURE_LABELS = {
    "buyer_late_rate":        "buyer's historical late-payment rate",
    "amount":                 "invoice amount",
    "payment_terms_days":     "payment terms (days)",
    "month":                  "invoice month",
    "days_to_due":            "days until due",
    "amount_vs_buyer_avg":    "amount vs buyer's typical invoice",
    "category_late_rate":     "category late-payment rate",
    "supplier_enc":           "supplier identity",
    "buyer_enc":              "buyer identity",
    "category_enc":           "product category",
    "quarter":                "fiscal quarter",
    "is_eur":                 "EUR-denominated invoice",
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["issue_dt"] = pd.to_datetime(df["issue_date"])
    df["due_dt"]   = pd.to_datetime(df["due_date"])

    df["month"]          = df["issue_dt"].dt.month
    df["quarter"]        = df["issue_dt"].dt.quarter
    df["days_to_due"]    = (df["due_dt"] - df["issue_dt"]).dt.days
    df["is_eur"]         = (df["currency"] == "EUR").astype(int)

    # Historical late rate per buyer (leave-one-out to avoid leakage would be
    # ideal in production; for a hackathon prototype we use global buyer stats
    # computed on the labelled subset)
    labelled = df[df["is_late"].notna()]
    buyer_late = (labelled.groupby("buyer_name")["is_late"]
                           .mean()
                           .rename("buyer_late_rate"))
    cat_late   = (labelled.groupby("product_category")["is_late"]
                          .mean()
                          .rename("category_late_rate"))

    df = df.join(buyer_late, on="buyer_name")
    df = df.join(cat_late,   on="product_category")

    # Fill unknowns with overall mean
    overall_late = labelled["is_late"].mean()
    df["buyer_late_rate"]    = df["buyer_late_rate"].fillna(overall_late)
    df["category_late_rate"] = df["category_late_rate"].fillna(overall_late)

    # Amount vs buyer's average
    buyer_avg = (df.groupby("buyer_name")["amount"]
                   .mean()
                   .rename("buyer_avg_amount"))
    df = df.join(buyer_avg, on="buyer_name")
    df["amount_vs_buyer_avg"] = df["amount"] / df["buyer_avg_amount"].replace(0, 1)

    # Label-encode categoricals
    le = LabelEncoder()
    df["supplier_enc"] = le.fit_transform(df["supplier_name"])
    df["buyer_enc"]    = le.fit_transform(df["buyer_name"])
    df["category_enc"] = le.fit_transform(df["product_category"])

    return df


FEATURE_COLS = [
    "buyer_late_rate",
    "category_late_rate",
    "amount",
    "payment_terms_days",
    "month",
    "quarter",
    "days_to_due",
    "is_eur",
    "amount_vs_buyer_avg",
    "supplier_enc",
    "buyer_enc",
    "category_enc",
]


# ---------------------------------------------------------------------------
# Days-late estimator (simple regression on labelled late invoices)
# ---------------------------------------------------------------------------

def _build_days_late_estimator(df: pd.DataFrame):
    """Train a quick XGBoost regressor to estimate days late, given late=True."""
    late_df = df[df["days_late"] > 0].copy()
    if len(late_df) < 20:
        return None, None

    X = late_df[FEATURE_COLS].values
    y = late_df["days_late"].values

    reg = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    reg.fit(X, y)
    return reg, late_df[FEATURE_COLS].describe()  # unused, kept for symmetry


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train(df: pd.DataFrame):
    labelled = df[df["is_late"].notna()].copy()

    X = labelled[FEATURE_COLS].values
    y = labelled["is_late"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    probs = model.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, probs)
    print(f"  [payment_risk] XGBoost ROC-AUC on hold-out: {auc:.3f}")
    preds = (probs >= 0.5).astype(int)
    print(classification_report(y_te, preds, target_names=["on-time", "late"],
                                 zero_division=0))

    explainer = shap.TreeExplainer(model)
    return model, explainer


# ---------------------------------------------------------------------------
# SHAP → human-readable factors
# ---------------------------------------------------------------------------

def _top_factors(shap_values: np.ndarray, row: pd.Series,
                 n: int = 3) -> list[dict]:
    """Return top-n SHAP drivers for a single prediction."""
    factors = []
    for feat, shap_val, raw_val in zip(FEATURE_COLS, shap_values, row[FEATURE_COLS].values):
        factors.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "raw_value":    float(raw_val),
            "shap_value":   float(shap_val),
            "direction":    "increases risk" if shap_val > 0 else "reduces risk",
        })
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return factors[:n]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(invoices: list[dict] | None = None) -> list[dict]:
    """
    Train model on labelled invoices, then score all invoices.
    Returns a list of risk assessment dicts (see module docstring).
    """
    if invoices is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            invoices = json.load(f)

    df = pd.DataFrame(invoices)

    # Ground-truth label: paid >30 days late
    df["due_dt"]  = pd.to_datetime(df["due_date"])
    df["paid_dt"] = pd.to_datetime(df["actual_payment_date"], errors="coerce")
    df["days_late"] = (df["paid_dt"] - df["due_dt"]).dt.days.clip(lower=0).fillna(-1)
    df["is_late"] = np.where(
        df["actual_payment_date"].isna(), np.nan,
        (df["days_late"] > 30).astype(float)
    )

    df = build_features(df)

    print(f"  [payment_risk] Training on {df['is_late'].notna().sum()} labelled invoices…")
    model, explainer = _train(df)
    days_reg, _ = _build_days_late_estimator(df)

    # Score every invoice
    X_all   = df[FEATURE_COLS].values
    probs   = model.predict_proba(X_all)[:, 1]
    sv      = explainer.shap_values(X_all)  # shape: (n, features) for binary XGB

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        late_prob = float(probs[i])

        # Estimate days late
        if days_reg is not None and late_prob >= 0.4:
            est_days = max(1, int(round(days_reg.predict(X_all[i:i+1])[0])))
        else:
            est_days = 0

        results.append({
            "invoice_id":          row["invoice_id"],
            "buyer_name":          row["buyer_name"],
            "supplier_name":       row["supplier_name"],
            "amount":              float(row["amount"]),
            "currency":            row["currency"],
            "due_date":            row["due_date"],
            "actual_payment_date": row["actual_payment_date"],
            "is_late":             None if pd.isna(row["is_late"]) else bool(row["is_late"]),
            "late_prob":           late_prob,
            "confidence_pct":      int(round(late_prob * 100)),
            "top_factors":         _top_factors(sv[i], row),
            "days_late_estimate":  est_days,
        })

    results.sort(key=lambda x: -x["late_prob"])
    return results


if __name__ == "__main__":
    print("Running payment risk engine…")
    results = run()
    high_risk = [r for r in results if r["late_prob"] >= 0.6]
    print(f"\nHigh-risk invoices (prob ≥ 60%): {len(high_risk)}")
    for r in high_risk[:5]:
        print(f"  {r['invoice_id']}  {r['buyer_name']:<35} "
              f"prob={r['late_prob']:.0%}  ~{r['days_late_estimate']}d late  "
              f"amount={r['amount']:,.0f} {r['currency']}")
        for f in r["top_factors"]:
            print(f"    [{f['direction']:>14}]  {f['label']} = {f['raw_value']:.3g}  "
                  f"(SHAP {f['shap_value']:+.3f})")
