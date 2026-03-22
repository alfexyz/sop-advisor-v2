"""
Margin Monitor Engine
---------------------
Tracks unit_price trends per product_category from supplier invoices.
Detects margin erosion using rolling averages and flags when cost increases
exceed thresholds.

Public API
----------
    results = run(invoices: list[dict]) -> dict

Return structure:
    {
      "alerts":   list[dict],   # categories/suppliers with flagged erosion
      "trends":   list[dict],   # monthly price index per category
      "summary":  dict,         # overall stats
    }

Alert dict:
    category            str
    supplier_name       str | None
    alert_type          str   ("margin_erosion" | "price_spike" | "accelerating_trend")
    severity            str   ("critical" | "warning" | "info")
    price_change_pct    float
    period_label        str
    rolling_avg_recent  float
    rolling_avg_baseline float
    currency            str
    description         str
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "invoices.json"

# Thresholds
EROSION_WARNING_PCT  = 5.0   # >5% price increase triggers warning
EROSION_CRITICAL_PCT = 12.0  # >12% triggers critical alert
SPIKE_PCT            = 20.0  # single-month jump >20% is a price spike
ROLLING_WINDOW       = 3     # months for short rolling average
BASELINE_WINDOW      = 6     # months for baseline


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _monthly_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate median unit_price per category per month (RON-normalised)."""
    d = df.copy()
    # Rough EUR→RON for comparison (use 4.97 constant; good enough for trend)
    d["unit_price_ron"] = np.where(
        d["currency"] == "EUR",
        d["unit_price"] * 4.97,
        d["unit_price"],
    )
    d["year_month"] = d["issue_dt"].dt.to_period("M")

    monthly = (
        d.groupby(["product_category", "year_month"])
         .agg(
             median_price=("unit_price_ron", "median"),
             invoice_count=("invoice_id", "count"),
             avg_quantity=("quantity", "mean"),
         )
         .reset_index()
    )
    monthly["year_month_str"] = monthly["year_month"].astype(str)
    return monthly


def _supplier_price_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate median unit_price per supplier per category per quarter."""
    d = df.copy()
    d["unit_price_ron"] = np.where(
        d["currency"] == "EUR", d["unit_price"] * 4.97, d["unit_price"]
    )
    d["quarter"] = d["issue_dt"].dt.to_period("Q")

    return (
        d.groupby(["supplier_name", "product_category", "quarter"])
         .agg(median_price=("unit_price_ron", "median"),
              invoice_count=("invoice_id", "count"))
         .reset_index()
    )


def _detect_alerts(monthly: pd.DataFrame) -> list[dict]:
    alerts = []

    for category, grp in monthly.groupby("product_category"):
        grp = grp.sort_values("year_month").reset_index(drop=True)
        if len(grp) < BASELINE_WINDOW + 1:
            continue

        prices = grp["median_price"].values

        # Rolling averages
        baseline = np.median(prices[:BASELINE_WINDOW])
        recent   = np.median(prices[-ROLLING_WINDOW:])

        if baseline == 0:
            continue

        change_pct = (recent - baseline) / baseline * 100

        # Check for price spike (largest single-month jump)
        month_changes = np.diff(prices) / np.maximum(prices[:-1], 1) * 100
        max_spike = float(month_changes.max()) if len(month_changes) else 0

        # Determine alert type and severity
        alert_type = None
        severity   = None

        if max_spike >= SPIKE_PCT:
            alert_type = "price_spike"
            severity   = "critical" if max_spike >= SPIKE_PCT * 1.5 else "warning"
        elif change_pct >= EROSION_CRITICAL_PCT:
            alert_type = "margin_erosion"
            severity   = "critical"
        elif change_pct >= EROSION_WARNING_PCT:
            alert_type = "margin_erosion"
            severity   = "warning"

        # Detect accelerating trend (last 3-month slope > first 3-month slope)
        if len(prices) >= 8:
            early_slope = np.polyfit(range(3), prices[:3], 1)[0]
            late_slope  = np.polyfit(range(3), prices[-3:], 1)[0]
            if late_slope > early_slope * 2 and late_slope > 0:
                if alert_type is None:
                    alert_type = "accelerating_trend"
                    severity   = "warning"

        if alert_type:
            period_label = (f"{grp['year_month_str'].iloc[0]} → "
                            f"{grp['year_month_str'].iloc[-1]}")
            alerts.append({
                "category":             category,
                "supplier_name":        None,
                "alert_type":           alert_type,
                "severity":             severity,
                "price_change_pct":     round(change_pct, 1),
                "max_spike_pct":        round(max_spike, 1),
                "period_label":         period_label,
                "rolling_avg_recent":   round(recent, 2),
                "rolling_avg_baseline": round(baseline, 2),
                "currency":             "RON",
                "month_count":          len(grp),
            })

    return alerts


def _supplier_alerts(sup_trends: pd.DataFrame) -> list[dict]:
    """Flag individual suppliers whose prices increased fastest."""
    alerts = []
    for (supplier, category), grp in sup_trends.groupby(["supplier_name", "product_category"]):
        grp = grp.sort_values("quarter").reset_index(drop=True)
        if len(grp) < 3:
            continue

        prices = grp["median_price"].values
        baseline = float(prices[0])
        recent   = float(prices[-1])
        if baseline == 0:
            continue

        change_pct = (recent - baseline) / baseline * 100

        if change_pct >= EROSION_WARNING_PCT:
            severity = "critical" if change_pct >= EROSION_CRITICAL_PCT else "warning"
            alerts.append({
                "category":             category,
                "supplier_name":        supplier,
                "alert_type":           "margin_erosion",
                "severity":             severity,
                "price_change_pct":     round(change_pct, 1),
                "max_spike_pct":        0.0,
                "period_label":         (f"{str(grp['quarter'].iloc[0])} → "
                                         f"{str(grp['quarter'].iloc[-1])}"),
                "rolling_avg_recent":   round(recent, 2),
                "rolling_avg_baseline": round(baseline, 2),
                "currency":             "RON",
                "month_count":          len(grp),
            })

    return alerts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(invoices: list[dict] | None = None) -> dict:
    if invoices is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            invoices = json.load(f)

    df = pd.DataFrame(invoices)
    df["issue_dt"] = pd.to_datetime(df["issue_date"])

    monthly     = _monthly_prices(df)
    sup_trends  = _supplier_price_trends(df)

    cat_alerts = _detect_alerts(monthly)
    sup_alerts = _supplier_alerts(sup_trends)

    # De-duplicate: if a supplier alert exists for a category, skip the
    # category-level alert (more specific is better)
    supplier_cats = {a["category"] for a in sup_alerts}
    all_alerts = sup_alerts + [a for a in cat_alerts if a["category"] not in supplier_cats]

    # Sort: critical first, then by magnitude
    all_alerts.sort(key=lambda a: (0 if a["severity"] == "critical" else 1,
                                   -abs(a["price_change_pct"])))

    # Build trend series for UI charting
    trends = (
        monthly[["product_category", "year_month_str", "median_price", "invoice_count"]]
        .rename(columns={"year_month_str": "month", "median_price": "price_ron"})
        .to_dict(orient="records")
    )

    summary = {
        "total_alerts":    len(all_alerts),
        "critical_count":  sum(1 for a in all_alerts if a["severity"] == "critical"),
        "warning_count":   sum(1 for a in all_alerts if a["severity"] == "warning"),
        "categories_analysed": monthly["product_category"].nunique(),
        "date_range": {
            "start": str(monthly["year_month"].min()),
            "end":   str(monthly["year_month"].max()),
        },
    }

    print(f"  [margin_monitor] {len(all_alerts)} alerts  "
          f"({summary['critical_count']} critical, {summary['warning_count']} warning)")

    return {"alerts": all_alerts, "trends": trends, "summary": summary}


if __name__ == "__main__":
    result = run()
    print(f"\nMargin alerts ({result['summary']['total_alerts']} total):")
    for a in result["alerts"]:
        sup = f" | {a['supplier_name']}" if a["supplier_name"] else ""
        print(f"  [{a['severity'].upper():>8}]  {a['category']:<20}{sup}")
        print(f"              {a['alert_type']} | +{a['price_change_pct']:.1f}%  "
              f"({a['period_label']})")
        print(f"              baseline {a['rolling_avg_baseline']:.2f} RON → "
              f"recent {a['rolling_avg_recent']:.2f} RON")
