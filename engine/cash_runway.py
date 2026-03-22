"""
Cash Runway Engine
------------------
Projects days of cash remaining based on:
  - Actual RON bank balance derived from banking_transactions.json (open banking)
  - Outstanding invoices not yet reflected in the bank account
  - Projected inflows discounted by payment-risk scores

Real balance source
-------------------
When banking_transactions.json is present the engine computes the true current
balance (opening_balance + sum of all settled transactions up to today) and uses
it as the starting point for the 90-day projection.  If the banking dataset is
absent it falls back to STARTING_CASH (500,000 RON) — the original behaviour.

Public API
----------
    results = run(invoices, payment_risk_results, starting_cash) -> dict

Return structure:
    {
      "runway_days":        int,
      "runway_label":       str,   ("critical" | "at_risk" | "stable" | "healthy")
      "projected_balance":  list[dict],   # daily balance for next 90 days
      "monthly_summary":    list[dict],
      "overdue_exposure":   float,        # RON value of overdue receivables
      "alerts":             list[dict],
    }
"""

import json
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).parent.parent / "data"
DATA_PATH  = DATA_DIR / "invoices.json"
BANKING_ACCOUNTS_PATH = DATA_DIR / "banking_accounts.json"
BANKING_TXNS_PATH     = DATA_DIR / "banking_transactions.json"

EUR_TO_RON = 4.97

# Thresholds (days)
CRITICAL_DAYS  = 30
AT_RISK_DAYS   = 60
STABLE_DAYS    = 90

GROSS_MARGIN   = 0.15
STARTING_CASH  = 500_000  # RON — fallback when banking data is absent


def _to_ron(amount: float, currency: str) -> float:
    return amount * EUR_TO_RON if currency == "EUR" else amount


def _real_ron_balance(today: date) -> float | None:
    """
    Derive the true RON balance for RomDistrib SA from banking_transactions.json.

    Sums all settled transactions (value_date <= today) on the RON account and
    adds the EUR account balance converted at EUR_TO_RON.  Returns None if the
    banking dataset does not exist.
    """
    if not BANKING_ACCOUNTS_PATH.exists() or not BANKING_TXNS_PATH.exists():
        return None

    with open(BANKING_ACCOUNTS_PATH, encoding="utf-8") as f:
        accounts = json.load(f)
    with open(BANKING_TXNS_PATH, encoding="utf-8") as f:
        txns = json.load(f)

    # Map account_id → opening balance and currency
    acct_map = {a["account_id"]: a for a in accounts}

    balance_ron = 0.0
    balance_eur = 0.0

    for acct in accounts:
        opening = acct["opening_balance"]
        if acct["currency"] == "RON":
            balance_ron += opening
        else:
            balance_eur += opening

    for txn in txns:
        if date.fromisoformat(txn["value_date"]) > today:
            continue
        acct = acct_map.get(txn["account_id"])
        if acct is None:
            continue
        if acct["currency"] == "RON":
            balance_ron += txn["amount"]
        else:
            balance_eur += txn["amount"]

    return round(balance_ron + balance_eur * EUR_TO_RON, 2)


# ---------------------------------------------------------------------------
# Build cash-flow timeline
# ---------------------------------------------------------------------------

def _build_cashflows(df: pd.DataFrame,
                     risk_map: dict[str, float],
                     today: date) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: date, amount_ron, flow_type
    flow_type: 'inflow' | 'outflow' | 'projected_inflow' | 'projected_outflow'
    """
    rows = []

    for _, inv in df.iterrows():
        amount_ron = _to_ron(float(inv["amount"]), inv["currency"])
        due_dt     = pd.to_datetime(inv["due_date"]).date()
        paid_dt    = (pd.to_datetime(inv["actual_payment_date"]).date()
                      if pd.notna(inv["actual_payment_date"]) else None)

        late_prob = risk_map.get(inv["invoice_id"], 0.5)

        # --- Outflow: the company pays its supplier ---
        if paid_dt:
            rows.append({"date": paid_dt, "amount_ron": -amount_ron,
                         "flow_type": "outflow"})
        elif due_dt >= today:
            # Future payable — slightly discounted for early payment likelihood
            rows.append({"date": due_dt, "amount_ron": -amount_ron,
                         "flow_type": "projected_outflow"})
        else:
            # Overdue payable not yet settled
            rows.append({"date": today, "amount_ron": -amount_ron,
                         "flow_type": "projected_outflow"})

        # --- Inflow: resale of goods at margin ---
        resale     = amount_ron * (1 + GROSS_MARGIN)
        # Inflow comes in after an assumed 30-day sales cycle
        inflow_dt  = due_dt + timedelta(days=30)

        if inflow_dt < today:
            # Historical inflow already received
            rows.append({"date": inflow_dt, "amount_ron": resale,
                         "flow_type": "inflow"})
        else:
            # Future inflow discounted by late probability
            expected = resale * (1 - late_prob * 0.4)
            rows.append({"date": inflow_dt, "amount_ron": expected,
                         "flow_type": "projected_inflow"})

    return pd.DataFrame(rows)


def _project_balance(cashflows: pd.DataFrame,
                     starting_cash: float,
                     today: date,
                     horizon_days: int = 90,
                     real_balance: bool = False) -> list[dict]:
    """Roll forward daily balance for `horizon_days` from today.

    When real_balance=True the starting_cash already reflects all settled
    transactions (from open banking), so historical invoice flows are skipped
    to avoid double-counting.
    """
    end_date = today + timedelta(days=horizon_days)

    if real_balance:
        # starting_cash IS the true current balance — no historical adjustment
        current_cash = starting_cash
    else:
        # Estimate current cash by replaying historical invoice flows
        historical = cashflows[cashflows["flow_type"].isin(["inflow", "outflow"])]
        historical_sum = historical[historical["date"] < today]["amount_ron"].sum()
        current_cash = starting_cash + historical_sum

    # Future: project day-by-day
    future = cashflows[
        cashflows["flow_type"].isin(["projected_inflow", "projected_outflow"]) &
        (cashflows["date"] >= today) &
        (cashflows["date"] <= end_date)
    ].copy()

    daily = (future.groupby("date")["amount_ron"].sum()
                   .reindex(pd.date_range(today, end_date, freq="D"), fill_value=0))

    records = []
    balance = current_cash
    for dt, delta in daily.items():
        balance += delta
        records.append({
            "date":    dt.strftime("%Y-%m-%d"),
            "balance": round(balance, 2),
            "delta":   round(float(delta), 2),
        })

    return records


def _runway_days(projected: list[dict]) -> int:
    """How many days until balance first goes negative."""
    for i, day in enumerate(projected):
        if day["balance"] < 0:
            return i
    return len(projected)  # still positive at end of horizon


def _monthly_summary(cashflows: pd.DataFrame, today: date) -> list[dict]:
    cf = cashflows.copy()
    cf["month"] = pd.to_datetime(cf["date"]).dt.to_period("M")

    summary = []
    for month, grp in cf.groupby("month"):
        inflow  = grp[grp["amount_ron"] > 0]["amount_ron"].sum()
        outflow = grp[grp["amount_ron"] < 0]["amount_ron"].sum()
        summary.append({
            "month":       str(month),
            "inflow_ron":  round(inflow, 2),
            "outflow_ron": round(abs(outflow), 2),
            "net_ron":     round(inflow + outflow, 2),
            "is_projected": pd.Period(month) >= pd.Period(today, "M"),
        })

    return sorted(summary, key=lambda x: x["month"])


def _overdue_exposure(df: pd.DataFrame, today: date) -> float:
    """Total RON value of receivables not yet collected and past due."""
    overdue = df[
        (df["actual_payment_date"].isna()) &
        (pd.to_datetime(df["due_date"]).dt.date < today)
    ]
    return round(sum(_to_ron(float(r["amount"]), r["currency"])
                     for _, r in overdue.iterrows()), 2)


def _build_alerts(runway: int, overdue: float,
                  monthly: list[dict]) -> list[dict]:
    alerts = []

    if runway <= CRITICAL_DAYS:
        alerts.append({
            "alert_type": "critical_runway",
            "severity":   "critical",
            "message":    f"Cash runway is critically low — estimated {runway} days remaining.",
            "value":      runway,
        })
    elif runway <= AT_RISK_DAYS:
        alerts.append({
            "alert_type": "low_runway",
            "severity":   "warning",
            "message":    f"Cash runway is {runway} days — below the 60-day safety threshold.",
            "value":      runway,
        })

    if overdue > 50_000:
        alerts.append({
            "alert_type": "overdue_exposure",
            "severity":   "critical" if overdue > 150_000 else "warning",
            "message":    f"Overdue receivables: {overdue:,.0f} RON not yet collected.",
            "value":      overdue,
        })

    # Consecutive negative-net months
    neg_months = [m for m in monthly if m["is_projected"] and m["net_ron"] < 0]
    if len(neg_months) >= 2:
        alerts.append({
            "alert_type": "consecutive_negative",
            "severity":   "warning",
            "message":    (f"{len(neg_months)} consecutive projected months with negative "
                           f"cash flow."),
            "value":      len(neg_months),
        })

    return alerts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(invoices: list[dict] | None = None,
        payment_risk_results: list[dict] | None = None,
        starting_cash: float = STARTING_CASH) -> dict:

    if invoices is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            invoices = json.load(f)

    # Build risk map: invoice_id → late_prob
    risk_map: dict[str, float] = {}
    if payment_risk_results:
        risk_map = {r["invoice_id"]: r["late_prob"] for r in payment_risk_results}

    df    = pd.DataFrame(invoices)
    today = date(2025, 6, 30)  # fixed "today" for reproducible demo

    # Use real bank balance when open-banking data is available
    real_balance = _real_ron_balance(today)
    if real_balance is not None:
        starting_cash = real_balance
        balance_source = "open_banking"
    else:
        balance_source = "estimate"

    cashflows = _build_cashflows(df, risk_map, today)
    projected = _project_balance(cashflows, starting_cash, today, horizon_days=90,
                                 real_balance=(balance_source == "open_banking"))
    runway    = _runway_days(projected)
    monthly   = _monthly_summary(cashflows, today)
    overdue   = _overdue_exposure(df, today)
    alerts    = _build_alerts(runway, overdue, monthly)

    if runway <= CRITICAL_DAYS:
        label = "critical"
    elif runway <= AT_RISK_DAYS:
        label = "at_risk"
    elif runway <= STABLE_DAYS:
        label = "stable"
    else:
        label = "healthy"

    current_balance = projected[0]["balance"] if projected else starting_cash

    print(f"  [cash_runway] runway={runway}d  label={label}  "
          f"balance={current_balance:,.0f} RON ({balance_source})  "
          f"overdue={overdue:,.0f} RON  alerts={len(alerts)}")

    return {
        "runway_days":       runway,
        "runway_label":      label,
        "current_balance":   round(current_balance, 2),
        "balance_source":    balance_source,
        "overdue_exposure":  overdue,
        "projected_balance": projected,
        "monthly_summary":   monthly,
        "alerts":            alerts,
        "today":             str(today),
    }


if __name__ == "__main__":
    result = run()
    print(f"\nCash Runway: {result['runway_days']} days ({result['runway_label'].upper()})")
    print(f"Current balance : {result['current_balance']:>12,.0f} RON")
    print(f"Overdue exposure: {result['overdue_exposure']:>12,.0f} RON")
    print(f"\nProjected balance (next 30 days):")
    for day in result["projected_balance"][:30:3]:
        bar = "#" * max(0, int(day["balance"] / 50_000))
        print(f"  {day['date']}  {day['balance']:>12,.0f} RON  {bar}")
    if result["alerts"]:
        print(f"\nAlerts:")
        for a in result["alerts"]:
            print(f"  [{a['severity'].upper()}] {a['message']}")
