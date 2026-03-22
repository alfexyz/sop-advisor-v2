"""
Advisory Layer
--------------
Consumes outputs from all three engines and produces plain-language advice
cards suitable for display in the Flask UI.

Public API
----------
    cards = generate_advice(risk_results, margin_results, cash_results) -> dict

Returns:
    {
      "payment_risk":  {"alerts": [...], "headline": str, "summary": str},
      "margin":        {"alerts": [...], "headline": str, "summary": str},
      "cash_runway":   {"alerts": [...], "headline": str, "summary": str},
    }

Each alert dict has at minimum:
    title        str   — one-line headline
    body         str   — 2–4 sentence explanation + recommendation
    severity     str   — "critical" | "warning" | "info"
    metric       str   — formatted key metric for the card badge
    invoice_id   str   — (payment alerts only)
"""

from __future__ import annotations

from datetime import date, datetime


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_amount(amount: float, currency: str) -> str:
    if currency == "EUR":
        return f"€{amount:,.0f}"
    return f"{amount:,.0f} RON"


def _fmt_pct(p: float) -> str:
    return f"{p:.0f}%"


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _days_label(n: int) -> str:
    return "1 day" if n == 1 else f"{n} days"


# ---------------------------------------------------------------------------
# SHAP factor → readable sentence fragment
# ---------------------------------------------------------------------------

_FACTOR_TEMPLATES = {
    "buyer's historical late-payment rate": (
        lambda v, d: (
            f"their historical late-payment rate is {v*100:.0f}%"
            if d == "increases risk"
            else f"their strong payment history ({100 - v*100:.0f}% on-time)"
        )
    ),
    "invoice amount": (
        lambda v, d: (
            f"the invoice size ({v:,.0f}) is above their comfort zone"
            if d == "increases risk"
            else f"the invoice is within their normal size range"
        )
    ),
    "amount vs buyer's typical invoice": (
        lambda v, d: (
            f"this invoice is {v:.1f}× their usual size"
            if d == "increases risk"
            else f"this is a standard-sized invoice for this buyer"
        )
    ),
    "invoice month": (
        lambda v, d: (
            f"month {int(v)} has historically seen more late payments"
            if d == "increases risk"
            else f"this month has a good historical payment record"
        )
    ),
    "fiscal quarter": (
        lambda v, d: (
            f"Q{int(v)} tends to be a tight cash period for this buyer"
            if d == "increases risk"
            else f"Q{int(v)} is typically a strong payment quarter"
        )
    ),
    "payment terms (days)": (
        lambda v, d: (
            f"extended {int(v)}-day terms stretch payment discipline"
            if d == "increases risk"
            else f"short {int(v)}-day terms help ensure timely payment"
        )
    ),
    "EUR-denominated invoice": (
        lambda v, d: (
            "EUR-denominated invoices face extra FX friction"
            if d == "increases risk"
            else "RON invoices avoid FX delays"
        )
    ),
    # Encoded identity features — skip, they add no narrative value
    "supplier identity": (lambda v, d: None),
    "buyer identity":    (lambda v, d: None),
    "product category":  (lambda v, d: None),
}

def _factor_to_phrase(factor: dict) -> str | None:
    """Returns a human-readable phrase, or None if the feature should be skipped."""
    label = factor["label"]
    val   = factor["raw_value"]
    dirn  = factor["direction"]
    fn = _FACTOR_TEMPLATES.get(label)
    if fn:
        try:
            return fn(val, dirn)   # may return None intentionally
        except Exception:
            pass
    # Fallback for unknown features
    sign = "high" if dirn == "increases risk" else "low"
    return f"{label} is {sign} ({val:.2g})"


# ---------------------------------------------------------------------------
# Payment risk advice
# ---------------------------------------------------------------------------

def _payment_alert(r: dict) -> dict:
    buyer    = r["buyer_name"]
    inv_id   = r["invoice_id"]
    amount   = _fmt_amount(r["amount"], r["currency"])
    prob     = r["late_prob"]
    conf     = r["confidence_pct"]
    est_days = r["days_late_estimate"]
    factors  = r["top_factors"]

    # Severity
    if prob >= 0.80:
        severity = "critical"
    elif prob >= 0.55:
        severity = "warning"
    else:
        severity = "info"

    # Factor phrases — skip encoded identity features (return None)
    phrases = [
        p for f in factors
        if f["direction"] == "increases risk"
        for p in [_factor_to_phrase(f)] if p is not None
    ]
    if not phrases:
        phrases = [
            p for f in factors[:3]
            for p in [_factor_to_phrase(f)] if p is not None
        ]
    factor_str = "; ".join(phrases[:3]) if phrases else "multiple risk signals"

    # Days-late flavour text
    delay_text = ""
    if est_days > 0:
        delay_text = f" The model estimates payment will arrive approximately {_days_label(est_days)} late."

    # Recommendation
    if prob >= 0.80:
        rec = (
            "Recommendation: Send a payment reminder now, before the due date. "
            "Consider requesting a 20–30% prepayment or a bank guarantee for orders "
            "above the buyer's usual invoice size."
        )
    elif prob >= 0.55:
        rec = (
            "Recommendation: Flag for proactive follow-up 5 days before due date. "
            "Review open credit limit and avoid shipping additional orders until "
            "this invoice is settled."
        )
    else:
        rec = (
            "Recommendation: No immediate action required, but monitor closely "
            "if additional invoices are raised for this buyer this month."
        )

    body = (
        f"{buyer} is predicted to pay invoice {inv_id} ({amount}) late "
        f"with {conf}% confidence. "
        f"Key drivers: {factor_str}.{delay_text} "
        f"{rec}"
    )

    return {
        "invoice_id": inv_id,
        "title":      f"{buyer} — {_fmt_pct(prob * 100)} late-payment risk",
        "body":       body,
        "severity":   severity,
        "metric":     f"{conf}% confidence",
        "buyer_name": buyer,
        "amount":     r["amount"],
        "currency":   r["currency"],
        "due_date":   r["due_date"],
        "days_late_estimate": est_days,
    }


def _payment_advice(risk_results: list[dict]) -> dict:
    # Focus on unresolved high-risk invoices (unpaid or paid late)
    actionable = [
        r for r in risk_results
        if r["late_prob"] >= 0.55 and (
            r["actual_payment_date"] is None or r["is_late"] is True
        )
    ]
    # Deduplicate: one alert per buyer, keep highest-risk invoice
    seen_buyers: dict[str, dict] = {}
    for r in actionable:
        b = r["buyer_name"]
        if b not in seen_buyers or r["late_prob"] > seen_buyers[b]["late_prob"]:
            seen_buyers[b] = r

    top = sorted(seen_buyers.values(), key=lambda x: -x["late_prob"])[:5]
    alerts = [_payment_alert(r) for r in top]

    total_high  = len([r for r in risk_results if r["late_prob"] >= 0.55])
    total_crit  = len([r for r in risk_results if r["late_prob"] >= 0.80])
    exposure    = sum(
        r["amount"] * (4.97 if r["currency"] == "EUR" else 1)
        for r in risk_results
        if r["late_prob"] >= 0.55 and r["actual_payment_date"] is None
    )

    if total_crit >= 3:
        headline = f"⚠ {total_crit} invoices at critical late-payment risk"
    elif total_high > 0:
        headline = f"{total_high} invoices flagged as late-payment risk"
    else:
        headline = "Payment risk is within normal range"

    summary = (
        f"{total_high} open invoices carry elevated late-payment risk "
        f"({total_crit} critical). "
        f"Total at-risk receivables: {exposure:,.0f} RON. "
        f"Top buyers to watch: "
        f"{', '.join(a['buyer_name'] for a in alerts[:3])}."
    ) if alerts else "No significant payment risk detected at this time."

    return {"alerts": alerts, "headline": headline, "summary": summary}


# ---------------------------------------------------------------------------
# Margin advice
# ---------------------------------------------------------------------------

_ALERT_TYPE_VERBS = {
    "margin_erosion":     "has risen",
    "price_spike":        "spiked sharply",
    "accelerating_trend": "is accelerating upward",
}

def _margin_alert(a: dict) -> dict:
    cat      = a["category"]
    supplier = a["supplier_name"]
    chg      = a["price_change_pct"]
    baseline = a["rolling_avg_baseline"]
    recent   = a["rolling_avg_recent"]
    period   = a["period_label"]
    atype    = a["alert_type"]
    severity = a["severity"]

    verb = _ALERT_TYPE_VERBS.get(atype, "changed")
    who  = f"{supplier} ({cat})" if supplier else cat

    impact_ron  = recent - baseline
    impact_pct  = chg

    # Recommendation
    if severity == "critical":
        rec = (
            f"Recommendation: Initiate a supplier negotiation for {supplier or cat}. "
            f"Request a price freeze or volume-based discount. "
            f"Meanwhile, review your own selling prices for {cat} — "
            f"a {impact_pct:.0f}% cost increase may require a {impact_pct * 0.7:.0f}% "
            f"selling-price adjustment to protect gross margin."
        )
    elif atype == "accelerating_trend":
        rec = (
            f"Recommendation: The cost trend is accelerating — lock in forward pricing "
            f"agreements before the next contract renewal. "
            f"Evaluate alternative suppliers as a hedge."
        )
    else:
        rec = (
            f"Recommendation: Monitor {supplier or cat} pricing monthly. "
            f"If the trend continues for another quarter, consider a price renegotiation "
            f"or partial supplier switch."
        )

    body = (
        f"The unit cost of {cat} {verb} by {impact_pct:.1f}% over {period} "
        f"(from {baseline:.2f} RON to {recent:.2f} RON per unit, +{impact_ron:.2f} RON). "
        f"{rec}"
    )

    spike_note = ""
    if atype == "price_spike" and a.get("max_spike_pct", 0) > 0:
        spike_note = f" Largest single-month jump: +{a['max_spike_pct']:.1f}%."
        body = body.rstrip() + spike_note

    return {
        "title":            f"{who} — cost {verb} {_fmt_pct(abs(impact_pct))}",
        "body":             body,
        "severity":         severity,
        "metric":           f"+{impact_pct:.1f}% cost",
        "category":         cat,
        "supplier_name":    supplier,
        "price_change_pct": chg,
        "baseline_ron":     baseline,
        "recent_ron":       recent,
    }


def _margin_advice(margin_results: dict) -> dict:
    alerts = [_margin_alert(a) for a in margin_results["alerts"][:5]]
    s      = margin_results["summary"]

    if s["critical_count"] >= 2:
        headline = f"⚠ {s['critical_count']} product categories facing critical cost inflation"
    elif s["total_alerts"] > 0:
        headline = f"{s['total_alerts']} margin alerts across {s['categories_analysed']} categories"
    else:
        headline = "Supplier costs are stable across all categories"

    worst = margin_results["alerts"][0] if margin_results["alerts"] else None
    summary = (
        f"{s['total_alerts']} cost-inflation alerts detected "
        f"({s['critical_count']} critical, {s['warning_count']} warning) "
        f"across {s['categories_analysed']} product categories "
        f"({s['date_range']['start']} – {s['date_range']['end']}). "
    )
    if worst:
        who = worst["supplier_name"] or worst["category"]
        summary += (
            f"Highest impact: {who} with +{worst['price_change_pct']:.1f}% cost increase "
            f"over {worst['period_label']}."
        )

    return {"alerts": alerts, "headline": headline, "summary": summary}


# ---------------------------------------------------------------------------
# Cash runway advice
# ---------------------------------------------------------------------------

def _cash_runway_alert(a: dict, cash_results: dict) -> dict:
    atype    = a["alert_type"]
    severity = a["severity"]
    runway   = cash_results["runway_days"]
    balance  = cash_results["current_balance"]
    overdue  = cash_results["overdue_exposure"]

    if atype == "critical_runway":
        title = f"Cash runway critically low — {_days_label(runway)} remaining"
        body  = (
            f"At current burn rate, projected cash reserves will be depleted in "
            f"{_days_label(runway)}. Current balance: {balance:,.0f} RON. "
            f"Immediate action is required: accelerate collections on overdue invoices "
            f"({overdue:,.0f} RON outstanding), defer non-critical supplier payments, "
            f"and contact your bank about a short-term credit facility."
        )
        metric = f"{runway}d runway"

    elif atype == "low_runway":
        title = f"Cash runway at {_days_label(runway)} — below safety threshold"
        body  = (
            f"Projected cash runway is {_days_label(runway)}, below the recommended "
            f"60-day buffer. Balance: {balance:,.0f} RON. "
            f"Focus collections on overdue receivables ({overdue:,.0f} RON). "
            f"Review upcoming large supplier payments and negotiate extended terms "
            f"where possible."
        )
        metric = f"{runway}d runway"

    elif atype == "overdue_exposure":
        title = f"Overdue receivables: {overdue:,.0f} RON uncollected"
        body  = (
            f"{overdue:,.0f} RON in receivables is past due and not yet collected. "
            f"This represents real cash that should already be in your account. "
            f"Recommendation: Issue formal demand letters for invoices overdue by >30 days. "
            f"For invoices >60 days overdue, escalate to your collections process or "
            f"consider engaging a collection agency."
        )
        metric = f"{overdue:,.0f} RON"

    elif atype == "consecutive_negative":
        months = a["value"]
        title  = f"{months} consecutive months of projected negative cash flow"
        body   = (
            f"Cash flow projections show {months} consecutive months of net outflows. "
            f"This is typically caused by a mismatch between supplier payment terms "
            f"(short) and customer collection cycles (long). "
            f"Recommendation: Renegotiate supplier payment terms to net-45 or net-60, "
            f"and incentivise early payment from buyers with a 1–2% early-payment discount."
        )
        metric = f"{months} neg. months"
    else:
        title  = a.get("message", "Cash flow alert")
        body   = a.get("message", "")
        metric = ""

    return {
        "title":      title,
        "body":       body,
        "severity":   severity,
        "metric":     metric,
        "alert_type": atype,
    }


def _cash_advice(cash_results: dict) -> dict:
    runway  = cash_results["runway_days"]
    label   = cash_results["runway_label"]
    balance = cash_results["current_balance"]
    overdue = cash_results["overdue_exposure"]

    alerts = [_cash_runway_alert(a, cash_results) for a in cash_results["alerts"]]

    # Always add a baseline status card
    if label == "healthy":
        status_title = f"Cash position healthy — {_days_label(runway)}+ runway"
        status_body  = (
            f"Current cash balance is {balance:,.0f} RON with a projected runway of "
            f"{_days_label(runway)} based on current inflow/outflow trends. "
            f"No immediate liquidity concerns."
        )
        status_sev   = "info"
    elif label == "stable":
        status_title = f"Cash position stable — {_days_label(runway)} runway"
        status_body  = (
            f"Balance is {balance:,.0f} RON. Runway of {_days_label(runway)} is adequate "
            f"but leaves limited buffer. Maintain collections discipline."
        )
        status_sev   = "info"
    else:
        status_title = None  # alerts already cover critical/at_risk

    if status_title:
        alerts.insert(0, {
            "title":      status_title,
            "body":       status_body,
            "severity":   status_sev,
            "metric":     f"{runway}d runway",
            "alert_type": "status",
        })

    if label in ("critical", "at_risk"):
        headline = f"⚠ Cash runway is {label.replace('_', ' ')} — {_days_label(runway)} projected"
    else:
        headline = f"Cash runway: {_days_label(runway)} — {label}"

    summary = (
        f"Current balance: {balance:,.0f} RON. "
        f"Projected runway: {_days_label(runway)} ({label}). "
        f"Overdue receivables: {overdue:,.0f} RON. "
    )
    neg_months = [m for m in cash_results["monthly_summary"]
                  if m["is_projected"] and m["net_ron"] < 0]
    if neg_months:
        summary += f"{len(neg_months)} projected month(s) with negative net cash flow."

    return {"alerts": alerts, "headline": headline, "summary": summary}


# ---------------------------------------------------------------------------
# What-If simulator
# ---------------------------------------------------------------------------

def simulate_price_increase(
    margin_results: dict,
    category: str,
    increase_pct: float,
) -> dict:
    """
    Simulate the impact of a supplier raising prices by `increase_pct`%
    for `category`. Returns a plain-language impact assessment.
    """
    # Find the current trend for this category
    trend = next(
        (t for t in margin_results["trends"] if t["product_category"] == category),
        None,
    )

    # Find monthly data for this category to compute total volume
    cat_data = [t for t in margin_results["trends"] if t["product_category"] == category]
    if not cat_data:
        return {"error": f"No data found for category '{category}'"}

    # Most recent price
    cat_data_sorted = sorted(cat_data, key=lambda x: x["month"])
    recent_price    = cat_data_sorted[-1]["price_ron"]
    new_price       = recent_price * (1 + increase_pct / 100)
    price_delta     = new_price - recent_price

    # Estimate annualised impact
    total_invoices = sum(t.get("invoice_count", 0) for t in cat_data)
    months_covered = len(cat_data)
    avg_invoices_per_month = total_invoices / max(months_covered, 1)
    # Approximate average quantity
    annual_invoices = avg_invoices_per_month * 12

    gross_impact_ron = price_delta * annual_invoices * 10  # rough qty multiplier

    # Recommended selling price adjustment to maintain 15% gross margin
    # cost_new = cost_old * (1 + pct/100); to keep same GM%, raise price proportionally
    selling_price_adj = increase_pct * 0.85  # pass-through ~85% to preserve some margin

    if increase_pct <= 3:
        severity = "info"
        action   = "This increase is within normal inflationary range. No immediate action needed."
    elif increase_pct <= 10:
        severity = "warning"
        action   = (
            f"Consider passing {selling_price_adj:.1f}% of the cost increase through to "
            f"your customers. Negotiate with the supplier for a volume discount to offset."
        )
    else:
        severity = "critical"
        action   = (
            f"A {increase_pct:.0f}% price hike would significantly erode margins. "
            f"Immediately evaluate alternative suppliers for {category}. "
            f"If unavoidable, raise customer prices by at least {selling_price_adj:.1f}% "
            f"and review whether this product line remains viable."
        )

    return {
        "category":            category,
        "increase_pct":        increase_pct,
        "current_price_ron":   round(recent_price, 2),
        "new_price_ron":       round(new_price, 2),
        "price_delta_ron":     round(price_delta, 2),
        "estimated_annual_impact_ron": round(gross_impact_ron, 0),
        "recommended_selling_price_adj_pct": round(selling_price_adj, 1),
        "severity":            severity,
        "headline": (
            f"A {increase_pct:.0f}% price increase on {category} would cost "
            f"~{gross_impact_ron:,.0f} RON/year"
        ),
        "body": (
            f"If {category} supplier prices rise by {increase_pct:.0f}%, the unit cost "
            f"moves from {recent_price:.2f} RON to {new_price:.2f} RON "
            f"(+{price_delta:.2f} RON/unit). "
            f"Estimated annualised gross margin impact: ~{gross_impact_ron:,.0f} RON. "
            f"{action}"
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_advice(
    risk_results:   list[dict],
    margin_results: dict,
    cash_results:   dict,
) -> dict:
    """
    Master function. Consumes all three engine outputs and returns
    structured advisory cards for the Flask UI.
    """
    return {
        "payment_risk": _payment_advice(risk_results),
        "margin":       _margin_advice(margin_results),
        "cash_runway":  _cash_advice(cash_results),
    }
