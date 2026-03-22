"""
S&OP Advisor — Flask Application
---------------------------------
Runs the full engine pipeline once at startup (cached in memory),
then serves the dashboard and What-If API endpoint.

Routes
------
GET  /                          Dashboard
GET  /api/whatif                ?category=&increase=   What-If JSON
GET  /api/refresh               Re-run engines, return status JSON
"""

import json
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, request

from engine import payment_risk, margin_monitor, cash_runway
from advisor.advisor import generate_advice, simulate_price_increase

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "data" / "invoices.json"

# ---------------------------------------------------------------------------
# Engine cache — populated once at startup
# ---------------------------------------------------------------------------

_cache: dict = {}


def _run_engines() -> dict:
    t0 = time.time()
    with open(DATA_PATH, encoding="utf-8") as f:
        invoices = json.load(f)

    risk   = payment_risk.run(invoices)
    margin = margin_monitor.run(invoices)
    cash   = cash_runway.run(invoices, payment_risk_results=risk)
    cards  = generate_advice(risk, margin, cash)

    # Distil what the UI needs from the margin trends
    categories = sorted({t["product_category"] for t in margin["trends"]})

    # Monthly trend data per category (for Chart.js)
    trend_by_cat: dict[str, list] = {}
    for t in sorted(margin["trends"], key=lambda x: x["month"]):
        trend_by_cat.setdefault(t["product_category"], []).append({
            "month": t["month"],
            "price": t["price_ron"],
        })

    # Cash projection data (for Chart.js)
    cash_projection = cash["projected_balance"][:90]

    elapsed = round(time.time() - t0, 1)
    print(f"  [app] Engines ready in {elapsed}s")

    return {
        "cards":          cards,
        "categories":     categories,
        "trend_by_cat":   trend_by_cat,
        "cash_projection": cash_projection,
        "cash_summary":   {
            "runway_days":     cash["runway_days"],
            "runway_label":    cash["runway_label"],
            "current_balance": cash["current_balance"],
            "overdue":         cash["overdue_exposure"],
            "today":           cash["today"],
        },
        "margin_results": margin,   # kept for What-If simulator
        "elapsed":        elapsed,
    }


def get_cache() -> dict:
    if not _cache:
        _cache.update(_run_engines())
    return _cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    data = get_cache()
    return render_template(
        "dashboard.html",
        cards          = data["cards"],
        categories     = data["categories"],
        trend_by_cat   = json.dumps(data["trend_by_cat"]),
        cash_projection= json.dumps(data["cash_projection"]),
        cash_summary   = data["cash_summary"],
        elapsed        = data["elapsed"],
    )


@app.route("/api/whatif")
def api_whatif():
    category     = request.args.get("category", "")
    try:
        increase = float(request.args.get("increase", 10))
    except ValueError:
        return jsonify({"error": "increase must be a number"}), 400

    increase = max(0.1, min(increase, 100))
    data     = get_cache()
    result   = simulate_price_increase(data["margin_results"], category, increase)
    return jsonify(result)


@app.route("/api/refresh")
def api_refresh():
    _cache.clear()
    data = get_cache()
    return jsonify({"status": "ok", "elapsed": data["elapsed"]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Warming up engines…")
    get_cache()
    print("Starting Flask on http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
