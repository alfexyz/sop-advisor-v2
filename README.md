# S&OP Advisor v2

A financial intelligence dashboard for mid-market distributors. Combines ML-based payment risk prediction with supplier margin monitoring and cash runway projection, then synthesizes everything into plain-language advisory cards for financial leaders.

Built as a hackathon project around a fictional Romanian distributor (RomDistrib SA) using 2,000 synthetic e-Factura-style invoices and open banking data.

---

## Features

- **Payment Risk Engine** — XGBoost classifier predicts which invoices will be paid late (>30 days), with SHAP-derived plain-language explanations per buyer
- **Margin Monitor** — Detects supplier cost inflation (warning ≥5%, critical ≥12%) using rolling averages and trend acceleration
- **Cash Runway Engine** — 90-day forward cash projection using real open banking balances with payment-risk-adjusted inflow discounting
- **Advisory Layer** — Synthesizes engine outputs into actionable alert cards (critical / warning / info)
- **What-If Simulator** — Interactive: "What if supplier X raises prices by Y%?" → estimated annual impact + recommended selling-price adjustment
- **Interactive Dashboard** — Chart.js trend charts, togglable series, live refresh

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Python, Flask |
| ML | XGBoost, SHAP, scikit-learn |
| Data | Pandas, NumPy |
| Frontend | Bootstrap 5, Chart.js 4, Bootstrap Icons |
| Data format | JSON (invoices, banking) |

---

## Project Structure

```
sop-advisor-v2/
├── app.py                        # Flask app, routes, startup cache
├── advisor/
│   └── advisor.py                # Advisory synthesis layer
├── engine/
│   ├── payment_risk.py           # XGBoost payment prediction + SHAP
│   ├── margin_monitor.py         # Supplier cost trend detection
│   └── cash_runway.py            # 90-day cash projection
├── data/
│   ├── invoices.json             # 2,000 synthetic invoices (Jan 2024 – Jun 2025)
│   ├── invoices_ubl.json         # UBL-XML format invoices
│   ├── banking_accounts.json     # Open banking account balances
│   ├── banking_transactions.json # Open banking transaction history
│   ├── generate_dataset.py       # Invoice dataset generator
│   └── generate_banking_dataset.py
└── templates/
    └── dashboard.html            # Single-page dashboard UI
```

---

## Getting Started

### Install dependencies

```bash
pip install flask pandas numpy xgboost scikit-learn shap
```

### Run

```bash
python app.py
```

Open `http://localhost:5000`. On startup the app trains the ML models and warms up all three engines (~5–10 seconds). Results are cached in memory; use the **Refresh** button or `GET /api/refresh` to re-run.

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Render dashboard |
| `/api/whatif?category=X&increase=Y` | GET | Simulate a supplier price increase |
| `/api/refresh` | GET | Re-run all engines and clear cache |

**What-If example response:**
```json
{
  "category": "Electronics",
  "increase_pct": 10,
  "current_price_ron": 850.00,
  "new_price_ron": 935.00,
  "price_delta_ron": 85.00,
  "estimated_annual_impact_ron": 102000,
  "recommended_selling_price_adj_pct": 8.5,
  "severity": "warning",
  "headline": "A 10% price increase on Electronics would cost ~102,000 RON/year"
}
```

---

## Dataset

2,000 synthetic invoices with baked-in patterns for realistic ML training:

| Stat | Value |
|------|-------|
| Date range | Jan 2024 – Jun 2025 |
| Suppliers | 10 (with varying late-payment bias) |
| Buyers | 12 (with varying payment discipline) |
| Product categories | 9 |
| Currency | RON and EUR (FX rate: 4.97) |
| VAT rate | 19% (Romanian standard) |

Categories: Electronics, Raw Materials, Food & Beverage, Construction, Chemicals, Packaging, Logistics, Furniture, Medical

---

## Engine Details

### Payment Risk
- XGBoost classifier (200 estimators, max_depth=5, lr=0.05) trained on 12 features
- Alert threshold: ≥55% late probability
- SHAP values translated to human-readable risk factors (e.g., "their historical late-payment rate is 45%")
- Secondary XGBoost regressor estimates days-late when probability ≥40%

### Margin Monitor
- 6-month baseline vs. 3-month recent rolling average
- Thresholds: warning ≥5%, critical ≥12%, spike ≥20% single-month jump
- Detects accelerating trends (recent slope > 2× early slope)

### Cash Runway
- Sources real balance from `banking_accounts.json` if available; falls back to 500,000 RON
- Discounts future inflows by `payment_risk_probability × 0.4`
- Runway classification: critical ≤30d, at-risk 31–60d, stable 61–90d, healthy >90d
