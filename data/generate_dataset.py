"""
S&OP Advisor - Dataset Generator
Generates ~2000 realistic e-Factura-style invoices for a fictional Romanian
mid-size distributor over 18 months (Jan 2024 - Jun 2025).

Patterns baked in:
- Some suppliers are consistently late payers
- Some product categories have rising unit costs (margin erosion)
- Seasonal demand spikes (Q4 retail surge, Q2 construction)
- A few large clients with erratic payment behaviour
"""

import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

SUPPLIERS = [
    {"name": "Electro Impex SRL",       "category": "Electronics",    "late_bias": 0.10, "price_trend": +0.005},
    {"name": "TechRom Distribution SA",  "category": "Electronics",    "late_bias": 0.25, "price_trend": +0.012},
    {"name": "MetalCraft Industries SRL","category": "Raw Materials",  "late_bias": 0.05, "price_trend": +0.018},
    {"name": "Agro Supply SRL",          "category": "Food & Beverage","late_bias": 0.08, "price_trend": +0.003},
    {"name": "Construct Plus SA",        "category": "Construction",   "late_bias": 0.15, "price_trend": +0.009},
    {"name": "ChemTrade Romania SRL",    "category": "Chemicals",      "late_bias": 0.20, "price_trend": +0.007},
    {"name": "PackPro SRL",              "category": "Packaging",      "late_bias": 0.05, "price_trend": +0.002},
    {"name": "TransLog Romania SA",      "category": "Logistics",      "late_bias": 0.30, "price_trend": +0.001},
    {"name": "Furniture Depot SRL",      "category": "Furniture",      "late_bias": 0.12, "price_trend": +0.004},
    {"name": "MedSupply SRL",            "category": "Medical",        "late_bias": 0.03, "price_trend": +0.006},
]

BUYERS = [
    {"name": "Hypermarket Auchan RO",    "late_bias": 0.05,  "avg_amount": 45000, "amount_var": 0.4},
    {"name": "Dedeman SA",               "late_bias": 0.08,  "avg_amount": 38000, "amount_var": 0.35},
    {"name": "Altex Romania SRL",        "late_bias": 0.10,  "avg_amount": 22000, "amount_var": 0.5},
    {"name": "Profi Rom Food SRL",       "late_bias": 0.15,  "avg_amount": 12000, "amount_var": 0.6},
    {"name": "Leroy Merlin Romania SRL", "late_bias": 0.07,  "avg_amount": 31000, "amount_var": 0.3},
    {"name": "Grupul Romstal SA",        "late_bias": 0.35,  "avg_amount": 18000, "amount_var": 0.8},  # erratic
    {"name": "MegaImage SRL",            "late_bias": 0.12,  "avg_amount": 9500,  "amount_var": 0.45},
    {"name": "Arabesque SRL",            "late_bias": 0.40,  "avg_amount": 27000, "amount_var": 0.7},  # erratic
    {"name": "Kaufland Romania SCS",     "late_bias": 0.04,  "avg_amount": 52000, "amount_var": 0.25},
    {"name": "Carrefour Romania SA",     "late_bias": 0.06,  "avg_amount": 41000, "amount_var": 0.3},
    {"name": "Selgros Cash&Carry SRL",   "late_bias": 0.18,  "avg_amount": 15000, "amount_var": 0.55},
    {"name": "Bricostore Romania SA",    "late_bias": 0.22,  "avg_amount": 11000, "amount_var": 0.6},
]

PRODUCT_CATEGORIES = {
    "Electronics":    {"base_price": 850,  "unit": "bucată",    "seasonal_peak": [10, 11, 12]},
    "Raw Materials":  {"base_price": 420,  "unit": "tonă",      "seasonal_peak": [3, 4, 5, 6]},
    "Food & Beverage":{"base_price": 95,   "unit": "cutie",     "seasonal_peak": [11, 12, 1]},
    "Construction":   {"base_price": 310,  "unit": "m²",        "seasonal_peak": [4, 5, 6, 7]},
    "Chemicals":      {"base_price": 180,  "unit": "litru",     "seasonal_peak": [3, 4, 9, 10]},
    "Packaging":      {"base_price": 55,   "unit": "bax",       "seasonal_peak": [9, 10, 11]},
    "Logistics":      {"base_price": 2200, "unit": "transport", "seasonal_peak": [11, 12]},
    "Furniture":      {"base_price": 1400, "unit": "bucată",    "seasonal_peak": [3, 4, 8, 9]},
    "Medical":        {"base_price": 230,  "unit": "set",       "seasonal_peak": [1, 2, 10, 11]},
}

VAT_RATE = 0.19  # 19% standard Romanian VAT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2025, 6, 30)
TOTAL_DAYS = (END_DATE - START_DATE).days

def random_date_in_range(start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def months_since_start(d: datetime) -> float:
    return (d - START_DATE).days / 30.44

def seasonal_multiplier(month: int, peak_months: list[int]) -> float:
    """Returns 1.0–1.6 in peak months, 0.7–1.0 otherwise."""
    if month in peak_months:
        return 1.0 + random.uniform(0.2, 0.6)
    return random.uniform(0.7, 1.0)

def price_with_trend(base_price: float, monthly_trend: float, issue_date: datetime) -> float:
    """Apply compounding monthly price increase since START_DATE."""
    months = months_since_start(issue_date)
    trended = base_price * math.pow(1 + monthly_trend, months)
    # Add ±5% noise
    noise = random.uniform(-0.05, 0.05)
    return round(trended * (1 + noise), 2)

def payment_delay_days(buyer_late_bias: float, supplier_late_bias: float,
                       amount: float, buyer_avg_amount: float) -> int | None:
    """
    Returns number of days past due_date the payment arrived,
    or None if invoice is still unpaid (outstanding).
    """
    # Combined late probability
    combined_bias = min(0.9, buyer_late_bias + supplier_late_bias * 0.5)

    # Large invoices are harder to pay on time
    size_factor = min(2.0, amount / max(buyer_avg_amount, 1))
    adjusted_bias = min(0.95, combined_bias * (0.8 + 0.4 * size_factor))

    if random.random() < adjusted_bias:
        # Late payment: skewed distribution (most ~1–30 days, some 31–90)
        if random.random() < 0.6:
            delay = random.randint(1, 30)
        elif random.random() < 0.8:
            delay = random.randint(31, 60)
        else:
            delay = random.randint(61, 120)
        return delay
    return 0  # on time

def currency_for_category(category: str) -> str:
    # Raw materials and logistics often invoiced in EUR
    if category in ("Raw Materials", "Logistics", "Chemicals"):
        return "EUR" if random.random() < 0.45 else "RON"
    return "EUR" if random.random() < 0.10 else "RON"

# ---------------------------------------------------------------------------
# Invoice generator
# ---------------------------------------------------------------------------

def generate_invoices(n: int = 2000) -> list[dict]:
    invoices = []
    cutoff_for_unpaid = datetime(2025, 5, 15)  # recent invoices may still be outstanding

    for i in range(n):
        invoice_id = f"FACT-2024-{i+1:05d}"

        # Pick supplier and buyer
        supplier = random.choice(SUPPLIERS)
        buyer = random.choice(BUYERS)
        category = supplier["category"]
        cat_info = PRODUCT_CATEGORIES[category]

        # Issue date spread across 18 months, weighted slightly toward Q4 peaks
        issue_date = random_date_in_range(START_DATE, END_DATE)
        payment_terms_days = random.choice([14, 30, 30, 30, 45, 60])  # net terms
        due_date = issue_date + timedelta(days=payment_terms_days)

        # Pricing
        currency = currency_for_category(category)
        unit_price_ron = price_with_trend(cat_info["base_price"],
                                          supplier["price_trend"],
                                          issue_date)
        # Rough EUR→RON rate ~4.97, with slight drift
        eur_rate = 4.97 + random.uniform(-0.05, 0.05)
        unit_price = round(unit_price_ron / eur_rate, 2) if currency == "EUR" else unit_price_ron

        # Quantity with seasonal boost
        season_mult = seasonal_multiplier(issue_date.month, cat_info["seasonal_peak"])
        base_qty = random.randint(1, 80)
        quantity = max(1, int(base_qty * season_mult))

        amount = round(unit_price * quantity, 2)
        vat_amount = round(amount * VAT_RATE, 2)

        # Payment date
        delay = payment_delay_days(buyer["late_bias"],
                                   supplier["late_bias"],
                                   amount,
                                   buyer["avg_amount"])

        if due_date > cutoff_for_unpaid and random.random() < 0.25:
            # Recent invoice, plausibly still outstanding
            actual_payment_date = None
        elif delay is None:
            actual_payment_date = None
        else:
            actual_payment_date = (due_date + timedelta(days=delay)).strftime("%Y-%m-%d")

        invoices.append({
            "invoice_id": invoice_id,
            "supplier_name": supplier["name"],
            "buyer_name": buyer["name"],
            "issue_date": issue_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "actual_payment_date": actual_payment_date,
            "amount": amount,
            "vat_amount": vat_amount,
            "currency": currency,
            "product_category": category,
            "quantity": quantity,
            "unit_price": unit_price,
            "payment_terms_days": payment_terms_days,
        })

    # Sort by issue date
    invoices.sort(key=lambda x: x["issue_date"])
    return invoices

# ---------------------------------------------------------------------------
# UBL 2.1-inspired envelope
# ---------------------------------------------------------------------------

def wrap_as_efactura(invoice: dict) -> dict:
    """
    Wraps a flat invoice dict in a lightweight UBL 2.1-inspired structure,
    as used by ANAF's e-Factura system.
    """
    return {
        "Invoice": {
            "xmlns": "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2",
            "xmlns:cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
            "xmlns:cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
            "cbc:ID": invoice["invoice_id"],
            "cbc:IssueDate": invoice["issue_date"],
            "cbc:DueDate": invoice["due_date"],
            "cbc:DocumentCurrencyCode": invoice["currency"],
            "cac:AccountingSupplierParty": {
                "cac:Party": {"cac:PartyName": {"cbc:Name": invoice["supplier_name"]}}
            },
            "cac:AccountingCustomerParty": {
                "cac:Party": {"cac:PartyName": {"cbc:Name": invoice["buyer_name"]}}
            },
            "cac:InvoiceLine": {
                "cbc:InvoicedQuantity": invoice["quantity"],
                "cbc:UnitCode": PRODUCT_CATEGORIES[invoice["product_category"]]["unit"],
                "cbc:LineExtensionAmount": invoice["amount"],
                "cac:Item": {"cbc:Description": invoice["product_category"]},
                "cac:Price": {"cbc:PriceAmount": invoice["unit_price"]},
            },
            "cac:TaxTotal": {
                "cbc:TaxAmount": invoice["vat_amount"],
                "cac:TaxSubtotal": {
                    "cbc:Percent": VAT_RATE * 100,
                    "cac:TaxCategory": {"cbc:ID": "S", "cbc:Name": "Standard Rate"},
                },
            },
            "cac:LegalMonetaryTotal": {
                "cbc:TaxExclusiveAmount": invoice["amount"],
                "cbc:TaxInclusiveAmount": round(invoice["amount"] + invoice["vat_amount"], 2),
                "cbc:PayableAmount": round(invoice["amount"] + invoice["vat_amount"], 2),
            },
            # Non-UBL extension: payment tracking (internal use only)
            "_sop_ext": {
                "actual_payment_date": invoice["actual_payment_date"],
                "payment_terms_days": invoice["payment_terms_days"],
            },
        }
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent
    n = 2000

    print(f"Generating {n} e-Factura invoices...")
    invoices_flat = generate_invoices(n)

    # --- flat JSON (convenient for ML) ---
    flat_path = out_dir / "invoices.json"
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(invoices_flat, f, ensure_ascii=False, indent=2)
    print(f"  Flat dataset  → {flat_path}  ({len(invoices_flat)} records)")

    # --- UBL-wrapped JSON (realistic e-Factura format) ---
    wrapped = [wrap_as_efactura(inv) for inv in invoices_flat]
    ubl_path = out_dir / "invoices_ubl.json"
    with open(ubl_path, "w", encoding="utf-8") as f:
        json.dump(wrapped, f, ensure_ascii=False, indent=2)
    print(f"  UBL envelope  → {ubl_path}  ({len(wrapped)} records)")

    # --- Quick sanity stats ---
    total_amount    = sum(i["amount"] for i in invoices_flat)
    late_count      = sum(1 for i in invoices_flat
                          if i["actual_payment_date"] and i["due_date"]
                          and i["actual_payment_date"] > i["due_date"])
    unpaid_count    = sum(1 for i in invoices_flat if i["actual_payment_date"] is None)
    eur_count       = sum(1 for i in invoices_flat if i["currency"] == "EUR")
    categories      = {}
    for inv in invoices_flat:
        categories[inv["product_category"]] = categories.get(inv["product_category"], 0) + 1

    print(f"\n--- Dataset summary ---")
    print(f"  Date range    : {invoices_flat[0]['issue_date']} → {invoices_flat[-1]['issue_date']}")
    print(f"  Total invoiced: {total_amount:,.0f} (mixed RON/EUR)")
    print(f"  Late payments : {late_count} ({late_count/n*100:.1f}%)")
    print(f"  Unpaid/pending: {unpaid_count} ({unpaid_count/n*100:.1f}%)")
    print(f"  EUR invoices  : {eur_count} ({eur_count/n*100:.1f}%)")
    print(f"  By category   :")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {cnt:>4} invoices")


if __name__ == "__main__":
    main()
