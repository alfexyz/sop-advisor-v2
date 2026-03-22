"""
S&OP Advisor - Banking Dataset Generator
-----------------------------------------
Generates synthetic open-banking transactions for RomDistrib SA,
covering Jan 2024 – Jun 2025, linked to the invoice purchase dataset.

Two accounts:
  RON current account — main operational account
  EUR current account — for EUR-denominated supplier payments

Transaction types:
  invoice_payment_out  RomDistrib pays a supplier (linked to FACT-XXXX)
  sales_receipt_in     Monthly aggregate customer receipts (resale at +15%)
  salary               Monthly payroll
  rent                 Monthly warehouse / office rent
  utilities            Electricity, gas, internet
  vat_payment          Quarterly TVA remittance to ANAF
  bank_fee             Monthly account maintenance fees
  loan_repayment       Monthly term-loan installment

Reconciliation model for invoice_payment_out:
  65%  exact    — remittance_info contains the exact invoice ID
  15%  batch    — one bank transfer covers 2–4 invoices from the same supplier
  10%  fuzzy    — invoice ID is garbled (missing dashes, lowercase, off-by-one)
  10%  unmatched— generic description; no usable invoice reference

Output files:
  data/banking_accounts.json
  data/banking_transactions.json
"""

import json
import random
import hashlib
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

random.seed(123)

DATA_DIR       = Path(__file__).parent
INVOICES_PATH  = DATA_DIR / "invoices.json"
START_DATE     = date(2024, 1, 1)
END_DATE       = date(2025, 6, 30)
EUR_TO_RON     = 4.97
GROSS_MARGIN   = 0.15

# ---------------------------------------------------------------------------
# RomDistrib SA bank accounts
# ---------------------------------------------------------------------------

ACCOUNTS = [
    {
        "account_id":      "RO49RNCB0000123456789001",
        "owner":           "RomDistrib SA",
        "currency":        "RON",
        "opening_balance": 8_500_000.00,
        "opening_date":    "2024-01-01",
        "bank_name":       "Banca Comerciala Romana SA",
        "bic":             "RNCBROBU",
    },
    {
        "account_id":      "RO49RNCB0000123456789002",
        "owner":           "RomDistrib SA",
        "currency":        "EUR",
        "opening_balance": 80_000.00,
        "opening_date":    "2024-01-01",
        "bank_name":       "Banca Comerciala Romana SA",
        "bic":             "RNCBROBU",
    },
]

ACCOUNT_RON = ACCOUNTS[0]["account_id"]
ACCOUNT_EUR = ACCOUNTS[1]["account_id"]

BANK_CODES = ["RNCB", "BTRL", "INGB", "RZBR", "BRDE", "OTPV", "CECE"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_txn_counter = [0]


def _next_id() -> str:
    _txn_counter[0] += 1
    return f"TXN-{_txn_counter[0]:06d}"


def _company_iban(name: str) -> str:
    """Deterministic plausible IBAN derived from company name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    bank = BANK_CODES[h % len(BANK_CODES)]
    acct = str(h % 10 ** 16).zfill(16)
    return f"RO49{bank}{acct}"


def _garble_invoice_id(invoice_id: str) -> str:
    """Produce a realistic typo/variant of an invoice ID."""
    variants = [
        invoice_id.replace("-", ""),                      # FACT202400043
        invoice_id.replace("-", "/"),                     # FACT/2024/00043
        invoice_id.lower(),                               # fact-2024-00043
        "FC" + invoice_id[4:],                            # FC-2024-00043
        invoice_id + " " + str(random.randint(1, 3)),     # FACT-2024-00043 2
        # shift the numeric suffix by ±1
        invoice_id[:-5] + str(int(invoice_id[-5:]) + random.choice([-1, 1])).zfill(5),
    ]
    return random.choice(variants)


_UNMATCHED_DESCRIPTIONS = [
    "Plata conform contract",
    "Contravaloare marfa",
    "Regularizare sold",
    "Avans furnizor",
    "Plata partiala",
    "Achitare sold restant",
    "Transfer operational",
    "Decontare servicii",
]


# ---------------------------------------------------------------------------
# Invoice payment transactions (outflows)
# ---------------------------------------------------------------------------

def generate_invoice_transactions(invoices: list[dict]) -> list[dict]:
    """
    For each paid invoice, generate an outflow bank transaction from
    RomDistrib SA to the named supplier.

    Match confidence distribution (approximate):
      65% exact     single transaction, exact invoice ID in remittance
      15% batch     one transfer covers 2–4 invoices from the same supplier
      10% fuzzy     invoice ID garbled; engine must fuzzy-match to reconcile
      10% unmatched generic description; only amount + counterparty can match
    """
    paid = [inv for inv in invoices if inv["actual_payment_date"]]

    # Assign a payment type to each invoice
    type_pool = ["exact"] * 65 + ["batch"] * 15 + ["fuzzy"] * 10 + ["unmatched"] * 10
    invoice_types = {inv["invoice_id"]: random.choice(type_pool) for inv in paid}

    # -------------------------------------------------------------------
    # Batch grouping: same supplier, same currency, payment within 5 days
    # -------------------------------------------------------------------
    batch_candidates = [inv for inv in paid
                        if invoice_types[inv["invoice_id"]] == "batch"]
    batch_candidates.sort(key=lambda x: (
        x["currency"], x["supplier_name"], x["actual_payment_date"]
    ))

    batch_groups: list[list[dict]] = []
    batch_used: set[str] = set()

    for i, inv in enumerate(batch_candidates):
        if inv["invoice_id"] in batch_used:
            continue
        group = [inv]
        batch_used.add(inv["invoice_id"])

        for candidate in batch_candidates[i + 1:]:
            if candidate["invoice_id"] in batch_used:
                continue
            same_supplier  = candidate["supplier_name"] == inv["supplier_name"]
            same_currency  = candidate["currency"] == inv["currency"]
            close_in_time  = abs(
                (date.fromisoformat(candidate["actual_payment_date"]) -
                 date.fromisoformat(inv["actual_payment_date"])).days
            ) <= 5
            if same_supplier and same_currency and close_in_time and len(group) < 4:
                group.append(candidate)
                batch_used.add(candidate["invoice_id"])

        batch_groups.append(group)

    transactions = []

    # Generate batch transactions (groups of 2+); solo batches fall through
    # to individual processing below as "exact"
    for group in batch_groups:
        if len(group) == 1:
            # Promote to exact — no point labelling a single invoice "batch"
            invoice_types[group[0]["invoice_id"]] = "exact"
            continue

        total   = round(sum(inv["amount"] for inv in group), 2)
        curr    = group[0]["currency"]
        ids     = [inv["invoice_id"] for inv in group]
        pay_dt  = max(inv["actual_payment_date"] for inv in group)

        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_EUR if curr == "EUR" else ACCOUNT_RON,
            "value_date":          pay_dt,
            "booking_date":        pay_dt,
            "amount":              -total,
            "currency":            curr,
            "direction":           "debit",
            "counterparty_name":   group[0]["supplier_name"],
            "counterparty_iban":   _company_iban(group[0]["supplier_name"]),
            "remittance_info":     "Plata facturi " + ", ".join(ids),
            "transaction_type":    "invoice_payment_out",
            "matched_invoice_ids": ids,
            "match_confidence":    "exact",
        })

    # Generate individual transactions for exact / fuzzy / unmatched
    for inv in paid:
        if inv["invoice_id"] in batch_used and invoice_types[inv["invoice_id"]] == "batch":
            # Already handled in a multi-invoice group above
            continue

        match_type = invoice_types[inv["invoice_id"]]
        curr       = inv["currency"]

        if match_type == "exact":
            remittance   = f"Plata factura {inv['invoice_id']}"
            matched_ids  = [inv["invoice_id"]]
            confidence   = "exact"
        elif match_type == "fuzzy":
            remittance   = f"Plata {_garble_invoice_id(inv['invoice_id'])}"
            matched_ids  = [inv["invoice_id"]]   # ground-truth link preserved
            confidence   = "fuzzy"
        else:  # unmatched
            remittance   = random.choice(_UNMATCHED_DESCRIPTIONS)
            matched_ids  = []
            confidence   = "unmatched"

        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_EUR if curr == "EUR" else ACCOUNT_RON,
            "value_date":          inv["actual_payment_date"],
            "booking_date":        inv["actual_payment_date"],
            "amount":              -round(inv["amount"], 2),
            "currency":            curr,
            "direction":           "debit",
            "counterparty_name":   inv["supplier_name"],
            "counterparty_iban":   _company_iban(inv["supplier_name"]),
            "remittance_info":     remittance,
            "transaction_type":    "invoice_payment_out",
            "matched_invoice_ids": matched_ids,
            "match_confidence":    confidence,
        })

    return transactions


# ---------------------------------------------------------------------------
# Sales receipts — monthly aggregate inflows
# ---------------------------------------------------------------------------

def generate_sales_receipts(invoices: list[dict]) -> list[dict]:
    """
    Aggregate monthly inflows representing customer payments to RomDistrib SA.
    Derived from total monthly purchase value * (1 + GROSS_MARGIN), arriving
    ~30 days after the purchase month (simulating the resale cycle).
    Not linked to individual invoices — no remittance reference to reconcile.
    """
    from collections import defaultdict

    monthly: dict[tuple, dict[str, float]] = defaultdict(lambda: {"RON": 0.0, "EUR": 0.0})

    for inv in invoices:
        if not inv["actual_payment_date"]:
            continue
        pay_dt    = date.fromisoformat(inv["actual_payment_date"])
        month_key = (pay_dt.year, pay_dt.month)
        monthly[month_key][inv["currency"]] += inv["amount"]

    transactions = []

    for (year, month), totals in monthly.items():
        for curr, total in totals.items():
            if total == 0:
                continue
            receipt_dt = date(year, month, 1) + timedelta(days=random.randint(28, 38))
            if receipt_dt > END_DATE:
                continue
            receipt_amount = round(total * (1 + GROSS_MARGIN) * random.uniform(0.90, 1.10), 2)
            month_label    = date(year, month, 1).strftime("%B %Y")

            transactions.append({
                "transaction_id":      _next_id(),
                "account_id":          ACCOUNT_EUR if curr == "EUR" else ACCOUNT_RON,
                "value_date":          receipt_dt.isoformat(),
                "booking_date":        receipt_dt.isoformat(),
                "amount":              receipt_amount,
                "currency":            curr,
                "direction":           "credit",
                "counterparty_name":   "Clienti diversi",
                "counterparty_iban":   None,
                "remittance_info":     f"Incasare clienti {curr} {month_label}",
                "transaction_type":    "sales_receipt_in",
                "matched_invoice_ids": [],
                "match_confidence":    "n/a",
            })

    return transactions


# ---------------------------------------------------------------------------
# Operational transactions — fixed monthly overhead
# ---------------------------------------------------------------------------

def generate_operational_transactions() -> list[dict]:
    """
    Monthly recurring costs unrelated to the invoice dataset:
    salaries, rent, utilities, loan repayment, bank fees, quarterly VAT.
    """
    transactions = []
    current = START_DATE

    while current <= END_DATE:
        y, m     = current.year, current.month
        last_day = monthrange(y, m)[1]
        label    = current.strftime("%B %Y")

        # Salaries — 3rd of month
        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_RON,
            "value_date":          date(y, m, 3).isoformat(),
            "booking_date":        date(y, m, 3).isoformat(),
            "amount":              -round(random.uniform(245_000, 285_000), 2),
            "currency":            "RON",
            "direction":           "debit",
            "counterparty_name":   "Salariati RomDistrib SA",
            "counterparty_iban":   None,
            "remittance_info":     f"Salarii {label}",
            "transaction_type":    "salary",
            "matched_invoice_ids": [],
            "match_confidence":    "n/a",
        })

        # Rent — 1st of month
        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_RON,
            "value_date":          date(y, m, 1).isoformat(),
            "booking_date":        date(y, m, 1).isoformat(),
            "amount":              -35_000.00,
            "currency":            "RON",
            "direction":           "debit",
            "counterparty_name":   "Imobil Industrial Park Ploiesti SRL",
            "counterparty_iban":   _company_iban("Imobil Industrial Park Ploiesti SRL"),
            "remittance_info":     f"Chirie spatiu industrial {label}",
            "transaction_type":    "rent",
            "matched_invoice_ids": [],
            "match_confidence":    "n/a",
        })

        # Utilities — 15th of month
        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_RON,
            "value_date":          date(y, m, 15).isoformat(),
            "booking_date":        date(y, m, 15).isoformat(),
            "amount":              -round(random.uniform(7_000, 13_000), 2),
            "currency":            "RON",
            "direction":           "debit",
            "counterparty_name":   "E.ON Energie Romania SA",
            "counterparty_iban":   _company_iban("E.ON Energie Romania SA"),
            "remittance_info":     f"Factura utilitati {label}",
            "transaction_type":    "utilities",
            "matched_invoice_ids": [],
            "match_confidence":    "n/a",
        })

        # Loan repayment — 25th of month
        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_RON,
            "value_date":          date(y, m, min(25, last_day)).isoformat(),
            "booking_date":        date(y, m, min(25, last_day)).isoformat(),
            "amount":              -48_500.00,
            "currency":            "RON",
            "direction":           "debit",
            "counterparty_name":   "Banca Comerciala Romana SA",
            "counterparty_iban":   "RO49RNCB0000000000000099",
            "remittance_info":     f"Rata credit 2022/045 {label}",
            "transaction_type":    "loan_repayment",
            "matched_invoice_ids": [],
            "match_confidence":    "n/a",
        })

        # Bank fees RON — last day of month
        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_RON,
            "value_date":          date(y, m, last_day).isoformat(),
            "booking_date":        date(y, m, last_day).isoformat(),
            "amount":              -round(random.uniform(450, 620), 2),
            "currency":            "RON",
            "direction":           "debit",
            "counterparty_name":   "Banca Comerciala Romana SA",
            "counterparty_iban":   "RO49RNCB0000000000000099",
            "remittance_info":     "Comisioane bancare lunare",
            "transaction_type":    "bank_fee",
            "matched_invoice_ids": [],
            "match_confidence":    "n/a",
        })

        # Bank fees EUR — last day of month
        transactions.append({
            "transaction_id":      _next_id(),
            "account_id":          ACCOUNT_EUR,
            "value_date":          date(y, m, last_day).isoformat(),
            "booking_date":        date(y, m, last_day).isoformat(),
            "amount":              -round(random.uniform(40, 65), 2),
            "currency":            "EUR",
            "direction":           "debit",
            "counterparty_name":   "Banca Comerciala Romana SA",
            "counterparty_iban":   "RO49RNCB0000000000000099",
            "remittance_info":     "Comisioane cont valutar",
            "transaction_type":    "bank_fee",
            "matched_invoice_ids": [],
            "match_confidence":    "n/a",
        })

        # Quarterly VAT payment to ANAF — 25th of Mar / Jun / Sep / Dec
        if m in (3, 6, 9, 12):
            quarter = {3: "I", 6: "II", 9: "III", 12: "IV"}[m]
            transactions.append({
                "transaction_id":      _next_id(),
                "account_id":          ACCOUNT_RON,
                "value_date":          date(y, m, min(25, last_day)).isoformat(),
                "booking_date":        date(y, m, min(25, last_day)).isoformat(),
                "amount":              -round(random.uniform(165_000, 220_000), 2),
                "currency":            "RON",
                "direction":           "debit",
                "counterparty_name":   "ANAF - Agentia Nationala de Administrare Fiscala",
                "counterparty_iban":   "RO02TREZ7035069XXX006464",
                "remittance_info":     f"TVA trimestrul {quarter} {y}",
                "transaction_type":    "vat_payment",
                "matched_invoice_ids": [],
                "match_confidence":    "n/a",
            })

        # Advance to next month
        current = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)

    return transactions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(INVOICES_PATH, encoding="utf-8") as f:
        invoices = json.load(f)
    print(f"Loaded {len(invoices)} invoices.")

    print("Generating invoice payment transactions...")
    invoice_txns = generate_invoice_transactions(invoices)

    print("Generating sales receipt transactions...")
    sales_txns = generate_sales_receipts(invoices)

    print("Generating operational transactions...")
    op_txns = generate_operational_transactions()

    all_txns = sorted(invoice_txns + sales_txns + op_txns, key=lambda x: x["value_date"])

    # Write accounts
    accounts_path = DATA_DIR / "banking_accounts.json"
    with open(accounts_path, "w", encoding="utf-8") as f:
        json.dump(ACCOUNTS, f, ensure_ascii=False, indent=2)
    print(f"\nAccounts      → {accounts_path}")

    # Write transactions
    txns_path = DATA_DIR / "banking_transactions.json"
    with open(txns_path, "w", encoding="utf-8") as f:
        json.dump(all_txns, f, ensure_ascii=False, indent=2)
    print(f"Transactions  → {txns_path}  ({len(all_txns)} records)")

    # --- Stats ---
    total_debits  = sum(t["amount"] for t in all_txns if t["amount"] < 0)
    total_credits = sum(t["amount"] for t in all_txns if t["amount"] > 0)

    by_type: dict[str, int] = {}
    for t in all_txns:
        by_type[t["transaction_type"]] = by_type.get(t["transaction_type"], 0) + 1

    exact_n     = sum(1 for t in invoice_txns if t["match_confidence"] == "exact")
    fuzzy_n     = sum(1 for t in invoice_txns if t["match_confidence"] == "fuzzy")
    unmatched_n = sum(1 for t in invoice_txns if t["match_confidence"] == "unmatched")
    n_inv       = len(invoice_txns)

    ron_txns = [t for t in all_txns if t["account_id"] == ACCOUNT_RON]
    eur_txns = [t for t in all_txns if t["account_id"] == ACCOUNT_EUR]
    ron_close = ACCOUNTS[0]["opening_balance"] + sum(t["amount"] for t in ron_txns)
    eur_close = ACCOUNTS[1]["opening_balance"] + sum(t["amount"] for t in eur_txns)

    print(f"\n--- Banking dataset summary ---")
    print(f"  Total transactions : {len(all_txns)}")
    print(f"  Total debits       : {total_debits:>14,.0f}")
    print(f"  Total credits      : {total_credits:>14,.0f}")
    print(f"\n  By type:")
    for tt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {tt:<25} {cnt:>5}")
    print(f"\n  Invoice reconciliation ({n_inv} payment transactions):")
    print(f"    exact     : {exact_n:>5}  ({exact_n / n_inv * 100:.1f}%)")
    print(f"    fuzzy     : {fuzzy_n:>5}  ({fuzzy_n / n_inv * 100:.1f}%)")
    print(f"    unmatched : {unmatched_n:>5}  ({unmatched_n / n_inv * 100:.1f}%)")
    print(f"    batch     :        (remaining, grouped)")
    print(f"\n  Closing balances (2025-06-30):")
    print(f"    RON account : {ron_close:>14,.2f} RON")
    print(f"    EUR account : {eur_close:>14,.2f} EUR")


if __name__ == "__main__":
    main()
