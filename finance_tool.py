# finance_tool.py
# A single-file backend pipeline for bank CSVs:
# - Load/clean -> categorize -> detect recurring -> analyze -> charts -> SQLite upsert

import os
import re
import json
import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt














# ========= 1) LOAD CSV (skip bank's top summary lines) =========
CSV_PATH = "data/stmt.csv"  # <-- change to your file if needed
df = pd.read_csv(CSV_PATH, skiprows=6)

print("\n--- First 5 rows (raw) ---")
print(df.head())
print("\n--- Column names (raw) ---")
print(df.columns)
















# ========= 2) CLEAN & NORMALIZE =========
# Strip weird whitespace/characters
df["Date"] = df["Date"].astype(str).str.replace("\u200b", "", regex=False).str.strip()
df["Amount"] = df["Amount"].astype(str).str.replace(",", "").str.strip()
if "Running Bal." in df.columns:
    df["Running Bal."] = df["Running Bal."].astype(str).str.replace(",", "").str.strip()

# Parse typed columns
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
if "Running Bal." in df.columns:
    df["Running Bal."] = pd.to_numeric(df["Running Bal."], errors="coerce")

# Drop summary lines like "Beginning balance..." / "Ending balance..."
summary_mask = df["Description"].astype(str).str.contains(
    r"^(?:Beginning balance|Ending balance)", case=False, na=False
)
df = df[~summary_mask].copy()

# Derived fields
df["Type"] = np.where(df["Amount"] < 0, "Debit", "Credit")
df["Month"] = df["Date"].dt.to_period("M").astype(str)

# Merchant simplifier (normalize noisy descriptions)
def simplify_merchant(desc: str) -> str:
    s = str(desc).upper()
    s = re.sub(r"#\d+", "", s)                # remove store numbers (#1234)
    s = re.sub(r"\b\d{2}/\d{2}\b", "", s)     # inline dates like 12/31
    s = re.sub(r"\s{2,}", " ", s).strip()     # collapse spaces
    # common aliases / cleanups
    s = s.replace("CHICK-FIL-A", "CHICK FIL A")
    s = s.replace("AMAZON MARKETPLACE PMTS", "AMAZON")
    s = s.replace("PAYROLL DEP", "PAYROLL")
    return s

df["Merchant"] = df["Description"].apply(simplify_merchant)

print("\n--- Cleaned sample (first 8) ---")
print(df[["Date","Description","Amount","Type","Month","Merchant"]].head(8))
print("\nNull dates count:", df["Date"].isna().sum())
print("Unique months seen:", df["Month"].nunique())













# ========= 3) RULE-BASED CATEGORIES =========
RULES = {
    "WALMART": "Groceries/Household",
    "KROGER": "Groceries/Household",
    "DOLLAR GENERAL": "Groceries/Household",
    "BP": "Gas",
    "SHELL": "Gas",
    "CHICK FIL A": "Restaurants/Fast Food",
    "MCDONALD": "Restaurants/Fast Food",
    "TRACTOR SUPPLY": "Shopping/Hardware",
    "WALGREENS": "Pharmacy/Health",
    "AMAZON": "Shopping/Online",
    "NETFLIX": "Entertainment/Subscriptions",
    "SPOTIFY": "Entertainment/Subscriptions",
    "GEORGIA POWER": "Utilities",
    "VZWRLSS": "Utilities/Phone",
    "AT&T": "Utilities/Phone",
    "RENT": "Housing/Rent",
    "PAYROLL": "Income",
    "TRANSFER DAILY PAY": "Income",
    "ZELLE PAYMENT": "Transfers",
    "ZELLE TRANSFER": "Transfers",
}

def rule_category(merchant: str) -> str | None:
    if not isinstance(merchant, str):
        return None
    u = merchant.upper()
    # strict startswith first
    for k, v in RULES.items():
        if u.startswith(k):
            return v
    # then contains
    for k, v in RULES.items():
        if k in u:
            return v
    return None

def apply_rule_categories(df: pd.DataFrame) -> pd.DataFrame:
    if "Category" not in df.columns:
        df["Category"] = None
    mask = df["Category"].isna()
    df.loc[mask, "Category"] = df.loc[mask, "Merchant"].apply(rule_category)
    # fallback for anything unknown (can refine later with GPT)
    df["Category"] = df["Category"].fillna("Other")
    return df

df = apply_rule_categories(df)













# ========= 4) RECURRING DETECTION (subscriptions/bills) =========
def detect_recurring(df: pd.DataFrame, window_days=35, amt_tol=2.0) -> pd.DataFrame:
    """
    Flags merchants that look roughly monthly with similar amounts.
    Sets df['Recurring'] = 1/0.
    """
    dfd = df[df["Type"] == "Debit"].copy()
    dfd["AbsAmount"] = dfd["Amount"].abs()

    recurring_merchants = set()
    for merch, grp in dfd.groupby("Merchant"):
        grp = grp.sort_values("Date")
        if len(grp) < 3:
            continue
        dates = grp["Date"].values
        amts  = grp["AbsAmount"].values
        ok_int = ok_amt = 0
        for i in range(1, len(grp)):
            delta_days = (dates[i] - dates[i-1]).astype("timedelta64[D]").astype(int)
            if abs(delta_days - 30) <= window_days:
                ok_int += 1
            if abs(amts[i] - amts[i-1]) <= amt_tol:
                ok_amt += 1
        if ok_int >= 2 and ok_amt >= 2:
            recurring_merchants.add(merch)

    df["Recurring"] = df["Merchant"].isin(recurring_merchants).astype(int)
    return df

df = detect_recurring(df)











# ========= 5) ANALYSIS & CHARTS =========
# Monthly net (credits - debits)
monthly = df.groupby("Month")["Amount"].sum().reset_index(name="NetAmount")
print("\n--- Monthly net (credits - debits) ---")
print(monthly)

# Top merchants by spend (debits only)
top_merchants = (
    df[df["Type"] == "Debit"]
    .assign(AbsAmount=df["Amount"].abs())
    .groupby("Merchant")["AbsAmount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="TotalSpent")
)
print("\n--- Top merchants by spend (debits) ---")
print(top_merchants)

# Ensure outputs dir
os.makedirs("outputs", exist_ok=True)

# Save a cleaned snapshot
clean_path = "outputs/cleaned_transactions.csv"
df.to_csv(clean_path, index=False)
print(f"\nSaved cleaned CSV → {clean_path}")

# Monthly net bar chart
plt.figure()
plt.bar(monthly["Month"], monthly["NetAmount"])
plt.title("Monthly Net Flow")
plt.xlabel("Month"); plt.ylabel("Net Amount ($)")
plt.tight_layout()
plt.savefig("outputs/monthly_net_flow.png")
plt.close()
print("Saved chart → outputs/monthly_net_flow.png")

# Category pie (debits only)
cat = (df[df["Type"]=="Debit"]
       .groupby("Category")["Amount"]
       .sum().abs().sort_values(ascending=False))
plt.figure()
plt.pie(cat.values, labels=cat.index, autopct="%1.1f%%")
plt.title("Spending by Category (Debits)")
plt.tight_layout()
plt.savefig("outputs/category_spending_pie.png")
plt.close()
print("Saved chart → outputs/category_spending_pie.png")

# Recurring list (console + put in summary)
rec_series = (df[(df["Type"]=="Debit") & (df["Recurring"]==1)]
              .groupby("Merchant")["Amount"]
              .sum().abs().sort_values(ascending=False))
if len(rec_series):
    print("\nRecurring merchants (approx monthly):")
    for m, a in rec_series.items():
        print(f" - {m}: ~${a:.2f}")
else:
    print("\nNo recurring merchants detected yet.")












# ========= 6) SQLITE PERSISTENCE (idempotent) =========
def make_transaction_id(row) -> str:
    """
    Stable hash of fields to uniquely identify a transaction.
    If your bank has a unique ID, use that instead.
    """
    key = f"{row.get('Date')}|{row.get('Amount')}|{row.get('Description')}|{row.get('Running Bal.', '')}"
    return hashlib.sha256(key.encode()).hexdigest()

df["transaction_id"] = df.apply(make_transaction_id, axis=1)

DB_PATH = Path("transactions.db")
DDL = """
CREATE TABLE IF NOT EXISTS transactions (
  transaction_id TEXT PRIMARY KEY,
  date            TEXT,
  description     TEXT,
  merchant        TEXT,
  category        TEXT,
  amount          REAL,
  running_balance REAL,
  type            TEXT,
  month           TEXT,
  recurring       INTEGER DEFAULT 0
);
"""

def ensure_db(db_path: Path = DB_PATH):
    con = sqlite3.connect(db_path)
    con.execute(DDL)
    con.commit()
    con.close()

def fetch_existing_ids(db_path: Path = DB_PATH) -> set[str]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT transaction_id FROM transactions;")
    rows = cur.fetchall()
    con.close()
    return {r[0] for r in rows}

def insert_rows(df_new: pd.DataFrame, db_path: Path = DB_PATH):
    # defaults if missing
    if "Category" not in df_new.columns:
        df_new["Category"] = "Other"
    if "Recurring" not in df_new.columns:
        df_new["Recurring"] = 0

    con = sqlite3.connect(db_path)
    to_write = df_new.rename(columns={
        "Date": "date",
        "Description": "description",
        "Merchant": "merchant",
        "Amount": "amount",
        "Running Bal.": "running_balance",
        "Type": "type",
        "Month": "month",
        "transaction_id": "transaction_id",
        "Category": "category",
        "Recurring": "recurring",
    })[[
        "transaction_id","date","description","merchant","category",
        "amount","running_balance","type","month","recurring"
    ]].copy()
    to_write.to_sql("transactions", con, if_exists="append", index=False)
    con.close()

# Run upsert
ensure_db(DB_PATH)
existing_ids = fetch_existing_ids(DB_PATH)
df_new = df[~df["transaction_id"].isin(existing_ids)].copy()

print(f"\nNew rows to insert: {len(df_new)}")
if len(df_new):
    insert_rows(df_new, DB_PATH)
    print("Inserted new rows into SQLite.")
else:
    print("No new rows — database is up to date.")

# Quick DB stats
def quick_db_stats(db_path: Path = DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    total = cur.execute("SELECT COUNT(*) FROM transactions;").fetchone()[0]
    months = cur.execute("SELECT COUNT(DISTINCT month) FROM transactions;").fetchone()[0]
    con.close()
    print(f"\nDB stats → rows: {total}, distinct months: {months}")

quick_db_stats(DB_PATH)












# ========= 7) SMALL SUMMARY FILE =========
summary_lines = []
summary_lines.append("Personal Finance Analyzer — Summary")
summary_lines.append(f"Transactions this run: {len(df)}")
if len(monthly):
    last_net = monthly.tail(1)["NetAmount"].iloc[0]
    summary_lines.append(f"Latest month net (credits - debits): ${last_net:.2f}")
if len(cat):
    top_cat = cat.index[0]
    top_amt = cat.iloc[0]
    summary_lines.append(f"Top spending category: {top_cat} (${top_amt:.2f})")
if len(rec_series):
    summary_lines.append("Recurring merchants detected:")
    for m, a in rec_series.items():
        summary_lines.append(f"  - {m}: ~${a:.2f}/mo")

Path("outputs").mkdir(parents=True, exist_ok=True)
(Path("outputs") / "summary.txt").write_text("\n".join(summary_lines))
print("Saved summary → outputs/summary.txt")
