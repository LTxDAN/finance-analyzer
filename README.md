# Personal Finance Analyzer

A backend pipeline in Python to ingest bank CSV statements, normalize transactions, categorize spending, detect recurring charges, and generate analytics.  

This project was built to practice backend/data engineering concepts while solving a real-world problem: visualizing and analyzing personal banking activity.  

---

## ✨ Features

- **ETL Pipeline**
  - Ingests CSV bank exports
  - Cleans and normalizes dates, amounts, and balances
  - Deduplicates transactions via a stable hash

- **Transaction Intelligence**
  - Simplifies merchant names
  - Rule-based categorization (Groceries, Gas, Utilities, etc.)
  - Approximate recurring charge detection (subscriptions, bills)

- **Persistence**
  - SQLite database backend
  - Idempotent upserts (safe to re-run without duplicates)

- **Analytics & Reporting**
  - Monthly net flow (credits – debits)
  - Top merchants by spending
  - Spending breakdown by category
  - Recurring merchant detection
  - Outputs to text summary + charts

- **Visualizations**
  - Bar chart: monthly net flow
  - Pie chart: spending by category

---

## 🛠️ Tech Stack

- **Python 3**
- [Pandas](https://pandas.pydata.org/) – data cleaning & analysis  
- [NumPy](https://numpy.org/) – numerical operations  
- [Matplotlib](https://matplotlib.org/) – charts & plots  
- [SQLite](https://www.sqlite.org/) – local persistence  

---

## 📂 Project Structure

finance_analyzer/
├── data/ # Place your CSV exports here (ignored by Git)
├── outputs/ # Generated charts, reports, and cleaned CSVs
├── finance_tool.py # Main pipeline script
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🚀 Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. Place your CSV export

3. Put your bank CSV export in the data/ folder.

   (This repo includes a small data/sample.csv for testing.)

4. Run the pipeline
   in the terminal execute python finance_tool.py

5. Check results

   Cleaned data → outputs/cleaned_transactions.csv

   Monthly net chart → outputs/monthly_net_flow.png

   Category spending pie → outputs/category_spending_pie.png

   Text summary → outputs/summary.txt

   Transactions persisted → transactions.db


-Next Steps

 Expose analytics as REST endpoints via FastAPI

 Add a CLI with flags (--csv, --out, --use-gpt)

 Automate ingestion with cron/Task Scheduler

 (Optional) Integrate GPT for fuzzy merchant categorization with caching

   Built by LTxDAN
 as part of a backend/data engineering learning journey.