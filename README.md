# 🧪 Hands-on Data Science Workshop

Two 60-minute, practical sessions built around one story. **ShopSmart** (a fictional online store) has a messy order export. Participants take it from raw data to clean data, then to insights and a reusable pipeline.

| | Session 1: Understanding & Cleaning | Session 2: Analysis, Correlation & Anomalies |
|---|---|---|
| Notebook | `Session1_Understanding_and_Cleaning.ipynb` | `Session2_Analysis_Correlation_Anomalies.ipynb` |
| Topics | What data is, reading CSV/Excel/JSON/Parquet, data types, data-quality dimensions, duplicates, type fixing, consistency, business rules, cross-field checks, missing values (MCAR/MAR/MNAR) | EDA, skew, groupby/pivot, time series, Pearson vs Spearman, correlation pitfalls & causation, z-score/robust z/IQR, contextual anomalies, Isolation Forest, pipelines, validation, bronze/silver/gold, feature engineering (RFM), feature scaling / normalization (min-max, standard, robust, log) |
| Output | `data/orders_clean.parquet` | `data/gold_orders.parquet`, `data/gold_customers.parquet` |

## 📁 Structure
```
DataScienceWorkshop/
├── Session1_Understanding_and_Cleaning.ipynb        ← participant notebook (blank exercises)
├── Session2_Analysis_Correlation_Anomalies.ipynb    ← participant notebook (blank exercises)
├── solutions/                                       ← instructor versions (solutions + outputs, not published)
├── data/                                            ← datasets (generated)
├── generate_data.py                                 ← recreates the messy dataset (seeded, reproducible)
└── requirements.txt
```

## ⚙️ Setup (participants)
You need **Python 3.10+** (https://www.python.org/downloads/; on Windows, tick **"Add python.exe to PATH"**) and **Git** (https://git-scm.com/downloads).

```bash
# 1. Clone the repo
git clone https://github.com/alaaodeh95/data-science-workshop.git
cd data-science-workshop

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install the packages
pip install -r requirements.txt

# 4. Start JupyterLab (it opens in your browser)
jupyter lab
```
Then open `Session1_Understanding_and_Cleaning.ipynb` and run **Kernel → Restart Kernel and Run All Cells** to check that everything works.

**No Git?** Use **Code → Download ZIP** on the GitHub page and extract it.

> The instructor solutions (`solutions/`) are not published in this repo.

## 🗃️ The dataset
`generate_data.py` creates about 5,000 synthetic orders with **deliberately injected problems**, so every concept shows up:

| Issue | Example |
|---|---|
| Wrong separator, decimal & encoding | `orders_eu.csv` (`;`, `,` decimals, latin-1) |
| Nested JSON / junk Excel header rows | `customers.json`, `products.xlsx` |
| Mixed date formats | `2024-08-12`, `08/12/2024`, `August 12, 2024`, `2024.08.12` |
| Numbers stored as text | `$1,234.50`, `590.07 USD`, `"unknown"` ages |
| Inconsistent categories | `NY`, `NYC`, `new york`, `Seatle`, `Clothes` |
| 8 spellings of a boolean | `Yes / Y / 1 / True / No / N / 0 / False` |
| Missing values + fake missing markers | `NaN`, `?`, `N/A`, `-` |
| Impossible values | age 999, negative quantity, discount `15` instead of `0.15` |
| Exact & near duplicates | Same `order_id`, slightly different rows |
| Useless columns | Constant `currency`, 99%-empty `notes` |
| Hidden relationships | Delivery days ↓ rating, rating ↓ returns, discount ↑ quantity, holiday seasonality, non-linear session time vs order value |
| Anomalies | Huge quantities, prices 20–60× normal, 200-day deliveries, 15-hour sessions |

To regenerate the dataset: `python generate_data.py`.

## 🎓 Instructor tips (mixed audience)
- **Before the session**, open each notebook and run **Kernel → Restart Kernel and Run All Cells** to make sure everything works.
- Every notebook has a **⏱ time budget per section**. If you fall behind, skip the 🚀 stretch exercises and the optional Python crash course.
- Beginners can follow by just running the 💻 cells. Advanced participants can do the 🚀 stretch exercises.
- Good discussion moments:
  - S1 §2.2: *"why did reading fail?"*
  - S1 §4.7: *"should we fill missing ratings?"*
  - S2 §2: *"why are Pearson and Spearman so different?"*
  - S2 §2.3: *"do discounts cause the holiday spike?"*
- Session 2 starts from `data/orders_clean.parquet`, which is already included. Anyone who didn't finish Session 1 can still follow.
