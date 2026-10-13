"""
Generates the synthetic, intentionally messy e-commerce dataset used in the workshop.

Run:  python generate_data.py
Output (in ./data):
    orders_raw.csv      main messy dataset (~5,000 rows)
    customers.json      customer profiles (JSON)
    products.xlsx       product catalog (Excel)
    orders_eu.csv       small export with ';' separator, ',' decimals, latin-1 encoding
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 5000
N_CUSTOMERS = 1200
rng = np.random.default_rng(SEED)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# ---------------------------------------------------------------- products
categories = {
    "Electronics": (80, 900),
    "Home & Kitchen": (15, 250),
    "Clothing": (10, 120),
    "Books": (5, 45),
    "Sports": (15, 300),
    "Beauty": (5, 80),
}
products = []
pid = 100
for cat, (lo, hi) in categories.items():
    for i in range(10):
        products.append(
            {
                "product_id": f"P{pid}",
                "product_name": f"{cat.split(' ')[0]} Item {i + 1}",
                "category": cat,
                "list_price": round(float(rng.uniform(lo, hi)), 2),
                "weight_kg": round(float(rng.uniform(0.1, 8)), 2),
            }
        )
        pid += 1
products = pd.DataFrame(products)

# ---------------------------------------------------------------- customers
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle"]
customers = pd.DataFrame(
    {
        "customer_id": [f"C{1000 + i}" for i in range(N_CUSTOMERS)],
        "age": rng.normal(38, 12, N_CUSTOMERS).clip(18, 80).round().astype(int),
        "city": rng.choice(cities, N_CUSTOMERS, p=[0.25, 0.2, 0.18, 0.15, 0.12, 0.10]),
        "membership": rng.choice(["Basic", "Silver", "Gold"], N_CUSTOMERS, p=[0.6, 0.3, 0.1]),
        "signup_date": pd.to_datetime("2021-01-01")
        + pd.to_timedelta(rng.integers(0, 1100, N_CUSTOMERS), unit="D"),
    }
)

# ---------------------------------------------------------------- orders (clean "truth")
cust_idx = rng.integers(0, N_CUSTOMERS, N_ORDERS)
prod_idx = rng.integers(0, len(products), N_ORDERS)
cust = customers.iloc[cust_idx].reset_index(drop=True)
prod = products.iloc[prod_idx].reset_index(drop=True)

all_days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
day_weight = np.where(all_days.month >= 11, 1.8, 1.0)  # holiday season: more orders
order_date = pd.Series(rng.choice(all_days, N_ORDERS, p=day_weight / day_weight.sum()))
month = order_date.dt.month.to_numpy()
# Seasonality: bigger discounts in Nov/Dec
discount = np.where(
    month >= 11,
    rng.uniform(0.1, 0.4, N_ORDERS),
    rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], N_ORDERS),
).round(2)
# Higher discount -> more quantity
quantity = (1 + rng.poisson(1 + discount * 8)).astype(int)
unit_price = (prod["list_price"].to_numpy() * rng.uniform(0.95, 1.05, N_ORDERS)).round(2)
shipping_method = rng.choice(["Standard", "Express", "Economy"], N_ORDERS, p=[0.55, 0.25, 0.20])
base_days = np.select([shipping_method == "Express", shipping_method == "Standard"], [1.5, 4.0], default=7.5)
delivery_days = np.maximum(1, np.round(base_days + rng.gamma(2, 1.0, N_ORDERS))).astype(int)
shipping_cost = (
    np.select([shipping_method == "Express", shipping_method == "Standard"], [15.0, 6.0], default=2.5)
    + prod["weight_kg"].to_numpy() * 0.8
).round(2)
# Slower delivery -> lower rating (negative correlation)
rating_raw = 5.2 - 0.25 * delivery_days + rng.normal(0, 0.7, N_ORDERS)
customer_rating = np.clip(np.round(rating_raw), 1, 5).astype(float)
# Low ratings -> more returns
returned = rng.random(N_ORDERS) < (0.03 + 0.08 * (5 - customer_rating))
total_amount = (quantity * unit_price * (1 - discount) + shipping_cost).round(2)
# Time on site has a non-linear (log) link with order value -> Spearman > Pearson
session_minutes = (np.log1p(total_amount) * 3 + rng.normal(0, 1.2, N_ORDERS)).clip(0.5).round(1)
payment = rng.choice(
    ["Credit Card", "PayPal", "Debit Card", "Cash on Delivery"], N_ORDERS, p=[0.45, 0.25, 0.2, 0.1]
)

orders = pd.DataFrame(
    {
        "order_id": [f"ORD-{100000 + i}" for i in range(N_ORDERS)],
        "order_date": order_date,
        "customer_id": cust["customer_id"],
        "customer_age": cust["age"].astype(float),
        "city": cust["city"],
        "product_id": prod["product_id"],
        "category": prod["category"],
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
        "shipping_method": shipping_method,
        "shipping_cost": shipping_cost,
        "delivery_days": delivery_days.astype(float),
        "payment_method": payment,
        "session_minutes": session_minutes,
        "customer_rating": customer_rating,
        "returned": returned,
        "total_amount": total_amount,
    }
)


# ---------------------------------------------------------------- inject data-quality issues
def pick(frac):
    return rng.choice(N_ORDERS, max(1, int(N_ORDERS * frac)), replace=False)


out = orders.copy().astype(object)

# 1) Mixed date formats + missing dates
d = orders["order_date"]
fmt = rng.choice(["iso", "us", "text", "dots"], N_ORDERS, p=[0.7, 0.15, 0.1, 0.05])
out["order_date"] = np.select(
    [fmt == "iso", fmt == "us", fmt == "text"],
    [d.dt.strftime("%Y-%m-%d"), d.dt.strftime("%m/%d/%Y"), d.dt.strftime("%B %d, %Y")],
    default=d.dt.strftime("%Y.%m.%d"),
)
out.loc[pick(0.01), "order_date"] = np.nan

# 2) Inconsistent category / city spellings
city_variants = {
    "New York": ["new york", "NEW YORK", "New York ", "NY", "NYC"],
    "Los Angeles": ["los angeles", "LA", "Los Angeles  "],
    "Chicago": ["chicago", " Chicago"],
    "Houston": ["houston", "HOUSTON"],
    "Phoenix": ["phoenix"],
    "Seattle": ["seattle", "Seatle"],
}
for i in pick(0.12):
    out.at[i, "city"] = rng.choice(city_variants[orders.at[i, "city"]])
cat_variants = {
    "Electronics": ["electronics", "ELECTRONICS", "Electronic"],
    "Home & Kitchen": ["Home and Kitchen", "home & kitchen", "Home&Kitchen"],
    "Clothing": ["clothing", "Clothes"],
    "Books": ["books", "Book"],
    "Sports": ["sports", "Sport"],
    "Beauty": ["beauty", "BEAUTY "],
}
for i in pick(0.08):
    out.at[i, "category"] = rng.choice(cat_variants[orders.at[i, "category"]])

# 3) Prices stored as text with currency symbols / thousands separators
for i in pick(0.06):
    p = orders.at[i, "unit_price"]
    out.at[i, "unit_price"] = rng.choice([f"${p:,.2f}", f"{p:,.2f} USD", f"USD {p}"])

# 4) Missing values (different amounts per column, plus "fake" missing markers)
out.loc[pick(0.07), "customer_age"] = np.nan
for i in pick(0.01):
    out.at[i, "customer_age"] = rng.choice(["unknown", "N/A", "-"])
out.loc[pick(0.05), "payment_method"] = np.nan
out.loc[pick(0.03), "payment_method"] = "?"
out.loc[pick(0.10), "customer_rating"] = np.nan  # not every customer leaves a rating
out.loc[pick(0.02), "delivery_days"] = np.nan
out.loc[pick(0.01), "shipping_cost"] = np.nan

# 5) Invalid / impossible values
for i in pick(0.004):
    out.at[i, "customer_age"] = float(rng.choice([-1, 0, 150, 999]))
for i in pick(0.004):
    out.at[i, "quantity"] = int(rng.choice([0, -1, -3]))
for i in pick(0.03):  # discount typed as a percent (15) instead of a fraction (0.15)
    out.at[i, "discount"] = round(float(orders.at[i, "discount"]) * 100, 1)

# 6) Inconsistent booleans
out["returned"] = [
    rng.choice(["Yes", "Y", "1", "True"]) if r else rng.choice(["No", "N", "0", "False"], p=[0.7, 0.1, 0.1, 0.1])
    for r in orders["returned"]
]

# 7) Anomalies (suspicious orders - fraud / data-entry errors)
anom = pick(0.006)
half = len(anom) // 2
for i in anom[:half]:
    out.at[i, "quantity"] = int(rng.integers(60, 250))
for i in anom[half:]:
    out.at[i, "unit_price"] = round(float(orders.at[i, "unit_price"]) * int(rng.integers(20, 60)), 2)
for i in pick(0.003):
    out.at[i, "delivery_days"] = float(rng.integers(60, 200))
for i in pick(0.003):
    out.at[i, "session_minutes"] = round(float(rng.uniform(300, 900)), 1)
# total_amount is NOT recomputed -> anomalous rows won't reconcile (a consistency check!)

# 8) Exact duplicates and "near" duplicates (same order_id, slightly different row)
dups = out.sample(60, random_state=SEED)
near = out.sample(15, random_state=SEED + 1).copy()
near["customer_rating"] = np.nan
out = pd.concat([out, dups, near], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

# 9) A useless constant column and a mostly-empty column
out["currency"] = "USD"
notes = pd.Series([np.nan] * len(out), dtype="object")
notes[rng.choice(len(out), 40, replace=False)] = rng.choice(
    ["gift", "call before delivery", "fragile", "leave at door"], 40
)
out["notes"] = notes

out.to_csv(DATA / "orders_raw.csv", index=False)

# ---------------------------------------------------------------- side files for the "reading data" section
records = [
    {
        "customer_id": r.customer_id,
        "profile": {"age": int(r.age), "city": r.city},
        "membership": r.membership,
        "signup_date": r.signup_date.strftime("%Y-%m-%d"),
    }
    for r in customers.itertuples()
]
pd.Series(records).to_json(DATA / "customers.json", orient="records", indent=2)

with pd.ExcelWriter(DATA / "products.xlsx") as xw:
    pd.DataFrame({"info": ["Product catalog export", "Generated for the workshop"]}).to_excel(
        xw, sheet_name="README", index=False
    )
    title = pd.DataFrame([["PRODUCT CATALOG - CONFIDENTIAL"]])
    title.to_excel(xw, sheet_name="catalog", index=False, header=False)
    products.to_excel(xw, sheet_name="catalog", index=False, startrow=2)  # junk title rows on top

eu = orders.head(200)[["order_id", "city", "unit_price", "discount", "total_amount"]].copy()
eu["city"] = eu["city"].replace({"New York": "Nueva York", "Los Angeles": "Los Ángeles"})
eu.to_csv(DATA / "orders_eu.csv", sep=";", decimal=",", index=False, encoding="latin-1")

print(f"orders_raw.csv: {out.shape[0]} rows x {out.shape[1]} columns")
print("Files written to", DATA.resolve())
