"""Generates the bundled sample datasets (run: python data/generate_samples.py)."""
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)
out = Path(__file__).parent

# ---- Sales transactions -------------------------------------------------
n = 4000
dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
cust = rng.integers(1, 501, n)
cat = rng.choice(["Electronics", "Furniture", "Clothing", "Grocery", "Sports"], n, p=[.25, .15, .25, .25, .10])
region = rng.choice(["North", "South", "East", "West"], n)
base = {"Electronics": 300, "Furniture": 220, "Clothing": 60, "Grocery": 25, "Sports": 90}
price = np.array([base[c] for c in cat]) * rng.lognormal(0, .25, n)
qty = rng.integers(1, 8, n)
season = 1 + .25 * np.sin((dates.dayofyear.values - 80) / 365 * 2 * np.pi) + (dates - dates.min()).days.values / 2000
disc = rng.choice([0, .05, .1, .2], n, p=[.5, .2, .2, .1])
rev = price * qty * (1 - disc) * season
cost = rev * rng.uniform(.55, .8, n)
sales = pd.DataFrame({"Order_ID": np.arange(10000, 10000 + n), "Order_Date": dates, "Customer_ID": cust,
    "Region": region, "Category": cat, "Quantity": qty, "Unit_Price": price.round(2), "Discount": disc,
    "Revenue": rev.round(2), "Cost": cost.round(2)})
sales["Profit"] = (sales.Revenue - sales.Cost).round(2)
sales.loc[rng.choice(n, 120, replace=False), "Discount"] = np.nan
sales.loc[rng.choice(n, 40, replace=False), "Revenue"] *= 6   # injected outliers / anomalies
sales = pd.concat([sales, sales.sample(25, random_state=1)])   # duplicate rows
sales.sort_values("Order_Date").to_csv(out / "sales_transactions.csv", index=False)

# ---- Customer churn -----------------------------------------------------
m = 2500
tenure = rng.integers(1, 72, m)
contract = rng.choice(["Month-to-month", "One year", "Two year"], m, p=[.55, .25, .20])
monthly = rng.normal(70, 25, m).clip(20, 140)
support = rng.poisson(2, m)
internet = rng.choice(["DSL", "Fiber", "None"], m, p=[.35, .5, .15])
logit = (-1.2 + 1.3 * (contract == "Month-to-month") - 0.04 * tenure + 0.012 * monthly + 0.25 * support
         + 0.4 * (internet == "Fiber") - 0.8 * (contract == "Two year"))
churn = (rng.random(m) < 1 / (1 + np.exp(-logit))).astype(int)
ch = pd.DataFrame({"CustomerID": [f"C{i:05d}" for i in range(m)], "Tenure_Months": tenure, "Contract": contract,
    "Internet_Service": internet, "Monthly_Charges": monthly.round(2), "Support_Calls": support,
    "Total_Charges": (monthly * tenure).round(2), "Age": rng.integers(18, 80, m),
    "Gender": rng.choice(["Male", "Female"], m), "Churn": np.where(churn == 1, "Yes", "No")})
ch.loc[rng.choice(m, 80, replace=False), "Total_Charges"] = np.nan
ch.to_csv(out / "customer_churn.csv", index=False)
print("Sample datasets written to", out)
