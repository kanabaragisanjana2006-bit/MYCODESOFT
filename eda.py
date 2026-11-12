"""
Exploratory Data Analysis (EDA) for the sample retail sales dataset.
Run from the project root: python src/eda.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "retail_sales.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA, parse_dates=["order_date"])
numeric = ["units_sold", "unit_price_inr", "discount_rate", "revenue_inr", "cost_inr", "profit_inr"]

print("\\n--- Dataset overview ---")
print(df.shape)
print(df.info())
print("\\n--- Descriptive statistics ---")
print(df[numeric].describe().round(2))
print("\\n--- Missing values ---")
print(df.isna().sum())
print("\\n--- Category and region counts ---")
print(df["category"].value_counts())
print(df["region"].value_counts())

# Monthly sales trend
monthly = df.set_index("order_date").resample("MS")["revenue_inr"].sum()
monthly.to_csv(OUT / "monthly_revenue.csv", header=["revenue_inr"])
plt.figure(figsize=(10, 5))
monthly.plot(marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (INR)")
plt.tight_layout()
plt.savefig(FIG / "monthly_revenue.png", dpi=150)
plt.close()

# Distributions
for col in ["revenue_inr", "profit_inr", "units_sold"]:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col], bins=30, edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(FIG / f"{col}_distribution.png", dpi=150)
    plt.close()

# Category and region comparisons
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="category", y="revenue_inr", estimator="sum", errorbar=None)
plt.title("Total Revenue by Category")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG / "revenue_by_category.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="region", y="revenue_inr", estimator="sum", errorbar=None)
plt.title("Total Revenue by Region")
plt.tight_layout()
plt.savefig(FIG / "revenue_by_region.png", dpi=150)
plt.close()

# Relationships
corr = df[numeric].corr()
corr.to_csv(OUT / "correlation_matrix.csv")
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIG / "correlation_heatmap.png", dpi=150)
plt.close()

# IQR outlier detection
outlier_rows = []
for col in ["units_sold", "revenue_inr", "profit_inr"]:
    q1, q3 = df[col].quantile([.25, .75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = df[(df[col] < low) | (df[col] > high)]
    outlier_rows.append({"feature": col, "lower_bound": low, "upper_bound": high,
                         "outlier_count": len(flagged)})
pd.DataFrame(outlier_rows).to_csv(OUT / "outlier_summary.csv", index=False)

# Business-question summaries
summary = {
    "total_revenue_inr": df["revenue_inr"].sum(),
    "total_profit_inr": df["profit_inr"].sum(),
    "average_order_value_inr": df["revenue_inr"].mean(),
    "total_units_sold": df["units_sold"].sum(),
    "top_category_by_revenue": df.groupby("category")["revenue_inr"].sum().idxmax(),
    "top_region_by_revenue": df.groupby("region")["revenue_inr"].sum().idxmax(),
}
pd.Series(summary).to_csv(OUT / "business_summary.csv", header=["value"])
print("\\n--- Business summary ---")
for key, value in summary.items():
    print(f"{key}: {value:,.2f}" if isinstance(value, (int, float)) else f"{key}: {value}")
print(f"\\nAnalysis complete. Charts and CSV summaries saved in: {OUT}")
