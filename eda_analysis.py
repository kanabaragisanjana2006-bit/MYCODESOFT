from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "retail_sales_eda.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA, parse_dates=["Order_Date"])

numeric = ["Quantity", "Unit_Price", "Discount", "Sales", "Profit"]
print("\nDATASET INFO")
print(df.info())
print("\nDESCRIPTIVE STATISTICS")
print(df[numeric].describe().round(2))
print("\nMISSING VALUES")
print(df.isna().sum())

# IQR outlier analysis
rows = []
for col in numeric:
    q1, q3 = df[col].quantile([.25, .75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows.append({"Feature": col, "Q1": q1, "Q3": q3,
                 "Lower_Bound": low, "Upper_Bound": high,
                 "Outlier_Count": int(((df[col] < low) | (df[col] > high)).sum())})
pd.DataFrame(rows).to_csv(OUT / "outlier_summary.csv", index=False)
df[numeric].describe().T.to_csv(OUT / "descriptive_statistics.csv")

# Charts
plt.figure(figsize=(9, 5))
plt.hist(df["Sales"], bins=30, edgecolor="white")
plt.title("Distribution of Order Sales"); plt.xlabel("Sales ($)"); plt.ylabel("Number of Orders")
plt.tight_layout(); plt.savefig(OUT / "01_sales_distribution_histogram.png", dpi=170); plt.close()

cats = df.groupby("Category")["Sales"].median().sort_values(ascending=False).index
plt.figure(figsize=(10, 6))
plt.boxplot([df.loc[df.Category == c, "Sales"] for c in cats], tick_labels=cats)
plt.title("Sales Spread and Outliers by Category"); plt.xlabel("Category"); plt.ylabel("Sales ($)")
plt.xticks(rotation=18); plt.tight_layout()
plt.savefig(OUT / "02_sales_outliers_boxplot.png", dpi=170); plt.close()

corr = df[numeric].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(numeric)), numeric, rotation=25, ha="right")
ax.set_yticks(range(len(numeric)), numeric)
for i in range(len(numeric)):
    for j in range(len(numeric)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Correlation Heatmap")
fig.tight_layout(); fig.savefig(OUT / "03_correlation_heatmap.png", dpi=170); plt.close(fig)

plt.figure(figsize=(9, 5))
for cat, part in df.groupby("Category"):
    plt.scatter(part["Sales"], part["Profit"], alpha=.65, label=cat)
plt.title("Relationship Between Sales and Profit"); plt.xlabel("Sales ($)"); plt.ylabel("Profit ($)")
plt.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "04_sales_vs_profit_scatter.png", dpi=170, bbox_inches="tight"); plt.close()

monthly = df.assign(Month=df.Order_Date.dt.to_period("M").dt.to_timestamp()).groupby("Month")["Sales"].sum()
plt.figure(figsize=(11, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Monthly Sales Trend"); plt.xlabel("Month"); plt.ylabel("Total Sales ($)")
plt.gcf().autofmt_xdate(); plt.tight_layout()
plt.savefig(OUT / "05_monthly_sales_trend.png", dpi=170); plt.close()

counts = df.Category.value_counts()
plt.figure(figsize=(9, 5))
plt.bar(counts.index, counts.values)
plt.title("Number of Orders by Category"); plt.xlabel("Category"); plt.ylabel("Number of Orders")
plt.xticks(rotation=18); plt.tight_layout()
plt.savefig(OUT / "06_orders_by_category.png", dpi=170); plt.close()
print(f"\nAnalysis complete. Outputs saved in: {OUT}")
