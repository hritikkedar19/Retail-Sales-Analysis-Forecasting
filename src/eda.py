import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/charts", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_retail_sales.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

print("Dataset Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe())

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
best_region = df.groupby("Region")["Sales"].sum().idxmax()
best_category = df.groupby("Category")["Profit"].sum().idxmax()
best_segment = df.groupby("Segment")["Sales"].sum().idxmax()

report = f"""
RETAIL SALES ANALYSIS REPORT

Total Records: {len(df)}
Total Sales: {total_sales:,.2f}
Total Profit: {total_profit:,.2f}
Best Region by Sales: {best_region}
Best Category by Profit: {best_category}
Best Segment by Sales: {best_segment}

Key Findings:
1. {best_region} region generated the highest sales.
2. {best_category} category generated the highest profit.
3. {best_segment} segment contributed the most sales.
4. Discount has a visible impact on profit.
5. Seasonal sales patterns are visible across months and quarters.

Business Recommendations:
1. Increase focus on high-performing regions and categories.
2. Reduce unnecessary high discounts when profit margins are low.
3. Improve underperforming regions using targeted campaigns.
4. Use monthly forecasting for inventory planning.
"""

with open("outputs/reports/retail_sales_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

def bar_chart(group_col, value_col, title, filename):
    plt.figure(figsize=(9, 5))
    data = df.groupby(group_col)[value_col].sum().reset_index()
    sns.barplot(data=data, x=group_col, y=value_col)
    plt.title(title)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f"outputs/charts/{filename}")
    plt.show()

bar_chart("Region", "Sales", "Total Sales by Region", "sales_by_region.png")
bar_chart("Category", "Sales", "Total Sales by Category", "sales_by_category.png")
bar_chart("Category", "Profit", "Total Profit by Category", "profit_by_category.png")
bar_chart("Segment", "Sales", "Total Sales by Segment", "sales_by_segment.png")

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Discount", y="Profit", hue="Category")
plt.title("Discount vs Profit")
plt.tight_layout()
plt.savefig("outputs/charts/discount_vs_profit.png")
plt.show()

plt.figure(figsize=(9, 5))
sns.histplot(df["Sales"], kde=True)
plt.title("Sales Distribution")
plt.tight_layout()
plt.savefig("outputs/charts/sales_distribution.png")
plt.show()

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, y="Sales")
plt.title("Sales Outlier Detection")
plt.tight_layout()
plt.savefig("outputs/charts/sales_boxplot.png")
plt.show()

monthly = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
monthly["Date"] = pd.to_datetime(monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01")

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly, x="Date", y="Sales", marker="o")
plt.title("Monthly Sales Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/charts/monthly_sales_trend.png")
plt.show()

plt.figure(figsize=(9, 6))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/charts/correlation_heatmap.png")
plt.show()

print("EDA completed. Charts and report saved.")
