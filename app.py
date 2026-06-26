import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

st.set_page_config(page_title="Retail Sales Analysis", layout="wide")

st.title("Retail Sales Analysis & Demand Forecasting")
st.write("End-to-end real-world retail data project with EDA, visualization and sales forecasting.")

df = pd.read_csv("data/processed/cleaned_retail_sales.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
categories = st.sidebar.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

filtered = df[(df["Region"].isin(regions)) & (df["Category"].isin(categories))].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"₹{filtered['Sales'].sum():,.0f}")
col2.metric("Total Profit", f"₹{filtered['Profit'].sum():,.0f}")
col3.metric("Total Orders", filtered["Order_ID"].nunique())
col4.metric("Total Quantity", int(filtered["Quantity"].sum()))

st.subheader("Dataset Preview")
st.dataframe(filtered.head(30))

st.subheader("Sales by Region")
region_sales = filtered.groupby("Region", as_index=False)["Sales"].sum()
st.plotly_chart(px.bar(region_sales, x="Region", y="Sales"), use_container_width=True)

st.subheader("Profit by Category")
category_profit = filtered.groupby("Category", as_index=False)["Profit"].sum()
st.plotly_chart(px.bar(category_profit, x="Category", y="Profit"), use_container_width=True)

st.subheader("Sales Share by Segment")
segment_sales = filtered.groupby("Segment", as_index=False)["Sales"].sum()
st.plotly_chart(px.pie(segment_sales, names="Segment", values="Sales"), use_container_width=True)

st.subheader("Discount vs Profit")
st.plotly_chart(px.scatter(filtered, x="Discount", y="Profit", color="Category"), use_container_width=True)

st.subheader("Monthly Sales Trend")
monthly = filtered.groupby(["Year", "Month"], as_index=False)["Sales"].sum()
monthly["Date"] = pd.to_datetime(monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01")
st.plotly_chart(px.line(monthly, x="Date", y="Sales", markers=True), use_container_width=True)

st.subheader("Correlation Heatmap")
numeric_df = filtered.select_dtypes(include=["int64", "float64"])
st.plotly_chart(px.imshow(numeric_df.corr(), text_auto=True), use_container_width=True)

st.subheader("Sales Forecasting")
try:
    predictions = pd.read_csv("outputs/predictions/monthly_sales_predictions.csv")
    st.dataframe(predictions.tail(12))
    fig = px.line(
        predictions,
        x="Time_Index",
        y=["Sales", "Predicted_Sales"],
        title="Actual vs Predicted Monthly Sales"
    )
    st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.warning("Run python src/train_model.py first to generate forecasts.")

st.subheader("Conclusion")
st.write("""
This project shows how retail businesses can use data analysis and forecasting to understand sales trends,
track profitable categories, study discount impact, and plan inventory using predicted monthly sales.
""")
