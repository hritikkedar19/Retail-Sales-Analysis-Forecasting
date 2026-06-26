import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/retail_sales.csv")

print("Before cleaning")
print("Shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Quarter"] = df["Order_Date"].dt.quarter

df.to_csv("data/processed/cleaned_retail_sales.csv", index=False)

print("\nAfter cleaning")
print("Shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())
print("Cleaned data saved to data/processed/cleaned_retail_sales.csv")
