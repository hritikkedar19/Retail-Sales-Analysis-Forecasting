import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/predictions", exist_ok=True)
os.makedirs("outputs/charts", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_retail_sales.csv")

monthly = df.groupby(["Year", "Month"]).agg({
    "Sales": "sum",
    "Profit": "sum",
    "Quantity": "sum",
    "Discount": "mean"
}).reset_index()

monthly["Time_Index"] = range(1, len(monthly) + 1)

X = monthly[["Year", "Month", "Quantity", "Discount", "Time_Index"]]
y = monthly["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/predictions/model_results.csv", index=False)

best_model_name = results_df.sort_values("RMSE").iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, "models/sales_forecast_model.pkl")
joblib.dump(list(X.columns), "models/features.pkl")

monthly["Predicted_Sales"] = best_model.predict(X)
monthly.to_csv("outputs/predictions/monthly_sales_predictions.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(monthly["Time_Index"], monthly["Sales"], marker="o", label="Actual Sales")
plt.plot(monthly["Time_Index"], monthly["Predicted_Sales"], marker="o", label="Predicted Sales")
plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Time Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/charts/actual_vs_predicted_sales.png")
plt.show()

print("Model training completed.")
print(results_df)
print("Best model:", best_model_name)
