# Retail Sales Analysis & Demand Forecasting

## Project Overview
This is an end-to-end real-world data science project based on the retail domain.  
The project analyzes retail sales data, discovers business insights, visualizes trends, and builds a sales forecasting model.

## Domain
Retail / Business Analytics

## Key Features
- Real-world style retail dataset
- Data cleaning and preprocessing
- Missing value handling
- Duplicate removal
- Exploratory Data Analysis
- Statistical summaries
- Sales and profit analysis
- Category and region analysis
- Discount impact analysis
- Correlation analysis
- Monthly sales trend analysis
- Sales forecasting using Machine Learning
- Linear Regression and Random Forest models
- Model evaluation using MAE, RMSE, and R2 Score
- Interactive Streamlit dashboard
- Business insights and conclusion report

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Streamlit
- Joblib

## Folder Structure
```text
Retail_Sales_Analysis_Forecasting/
│
├── data/
│   ├── raw/
│   │   └── retail_sales.csv
│   └── processed/
│       └── cleaned_retail_sales.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   └── train_model.py
│
├── models/
│   ├── sales_forecast_model.pkl
│   └── features.pkl
│
├── outputs/
│   ├── charts/
│   ├── predictions/
│   └── reports/
│
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

### Step 1: Install libraries
```bash
pip install -r requirements.txt
```

### Step 2: Clean the data
```bash
python src/data_cleaning.py
```

### Step 3: Run EDA
```bash
python src/eda.py
```

### Step 4: Train forecasting model
```bash
python src/train_model.py
```

### Step 5: Run dashboard
```bash
streamlit run app.py
```

If Streamlit is not recognized:
```bash
python -m streamlit run app.py
```

## Project Workflow
1. Load retail sales dataset
2. Clean missing and duplicate data
3. Perform statistical analysis
4. Analyze sales, profit, category, segment, and region
5. Visualize key trends
6. Build monthly sales forecasting model
7. Evaluate models
8. Present findings in a dashboard

## Evaluation Metrics
- MAE
- RMSE
- R2 Score

## Business Insights
- Identify top-performing regions
- Identify most profitable product categories
- Understand how discount affects profit
- Track monthly sales trends
- Forecast future sales for planning


