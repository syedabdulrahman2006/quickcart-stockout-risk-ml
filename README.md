[Uploading readme.,md…]()
# QuickCart Stockout Risk Prediction

A supervised machine learning project that predicts daily stockout risk for products across stores.

## Project Overview

The goal of this project is to classify each SKU-store-day into one of three stockout risk levels:

- Safe
- At-Risk
- Imminent

The project uses inventory, store, SKU, supplier, and event data to build a Random Forest classification model.

## Dataset

The project uses five datasets:

- `dim_stores.csv`
- `dim_skus.csv`
- `dim_suppliers.csv`
- `dim_events.csv`
- `fact_inventory_daily.csv`

The inventory dataset contains 21,600 daily SKU-store records covering October 1–30, 2026.

## Data Preparation

The following steps were performed:

1. Loaded all datasets using Pandas.
2. Converted supplier reliability values to numeric values.
3. Handled missing supplier reliability values using the median.
4. Standardized city display names.
5. Joined the inventory data with SKU, store, supplier, and event information.
6. Created additional features:
   - Reorder Gap
   - Days of Cover Ratio
   - Day of Month
   - Days Since Festival Start

## Machine Learning

### Model

Random Forest Classifier

### Train/Test Split

A time-based split was used:

- Training data: October 1–23, 2026
- Test data: October 24–30, 2026

This avoids randomly mixing future observations into the training data.

## Features

The model uses:

- Closing Stock
- Reorder Point
- Days of Cover
- Expected Lead Time
- Reorder Gap
- Days of Cover Ratio
- Supplier Reliability
- Day of Month
- Days Since Festival Start

## Results

The model achieved:

- **Accuracy:** 95%
- **At-Risk Recall:** 92%
- **Imminent Recall:** 77%
- **Safe Recall:** 100%

The confusion matrix and full classification report are generated when the Python script is executed.

## Output

The model generates:

`quickcart_predictions.csv`

This file contains:

- Date
- Store ID
- SKU ID
- Actual Stockout Risk
- Predicted Stockout Risk

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- VS Code

## How to Run

## How to Run

Make sure all CSV datasets are in the same folder as the Python file.

Run:

```bash
python quickcart_stockout.py
---

**Made by Syed Abdul Rahman**
