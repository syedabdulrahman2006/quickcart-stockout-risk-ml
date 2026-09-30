import pandas as pd

# Load the datasets
stores = pd.read_csv("dim_stores.csv")
skus = pd.read_csv("dim_skus.csv")
suppliers = pd.read_csv(
    "dim_suppliers.csv",
    na_values=["N/A", "missing", "--", "NA", "null"]
)
events = pd.read_csv("dim_events.csv", parse_dates=["date"])
inventory = pd.read_csv("fact_inventory_daily.csv", parse_dates=["date"])

# Check row counts
print("Stores:", len(stores))
print("SKUs:", len(skus))
print("Suppliers:", len(suppliers))
print("Events:", len(events))
print("Inventory:", len(inventory))

# Check date range
print("\nDate range:")
print(inventory["date"].min(), "to", inventory["date"].max())

# Check target distribution
print("\nStockout Risk:")
print(inventory["stockout_risk"].value_counts())
print("\nData loaded successfully!")


# Clean supplier reliability
suppliers["reliability_score"] = pd.to_numeric(
    suppliers["reliability_score"],
    errors="coerce"
)

suppliers["reliability_score"] = suppliers["reliability_score"].fillna(
    suppliers["reliability_score"].median()
)

# Clean city display names
stores["city_display"] = stores["city_display"].str.title()

# Join all tables
data = inventory.merge(skus, on="sku_id", how="left")
data = data.merge(stores, on="store_id", how="left")
data = data.merge(
    suppliers,
    left_on="supplier_id_x",
    right_on="supplier_id",
    how="left"
)
data = data.merge(events, on="date", how="left")

print("\nJoined dataset shape:")
print(data.shape)

print("\nJoined dataset columns:")
print(data.columns.tolist())

# Feature engineering

data["reorder_gap"] = (
    data["reorder_point"] - data["closing_stock"]
)

data["days_of_cover_ratio"] = (
    data["days_of_cover"] /
    data["lead_time_days_expected"].replace(0, 1)
)

data["day_of_month"] = data["date"].dt.day

data["days_since_festival_start"] = (
    data["date"] - pd.Timestamp("2026-10-22")
).dt.days.clip(lower=0)

print("\nNew features created:")
print(data[[
    "reorder_gap",
    "days_of_cover_ratio",
    "day_of_month",
    "days_since_festival_start"
]].head())

# Time-based train/test split

train = data[data["date"] <= "2026-10-23"]
test = data[data["date"] >= "2026-10-24"]

print("\nTrain rows:", len(train))
print("Test rows:", len(test))

# Select features and target

features = [
    "closing_stock",
    "reorder_point",
    "days_of_cover",
    "lead_time_days_expected",
    "reorder_gap",
    "days_of_cover_ratio",
    "reliability_score",
    "day_of_month",
    "days_since_festival_start"
]

X_train = train[features]
y_train = train["stockout_risk"]

X_test = test[features]
y_test = test["stockout_risk"]

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

# Train Random Forest model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Evaluate model

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save predictions

results = test[["date", "store_id", "sku_id", "stockout_risk"]].copy()
results["predicted_risk"] = y_pred

print("\nSample predictions:")
print(results.head(10))

# Feature importance

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Save predictions

results = test[["date", "store_id", "sku_id", "stockout_risk"]].copy()
results["predicted_risk"] = y_pred

results.to_csv("quickcart_predictions.csv", index=False)

print("\nPredictions saved to quickcart_predictions.csv")