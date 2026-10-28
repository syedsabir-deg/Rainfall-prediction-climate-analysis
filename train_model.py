
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("Data/Processed/rainfall_combined.csv")

# 2. Define inputs and target
X = df[["district", "year", "month"]]
y = df["rainfall"]

# 3. Split by year to test predictions on later years
train_mask = df["year"] <= 2018
test_mask = df["year"] > 2018

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Encode district as a categorical feature
preprocessor = ColumnTransformer(
    transformers=[
        ("district", OneHotEncoder(handle_unknown="ignore"), ["district"])
    ],
    remainder="passthrough"
)

# 5. Create the machine-learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed!")

# 7. Evaluate on later years
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# 8. Save the trained model
os.makedirs("Models", exist_ok=True)
joblib.dump(model, "Models/rainfall_model.pkl")

print("\nModel saved to Models/rainfall_model.pkl")
