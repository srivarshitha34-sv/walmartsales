from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Pointing to your CSV file path
data_path = Path("walmart_sales/Walmart_Store_sales.csv")
df = pd.read_csv(data_path)

if "Date" in df.columns:
  df = df.drop("Date", axis=1)

# Convert regression target into binary classification (1 = High Sales, 0 = Low Sales)
median_sales = df["Weekly_Sales"].median()
df["Sales_Category"] = np.where(df["Weekly_Sales"] >= median_sales, 1, 0)

X = df.drop(["Weekly_Sales", "Sales_Category"], axis=1)
y = df["Sales_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel training completed successfully!")

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")

joblib.dump(model, "retail_model.joblib")
joblib.dump(list(X.columns), "retail_features.joblib")
print("Model and features saved successfully!")