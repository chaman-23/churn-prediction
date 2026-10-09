import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("tele_data.csv")

# ---------------- DATA CLEANING ----------------
df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------- ENCODING ----------------
cat_cols = df.select_dtypes(include="object").columns
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Save feature names (VERY IMPORTANT)
joblib.dump(X.columns.tolist(), "features.pkl")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# ---------------- SAVE ARTIFACTS ----------------
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training complete")
print("Model, scaler & features saved")
