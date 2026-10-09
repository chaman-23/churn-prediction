from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ---------------- LOAD SAVED OBJECTS ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk = None

    if request.method == "POST":
        # Get user inputs
        tenure = float(request.form.get("tenure"))
        monthly_charges = float(request.form.get("MonthlyCharges"))

        # Auto-calculate TotalCharges
        total_charges = tenure * monthly_charges

        # Prepare input dictionary
        input_data = {}

        for feature in features:
            if feature == "tenure":
                input_data[feature] = tenure
            elif feature == "MonthlyCharges":
                input_data[feature] = monthly_charges
            elif feature == "TotalCharges":
                input_data[feature] = total_charges
            else:
                # For other encoded categorical features
                input_data[feature] = 0

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict probability
        prob = model.predict_proba(input_scaled)[0][1]

        # Output text
        prediction = f"Churn Probability: {round(prob * 100, 2)}%"

        # Business risk logic
        risk = "High" if prob >= 0.6 else "Low"

    return render_template(
        "index.html",
        prediction=prediction,
        risk=risk
    )

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

