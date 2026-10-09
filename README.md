# Telecom Customer Churn Prediction

Predicts whether a telecom customer is likely to churn, with a Flask web app for live predictions and a Power BI dashboard for business reporting.

**Author:** Chaman Kumar Singh · [LinkedIn](https://www.linkedin.com/in/chaman-singh-5a67aa2b6)

## Dataset facts (`tele_data.csv`)

- 7,043 customers, 21 columns
- Overall churn rate: **26.5%**
- Month-to-month contracts churn at **42.7%** vs **2.8%** on two-year contracts

## Approach

1. **Cleaning:** dropped `customerID`, converted `TotalCharges` to numeric and filled 11 missing values with the median, encoded the target.
2. **Encoding:** one-hot encoding of categorical columns (`pd.get_dummies`).
3. **Split:** 80/20 stratified train/test split.
4. **Scaling:** `StandardScaler` fitted on the training set.
5. **Model:** Logistic Regression with `class_weight="balanced"` because churners are the minority class.
6. **Artifacts:** model, scaler and feature list saved with `joblib` so inference uses identical preprocessing.
7. **Deployment:** Flask app loads the artifacts and serves predictions through a web form.
8. **Reporting:** Power BI dashboard in `churn_analysis.pbix`.

## Structure

```
train_model.py        data cleaning, encoding, training
app.py                Flask app for live predictions
templates/            HTML templates
churn_model.pkl       trained model
scaler.pkl            fitted scaler
features.pkl          feature column names
churn_analysis.pbix   Power BI dashboard
tele_data.csv         dataset
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py   # then open http://localhost:5000
```

## Next steps

- Evaluate on the held-out test set (recall, precision, ROC-AUC) and tune the decision threshold
- Compare against tree-based models
