# AI-Powered Fraud Detection System

A local, educational Streamlit dashboard for reviewing historical credit-card transactions and using two existing saved classifiers to assess an individual transaction. The app loads the supplied Logistic Regression and Random Forest artifacts; it does not retrain models at startup.

> **Educational prototype only.** This is not a production banking security system. A “Genuine” result is not a guarantee that a transaction is safe, and a model score is not a calibrated real-world risk probability.

## Features

- Overview metrics and interactive historical class, elapsed-time, and amount charts.
- Session-persistent Light/Dark appearance toggle.
- Transaction prediction with saved Logistic Regression and Random Forest models.
- Inputs generated for all 30 model features in their original training order.
- Real historical examples from the local dataset, plus an explicit manual-entry form that requires every field.
- Cached held-out evaluation with Accuracy, fraud-class Precision/Recall/F1, and confusion matrices.
- Filterable historical data explorer.
- Consistent fintech Light and Dark themes with responsive layouts, accessible status colors, and theme-aware Plotly charts.
- Helpful data/model validation and missing-file messages.

## Technology

Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Joblib, and KaggleHub. Scikit-learn is pinned to 1.7.2 to match the saved model serialization version.

## Project structure

```text
FRAUD_DETECTION/
├── app.py
├── creditcard.csv                    # Local dataset; excluded from Git
├── logistic_regression_model.pkl     # Existing saved pipeline
├── random_forest_model.pkl           # Existing saved classifier
├── requirements.txt
├── README.md
└── .gitignore
```

The existing assets are stored in the project root, so the application resolves those actual paths rather than moving or rewriting them. No model-training notebook was present in this project folder; the saved models are reused without modification.

The supplied project folder did not contain a Jupyter notebook. No notebook has been recreated or substituted; the existing trained artifacts and inference path are preserved.

## Dataset

The app checks `FRAUD_DATA_PATH`, `creditcard.csv` beside `app.py`, and `data/creditcard.csv`, in that order. Paths are resolved with `pathlib`. If no local file exists, KaggleHub automatically downloads the real dataset `mlg-ulb/creditcardfraud` and locates its `creditcard.csv` in KaggleHub's returned cache directory. The downloaded path is cached as a Streamlit resource and the validated DataFrame is cached as Streamlit data, so normal reruns do not re-download or re-parse the dataset. KaggleHub also maintains its own disk cache for the running environment.

The CSV is approximately 151 MB, exceeds GitHub's 100 MB per-file limit, and is ignored by Git. It is therefore not committed. If neither local path is available, KaggleHub automatically downloads the real source dataset. In Streamlit Cloud, add these entries under **App settings → Secrets**:

```toml
KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_api_key"
```

Create the API key in your Kaggle account settings, keep it only in Streamlit Secrets (never in source control), and accept the dataset's Kaggle terms with that account. KaggleHub's download cache is reused during the app process and across normal reruns. For local runs, configure the same two values as environment variables or use Kaggle's supported local credentials file. If credentials, Kaggle access, or the dataset file are unavailable, the app gives a user-friendly setup error. No records are synthesized, modified, or used to retrain the models, and no live banking feed is used.

The inspected dataset contains 284,807 rows and 31 columns: `Time`, `V1` through `V28`, `Amount`, and `Class`. `Class=0` is genuine; `Class=1` is fraud. It contains 492 fraud records (about 0.173%) and 1,081 duplicate rows. The app preserves all supplied rows for historical statistics and reproduces the saved models' inferred training split for evaluation.

## Models and feature order

The saved artifacts were inspected and expose the same 30-feature order:

```text
Time, V1, V2, ..., V28, Amount
```

The Logistic Regression artifact is a `StandardScaler` + `LogisticRegression` pipeline. The Random Forest artifact is a fitted `RandomForestClassifier`. Both support `predict_proba`. Their pickles report scikit-learn 1.7.2; loading pickles from untrusted sources can execute code, so use only the supplied trusted artifacts.

The app checks each model's `feature_names_in_`, feature count, and binary classes before making it available. Predictions always pass a DataFrame with the saved feature order.

## Evaluation methodology

No estimator is refitted in the dashboard. The saved scaler's training statistics match a reproducible `train_test_split(test_size=0.2, random_state=42, stratify=Class)` training partition to floating-point precision. The Model Performance page evaluates both unchanged saved models against the corresponding held-out 20% partition and caches the result. Fraud (`Class=1`) is the positive class. Accuracy is shown with fraud precision, recall, F1, and the confusion matrix because the class imbalance makes accuracy alone misleading.

Metrics are computed from the local dataset and loaded models when the page is opened; the project does not claim preset or invented performance scores.

The existing saved models were evaluated on the recovered 56,962-row held-out partition (98 fraud cases):

| Saved model | Accuracy | Fraud precision | Fraud recall | Fraud F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9755 | 0.0610 | 0.9184 | 0.1144 |
| Random Forest | 0.9995 | 0.9605 | 0.7449 | 0.8391 |

These are dataset-specific held-out results, not a guarantee of future performance. The Logistic Regression model favors recall but produces many false positives; the Random Forest model is more precise but misses more fraud cases. Review the confusion matrices and operating threshold tradeoffs rather than relying on accuracy alone.

## Installation

Windows PowerShell, from this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Keep the existing trusted model files in the project root and place the dataset there as `creditcard.csv`.

## Run the application

```bash
streamlit run app.py
```

Use **Overview** for historical dataset statistics, **Transaction prediction** to compare saved models or assess a real dataset example, **Model performance** for held-out metrics, **Data explorer** for filters and interactive charts, and **About** for model limitations. The sidebar appearance selector switches the entire interface and Plotly charts between Light and Dark themes and stays selected while navigating during the session. Manual prediction requires `Time`, `Amount`, and all `V1`–`V28` fields; blank values are rejected instead of being silently filled with zeros.

## Screenshots

Add screenshots of Overview, Transaction prediction in both themes, and Model performance here after launch.

## Limitations and future improvements

This dataset is highly imbalanced and historical. Model probabilities are not calibrated risk estimates; the model may not generalize to other time periods, populations, or banks. Before any operational use, obtain domain approval and independently validate calibration, recall/precision tradeoffs, fairness, drift, security, and incident handling. Future work could add explicit calibration evaluation, threshold analysis, model monitoring, and an auditable review workflow.

## Author

Add author name and portfolio links here.
