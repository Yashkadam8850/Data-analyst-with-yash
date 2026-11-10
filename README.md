# 📈 IntelliAnalytics – Advanced Data Intelligence Platform

Upload any CSV / Excel file and get automated profiling, cleaning, EDA, statistics, BI, ML, forecasting,
explainable AI, insights and PDF reporting. No dataset-specific column names are hard-coded: types are detected
automatically and business roles (revenue, customer, date…) are only *suggested* from column-name keywords – you can always override them.

## Project structure
```
IntelliAnalytics/
├── app.py                  # Entry point: layout, theme, data loading, navigation
├── requirements.txt
├── .streamlit/config.toml
├── data/
│   ├── sales_transactions.csv   # sample: BI, RFM, cohort, forecasting
│   ├── customer_churn.csv       # sample: classification, churn, SHAP
│   └── generate_samples.py
└── modules/
    ├── utils.py        # loading, type detection, quality score, KPI cards, theming
    ├── dashboard.py    # 1. Executive dashboard
    ├── cleaning.py     # 2. Missing values, duplicates, outliers, types, feature engineering
    ├── eda.py          # 3. Univariate / bivariate / multivariate, correlations, distributions
    ├── stats.py        # 4. t-test, ANOVA, chi-square, correlation tests, OLS diagnostics, CIs
    ├── bi.py           # 5. Revenue/profit, trends, Pareto, RFM, cohort, segmentation, KPI alerts
    ├── ml.py           # 6. Leakage-safe pipelines, 4+ models, CV, tuning, K-Means
    ├── predictive.py   # 7. Sales/demand/time-series forecasting, churn & risk scoring
    ├── explain.py      # 8. Permutation importance, SHAP, prediction explanations, what-if
    ├── insights.py     # 9. Automated trends, anomalies, opportunities, recommendations
    └── report.py       # 10. PDF report, cleaned data export, chart export
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Python 3.10 – 3.12 recommended.

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository (`app.py` and `requirements.txt` at the repo root).
2. Go to https://share.streamlit.io → **New app** → select the repo, branch and `app.py`.
3. (Advanced settings) choose Python 3.11 or 3.12 → **Deploy**.

If PNG chart export fails on the cloud, add a `packages.txt` file containing `chromium` – the app still works
and falls back to HTML chart export / PDF without embedded images.

## Design notes
- **No data leakage:** the train/test split happens *before* any fitting; imputation, scaling and encoding live inside an
  sklearn `Pipeline`, so they are fitted on training folds only. Cross-validation runs on the training set; the test set is
  used once for final metrics. Identifier/date columns are auto-excluded, near-perfect target correlations are flagged,
  and a time-based split is available. Churn/risk scores for all rows use out-of-fold predictions.
- **Explainability:** SHAP (tree + linear explainers) on held-out rows, permutation importance on original features.
- **Resilience:** modules degrade gracefully (clear messages) when a dataset lacks the needed column types.

## Typical workflows
- *Sales data* → Dashboard → Cleaning → BI (Pareto, RFM, cohort) → Predictive (forecast) → Insights → PDF.
- *Churn data* → Cleaning → ML (target = churn) → Explainable AI → Predictive → Churn & Risk.
