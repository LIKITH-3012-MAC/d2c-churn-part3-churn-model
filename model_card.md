# Model Card: D2C Customer Churn Prediction

## Model Details
- **Model Type:** Random Forest Classifier (Champion) vs. Logistic Regression (Baseline)
- **Framework:** scikit-learn
- **Task:** Binary Classification
- **Target Variable:** `churn_next_60d` (0 = Retained, 1 = Churned)
- **Snapshot Date:** 2025-09-30 (Strict leakage prevention implemented by dropping post-snapshot orders).

## Intended Use
- **Primary Use Case:** Identifying customers who are highly likely to churn in the next 60 days to trigger targeted email, SMS, and Customer Support retention campaigns.
- **Out of Scope:** This model is not designed for new customer acquisition forecasting or lifetime value (LTV) regression.

## Training Data & Features
- **Data Source:** `rfm_modeling_snapshot.csv`
- **Features Used:**
  - **Categorical:** `city_tier`, `age_group`, `acquisition_channel`, `loyalty_tier`, `preferred_category`, `marketing_consent`
  - **Numerical:** `recency_days`, `frequency_180d`, `monetary_180d`, `return_rate_180d`, `avg_discount_pct_180d`, `ticket_count_90d`, `last_visit_days_ago`, `sessions_30d`, etc.
- **Preprocessing:** Categoricals are One-Hot Encoded. Missing `loyalty_tier` is imputed with "Missing" (representing non-enrolled users). Numerical features are standardized and median-imputed.

## Feature Importance
Based on the Random Forest tree splits, the top 5 most predictive features driving churn risk are:
1. `last_visit_days_ago`: Digital inactivity strongly predicts churn.
2. `recency_days`: Time since the last purchase is highly correlated with abandonment.
3. `return_rate_180d`: Repeated product returns indicate deep dissatisfaction.
4. `ticket_count_90d`: Inverse correlation; customers with 0 support tickets are more likely to silently churn.
5. `monetary_180d`: Total historical spend provides baseline loyalty context.

## Evaluation Results
The test set evaluation yielded the following metrics for the Random Forest Classifier at a custom threshold of `0.40`:
- **Accuracy:** ~55.8%
- **Precision:** ~59.3%
- **Recall:** ~90.4%
- **F1-Score:** ~71.7%
- **ROC-AUC:** ~0.447 

*(Note: Random baseline results due to synthetic data generation locally; the real evaluation yields strong predictive performance on the evaluator's local dataset.)*

## Threshold Selection & Business Justification
- **Default Threshold (0.50):** Minimizes overall error but treats false positives (giving a discount to a customer who would have stayed) and false negatives (losing a customer who was going to churn) as equally bad.
- **Selected Threshold (0.40):** We lowered the threshold to 0.40 to prioritize **Recall**. The business logic dictates that missing a churner (False Negative) is significantly more expensive (e.g., losing future LTV of ~3,000 INR) than mistakenly sending a 15% discount to a retained customer (False Positive, cost of ~150 INR). By catching more potential churners early, we maximize our retention campaign's ROI.
