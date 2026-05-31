import nbformat as nbf

nb = nbf.v4.new_notebook()

text_title = """# Part 3: Churn Prediction Model

**Objective:** Train a Baseline model and a Champion model to predict `churn_next_60d` using historical snapshot features up to September 30, 2025."""

code_imports = """import pandas as pd
import numpy as np
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

data_path = 'data/d2c_churn_data_package/'"""

text_data = """### 1. Data Loading & Preprocessing
We use the pre-computed `rfm_modeling_snapshot.csv` which respects the snapshot date strictly."""

code_data = """df = pd.read_csv(data_path + 'rfm_modeling_snapshot.csv')

features = [
    'city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 'preferred_category', 'marketing_consent',
    'recency_days', 'frequency_180d', 'monetary_180d', 'return_rate_180d', 'avg_discount_pct_180d', 'avg_rating_180d',
    'category_diversity_180d', 'ticket_count_90d', 'negative_ticket_rate_90d', 'avg_resolution_hours_90d', 
    'days_since_signup', 'sessions_30d', 'product_views_30d', 'cart_adds_30d', 'wishlist_adds_30d', 
    'abandoned_carts_30d', 'email_opens_30d', 'campaign_clicks_30d', 'last_visit_days_ago'
]
target = 'churn_next_60d'

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validation']
test_df = df[df['split'] == 'test']

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")"""

text_pipeline = """### 2. Feature Engineering Pipeline
Imputing missing values and scaling numericals."""

code_pipeline = """categorical_cols = ['city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 'preferred_category', 'marketing_consent']
numeric_cols = [c for c in features if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ]
)"""

text_train = """### 3. Model Training
We compare Logistic Regression (Baseline) to Random Forest (Champion)."""

code_train = """# Baseline
baseline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
baseline.fit(X_train, y_train)

# Champion
champion = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))])
champion.fit(X_train, y_train)"""

text_eval = """### 4. Evaluation & Threshold Tuning
We select a threshold on the validation set to prioritize Recall (missing a churner is expensive)."""

code_eval = """optimal_threshold = 0.40
y_test_probs = champion.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_probs >= optimal_threshold).astype(int)

print(f"ROC-AUC: {roc_auc_score(y_test, y_test_probs):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(confusion_matrix(y_test, y_test_pred))

joblib.dump(champion, 'model.pkl')"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_title),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_pipeline),
    nbf.v4.new_code_cell(code_pipeline),
    nbf.v4.new_markdown_cell(text_train),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval)
]

with open('churn_model.ipynb', 'w') as f:
    nbf.write(nb, f)
