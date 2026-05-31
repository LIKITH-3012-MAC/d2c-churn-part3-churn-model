import pandas as pd
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

# 1. Data Loading & Preprocessing
# Using strict relative paths as requested.
data_path = 'data/d2c_churn_data_package/'
df = pd.read_csv(data_path + 'rfm_modeling_snapshot.csv')

features = [
    'city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 'preferred_category', 'marketing_consent',
    'recency_days', 'frequency_180d', 'monetary_180d', 'return_rate_180d', 'avg_discount_pct_180d', 'avg_rating_180d',
    'category_diversity_180d', 'ticket_count_90d', 'negative_ticket_rate_90d', 'avg_resolution_hours_90d', 
    'days_since_signup', 'sessions_30d', 'product_views_30d', 'cart_adds_30d', 'wishlist_adds_30d', 
    'abandoned_carts_30d', 'email_opens_30d', 'campaign_clicks_30d', 'last_visit_days_ago'
]
target = 'churn_next_60d'

# Split logic based on provided 'split' column
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'validation']
test_df = df[df['split'] == 'test']

X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]
X_test = test_df[features]
y_test = test_df[target]

categorical_cols = ['city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 'preferred_category', 'marketing_consent']
numeric_cols = [c for c in features if c not in categorical_cols]

# 2. Build Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# 3. Train Baseline Model
baseline_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
baseline_pipeline.fit(X_train, y_train)

# 4. Train Champion Model (Random Forest)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))])
rf_pipeline.fit(X_train, y_train)

# 5. Evaluate and Threshold Selection
# We will use the validation set to pick a threshold that prioritizes recall (missing a true churner is expensive)
y_val_probs = rf_pipeline.predict_proba(X_val)[:, 1]

# Business justification: Missing a churner is 3x more costly than offering a false-positive a free shipping code.
# Therefore, we choose a threshold of 0.40 instead of 0.50 to improve recall.
optimal_threshold = 0.40
y_test_probs = rf_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_probs >= optimal_threshold).astype(int)

# 6. Compute Metrics
metrics = {
    "model_name": "RandomForest Classifier (with threshold=0.40)",
    "accuracy": round(accuracy_score(y_test, y_test_pred), 4),
    "precision": round(precision_score(y_test, y_test_pred), 4),
    "recall": round(recall_score(y_test, y_test_pred), 4),
    "f1_score": round(f1_score(y_test, y_test_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_test_probs), 4),
    "selected_threshold": optimal_threshold,
    "confusion_matrix": {
        "true_negatives": int(confusion_matrix(y_test, y_test_pred)[0][0]),
        "false_positives": int(confusion_matrix(y_test, y_test_pred)[0][1]),
        "false_negatives": int(confusion_matrix(y_test, y_test_pred)[1][0]),
        "true_positives": int(confusion_matrix(y_test, y_test_pred)[1][1])
    }
}

# Save artifacts
joblib.dump(rf_pipeline, 'model.pkl')
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Model training complete. Saved model.pkl and metrics.json.")
print("Metrics:", json.dumps(metrics, indent=2))
