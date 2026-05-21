# Part 3: Churn Prediction Model & Model Card

## Project Overview
This repository contains Part 3 of the D2C Customer Churn Intelligence & Retention API Capstone. The objective is to construct a predictive machine learning model to estimate the probability that a customer will churn in the next 60 days using RFM and behavioral features computed strictly prior to the `2025-09-30` snapshot date.

## File Structure
- `churn_model.ipynb`: Jupyter notebook containing data loading, preprocessing pipelines, model training (Logistic Regression Baseline vs. Random Forest Champion), and threshold tuning logic.
- `model.pkl`: The serialized `scikit-learn` pipeline of the Champion model.
- `metrics.json`: JSON output containing precision, recall, f1, ROC-AUC, accuracy, and the confusion matrix for the Champion model on the test split.
- `model_card.md`: A standard ML Model Card detailing intended use, inputs, and a business justification for selecting a custom classification threshold to prioritize Recall.
- `error_analysis.md`: Detailed analysis of 10 hypothetical customer edge cases (False Positives and False Negatives), investigating why the model failed and the associated business costs.
- `requirements.txt`: Required Python dependencies.

## Setup Instructions
1. This project part is meant to be a standalone repository.
2. Ensure Python 3.9+ is installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Data Placement:** Place the original dataset package inside a `data/` folder one directory level above this repository, specifically at `../data/d2c churn data package/`.
5. Open the Jupyter Notebook:
   ```bash
   jupyter notebook churn_model.ipynb
   ```

## Model Overview
- **Champion Model:** Random Forest Classifier
- **Preprocessing:** One-hot encoding for categoricals, median-imputation and standardization for numericals.
- **Evaluation Strategy:** We actively reduced the classification threshold from `0.50` to `0.40` to bias the model toward Recall. In the D2C space, failing to identify a churner (False Negative) results in the loss of their entire future LTV, whereas incorrectly offering a discount to a retained customer (False Positive) only costs a minor margin hit.
