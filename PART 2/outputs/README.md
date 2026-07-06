## Overview

This project focuses on building and evaluating **Regression** and **Binary Classification models** using a large transactional dataset containing **39,422 rows and 434 features**. The goal is to predict:

- Transaction amount (Regression)
- Fraud detection (Classification)

## The project includes preprocessing, feature engineering, model training, evaluation, threshold tuning, and statistical validation.

## Dataset Information

- **Rows:** 39,422
- **Columns:** 434
- **Target Variables:**
  - Regression → `TransactionAmt`
  - Classification → `isFraud`

---

## Project Pipeline

### Data Preparation

- Missing value handling
- Encoding categorical variables
- Feature scaling
- Train-test split (75/25)
  Final feature set:
- **Numeric features:** 400
- **Categorical features:** 31
- **Final processed features:** 1215 (after encoding)

---

## Regression Models

### Models Used:

- Linear Regression
- Ridge Regression

### Performance Summary:

| Model             | MSE      | RMSE   | R² Score |
| ----------------- | -------- | ------ | -------- |
| Linear Regression | 41998.46 | 204.94 | 0.1985   |
| Ridge Regression  | 41922.57 | 204.75 | 0.1999   |

### Key Insights:

- Ridge regression slightly improved performance.
- Model explains ~20% variance (low predictive strength).
- Most influential features include:
  - `V127`, `V128`, `V133`
  - Certain email domain categories

---

## Classification Model (Fraud Detection)

### Model Used:

- Logistic Regression

### Dataset Imbalance:

- Non-Fraud (0): 97.16%
- Fraud (1): 2.84%

---

### Performance Metrics:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.8686 |
| Precision | 0.1394 |
| Recall    | 0.7009 |
| F1 Score  | 0.2326 |
| ROC AUC   | 0.8526 |

### Confusion Matrix:

[[6692 969]
[ 67 157]]

---

### Key Observations:

- High recall (70%) → good fraud detection capability
- Low precision → many false positives
- Strong class imbalance impact

---

## Threshold Optimization

| Threshold | Precision | Recall | F1 Score  |
| --------- | --------- | ------ | --------- |
| 0.3       | 0.077     | 0.808  | 0.141     |
| 0.5       | 0.139     | 0.701  | 0.233     |
| 0.7       | 0.281     | 0.621  | **0.387** |

### Best Threshold: **0.7**

- Best trade-off between precision and recall

---

## Regularization Comparison

| Model             | Precision | Recall | F1 Score | ROC AUC |
| ----------------- | --------- | ------ | -------- | ------- |
| Logistic (C=1)    | 0.281     | 0.621  | 0.387    | 0.8526  |
| Logistic (C=0.01) | 0.125     | 0.719  | 0.213    | 0.8588  |

---

## Statistical Validation

- Bootstrap Mean Difference: **-0.0058**
- 95% Confidence Interval: **[-0.0218, 0.0089]**
- Conclusion: No statistically significant difference between models

---

## Outputs Generated

- Trained Regression Models (Linear, Ridge)
- Logistic Regression Model
- ROC Curve Plot
- Prediction Files
- Threshold Analysis Report
- Bootstrap Confidence Results

---

## Key Learnings

- Feature engineering significantly increases dimensionality (1215 features)
- Class imbalance strongly affects precision-recall tradeoff
- Threshold tuning improves fraud detection performance
- Ridge regression provides slight stability improvement over linear regression
- ROC-AUC remains strong despite imbalance

---

## Conclusion

This project demonstrates a complete end-to-end machine learning pipeline including:
✔ Data preprocessing  
✔ Regression modeling  
✔ Fraud classification  
✔ Model evaluation  
✔ Threshold optimization  
✔ Statistical validation  
While regression performance is limited, classification results show strong fraud detection capability with optimized threshold tuning.

---
