# Part 4 - LLM Fraud Detection Report

## Best Model

- Model: Tuned Random Forest (GridSearchCV)
- Source: best_model.pkl

## Prediction Summary

|   Sample ID | Risk Level   |   Fraud Probability | Prediction      |   Explanation Confidence | Summary                                                                         | Recommendation                               |
|------------:|:-------------|--------------------:|:----------------|-------------------------:|:--------------------------------------------------------------------------------|:---------------------------------------------|
|           1 | Low          |                 0   | Legitimate      |                      1   | Low risk transaction with a recognized email domain and a common payment method | Approve transaction                          |
|       33858 | Medium       |                 0.5 | Review Manually |                      0.5 | Transaction requires manual review due to uncertain fraud probability           | Verify transaction details with the customer |
|       19302 | High         |                 1   | Fraud           |                      1   | High risk transaction due to unusual features                                   | Reject transaction                           |

## Validation

Validated Responses : 3/3
