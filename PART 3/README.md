## Overview
The objective of Part 3 is to improve the fraud detection model developed in Part 2 by evaluating multiple ensemble learning algorithms, reducing overfitting, performing hyperparameter tuning, validating model performance using cross-validation, and building a reusable machine learning pipeline.
---
# Models Implemented
The following classification models were trained and evaluated:
- Decision Tree (Default)
- Controlled Decision Tree
- Decision Tree (Gini Criterion)
- Decision Tree (Entropy Criterion)
- Random Forest
- Gradient Boosting
- Tuned Random Forest (GridSearchCV)
---
# Decision Tree Analysis
## Default Decision Tree

| Metric | Value |
|---------|------:|
| Training Accuracy | 1.0000 |
| Testing Accuracy | 0.9698 |
| ROC-AUC | 0.7374 |

The default decision tree achieved perfect training accuracy but noticeably lower testing accuracy, indicating overfitting. Decision Trees are considered high-variance models because they greedily split the training data until every leaf becomes nearly pure.
---
## Controlled Decision Tree
Parameters used
```python
max_depth = 5
min_samples_split = 20
```
| Metric | Value |
|---------|------:|
| Training Accuracy | 0.9771 |
| Testing Accuracy | 0.9751 |
| ROC-AUC | 0.7564 |
Limiting the tree depth reduced overfitting and improved the model's ability to generalize to unseen data.

### Parameter Explanation
**max_depth**
Limits the maximum depth of the tree, preventing very deep trees that memorize the training data.
**min_samples_split**
Requires a minimum number of samples before a node can be split, reducing noisy splits.
---
# Gini vs Entropy
Both impurity measures were compared.
| Criterion | Accuracy | ROC-AUC |
|-----------|---------:|---------:|
| Gini | 0.9754 | 0.7388 |
| Entropy | 0.9776 | 0.8003 |

### Gini Impurity
\[
Gini = 1-\sum p_i^2
\]

### Entropy
\[
Entropy=-\sum p_i\log_2(p_i)
\]
A Gini value of **0** indicates that all observations within the node belong to a single class.
Entropy produced slightly better discrimination for this dataset.
---
# Random Forest
Parameters
```python
n_estimators = 100
max_depth = 10
random_state = 42
```
| Metric | Value |
|---------|------:|
| Training Accuracy | 0.9789 |
| Testing Accuracy | 0.9769 |
| ROC-AUC | 0.8759 |
Random Forest substantially improved performance over individual decision trees by combining multiple trees trained on different bootstrap samples.

### Top Important Features
- V258
- V255
- C1
- V245
- C4
Feature importance in Random Forest represents the average reduction in Gini impurity contributed by each feature across all trees in the ensemble.
---
# Feature Ablation Study
The five least important features were removed and the Random Forest model was retrained.
| Model | ROC-AUC |
|--------|---------:|
| Original Model | 0.8759 |
| Reduced Model | 0.8709 |
The small reduction in ROC-AUC suggests that the removed features contributed very little predictive information.
---
# Gradient Boosting
Parameters
```python
n_estimators = 100
learning_rate = 0.1
max_depth = 3
```
Gradient Boosting sequentially builds weak learners, where each new tree attempts to correct errors made by the previous trees.
Its performance was included in the cross-validation comparison.
---

# Cross Validation
Five-fold Stratified Cross Validation was performed using ROC-AUC.
| Model | Mean ROC-AUC | Standard Deviation |
|--------|-------------:|-------------------:|
| Logistic Regression | 0.8647 | Low |
| Controlled Decision Tree | 0.7240 | Moderate |
| Random Forest | 0.8610 | Low |
| Gradient Boosting | **0.8788** | Lowest |
Cross-validation provides a more reliable estimate of model performance than a single train-test split because every sample is used for both training and validation.
---

# Hyperparameter Tuning
GridSearchCV was used to optimize the Random Forest model.
Parameter Grid
```python
n_estimators = [50,100,200]
max_depth = [5,10,None]
min_samples_leaf = [1,5]
```
Total configurations evaluated
```
18 parameter combinations × 5-fold cross validation = 90 model fits
```
The best parameter combination was automatically selected based on ROC-AUC.
Grid Search evaluates every possible parameter combination, whereas Randomized Search evaluates only a subset, making it faster but less exhaustive.
---

# Learning Curve
A learning curve was generated using training fractions of:
- 20%
- 40%
- 60%
- 80%
- 100%
The testing ROC-AUC increased as more training data became available before gradually stabilizing, indicating improved generalization with additional data.
---
# Model Serialization
The final tuned pipeline was saved using Joblib.
```python
joblib.dump(best_pipeline,"best_model.pkl")
```
The saved model can later be reloaded using
```python
model = joblib.load("best_model.pkl")
prediction = model.predict(new_data)
```
This enables the trained model to be reused without retraining.
---
# Final Model Comparison
| Model | Test ROC-AUC |
|--------|-------------:|
| Decision Tree | 0.7374 |
| Controlled Decision Tree | 0.7564 |
| Random Forest | 0.8759 |
| Gradient Boosting | 0.8772 |
| Tuned Random Forest | **0.8961** |
---
# Recommended Model
The **Tuned Random Forest** was selected as the final deployment model because it achieved the highest ROC-AUC while maintaining excellent test accuracy and strong generalization performance after cross-validation and hyperparameter tuning.
---
# Conclusion
Part 3 successfully demonstrated the use of ensemble learning techniques, feature importance analysis, feature ablation, cross-validation, hyperparameter optimization, learning curve analysis, and model serialization. Compared to a single Decision Tree, the ensemble models significantly improved fraud detection performance. The tuned Random Forest achieved the best overall performance and was selected as the final production-ready model.