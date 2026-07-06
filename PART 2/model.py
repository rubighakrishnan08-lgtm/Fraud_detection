import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler,OneHotEncoder)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression,Ridge,LogisticRegression)
from sklearn.metrics import (mean_squared_error,r2_score,confusion_matrix, classification_report,roc_curve,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score)

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = Path("cleaned_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# LOAD DATASET
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(df.head())

# VALIDATION
required_columns = ["TransactionAmt","isFraud"]
missing = [col for col in required_columns
           if col not in df.columns]
if len(missing) > 0:
    raise ValueError(
        f"Missing Columns : {missing}")
print("\nValidation Completed")

# TARGET VARIABLES
y_reg = df["TransactionAmt"]
y_clf = df["isFraud"].astype(int)

# FEATURE MATRIX
drop_columns = ["TransactionAmt","isFraud"]
if "TransactionID" in df.columns:
    drop_columns.append("TransactionID")
X = df.drop(columns=drop_columns,errors="ignore")
print(f"Rows    : {X.shape[0]}")
print(f"Columns : {X.shape[1]}")

# FEATURE TYPES
numeric_features = X.select_dtypes(include=[np.number,"bool" ]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number,"bool"]).columns.tolist()
print(f"Numeric Features     : {len(numeric_features)}")
print(f"Categorical Features : {len(categorical_features)}")

# TRAIN TEST SPLIT
# Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X,y_reg,test_size=TEST_SIZE,random_state=RANDOM_STATE)
# Classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X,y_clf,test_size=TEST_SIZE,stratify=y_clf,random_state=RANDOM_STATE)
print("Train-Test Split Completed")
print(X_train_reg.shape,X_test_reg.shape)
print(X_train_clf.shape,X_test_clf.shape)

# PREPROCESSING PIPELINE
numeric_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())])
categorical_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore",drop="first"))])
preprocessor = ColumnTransformer(transformers=[
    ("numeric",numeric_pipeline,numeric_features),
    ("categorical",categorical_pipeline,categorical_features)])
feature_names = None
print("\nPreprocessing Pipeline Created Successfully")

# REGRESSION PREPROCESSING
preprocessor.fit(X_train_reg)
print("Preprocessor fitted successfully.")
X_train_reg_scaled = preprocessor.transform(X_train_reg)
X_test_reg_scaled = preprocessor.transform(X_test_reg)
feature_names = preprocessor.get_feature_names_out()
print("\nProcessed Training Shape :", X_train_reg_scaled.shape)
print("Processed Testing Shape  :", X_test_reg_scaled.shape)
print("Total Features :", len(feature_names))

# LINEAR REGRESSION
linear_model = LinearRegression()
linear_model.fit(X_train_reg_scaled,y_train_reg)
print("Linear Regression Training Completed")

# PREDICTIONS
linear_predictions = linear_model.predict(X_test_reg_scaled)

# METRICS
linear_mse = mean_squared_error(y_test_reg,linear_predictions)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test_reg,linear_predictions)
print(f"MSE  : {linear_mse:.4f}")
print(f"RMSE : {linear_rmse:.4f}")
print(f"R²   : {linear_r2:.4f}")

# RIDGE REGRESSION
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg_scaled,y_train_reg)
print("Ridge Regression Training Completed")
ridge_predictions = ridge_model.predict(X_test_reg_scaled)
ridge_mse = mean_squared_error(y_test_reg,ridge_predictions)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test_reg,ridge_predictions)
print(f"MSE  : {ridge_mse:.4f}")
print(f"RMSE : {ridge_rmse:.4f}")
print(f"R²   : {ridge_r2:.4f}")

# COMPARISON TABLE
comparison = pd.DataFrame({"Model":["Linear Regression","Ridge Regression"],
                           "MSE":[linear_mse,ridge_mse],
                           "RMSE":[linear_rmse,ridge_rmse],
                           "R2":[linear_r2,ridge_r2]})
print(comparison)
comparison.to_csv(OUTPUT_DIR /"Regression_Model_Comparison.csv",index=False)

# COEFFICIENTS
coefficients = pd.DataFrame({"Feature":feature_names,"Coefficient":linear_model.coef_})
coefficients["Absolute"] = np.abs(coefficients["Coefficient"])
coefficients = coefficients.sort_values("Absolute",ascending=False)
print("\nTop 10 Coefficients")
print(coefficients.head(10))
coefficients.to_csv(OUTPUT_DIR /"Linear_Regression_Coefficients.csv",index=False)
print("\nTop 3 Features")
print(coefficients.head(3))

# SAVE MODELS
joblib.dump(linear_model,OUTPUT_DIR /"LinearRegression.pkl")
joblib.dump(ridge_model,OUTPUT_DIR /"RidgeRegression.pkl")
joblib.dump(preprocessor,OUTPUT_DIR /"Regression_Preprocessor.pkl")
print("\nRegression Models Saved Successfully")

# LOGISTIC REGRESSION CLASSIFICATION
# PREPROCESS DATA
X_train_clf_scaled = preprocessor.fit_transform(X_train_clf)
X_test_clf_scaled = preprocessor.transform(X_test_clf)
print("Classification Dataset Ready")
print("Training Shape :", X_train_clf_scaled.shape)
print("Testing Shape  :", X_test_clf_scaled.shape)

# CLASS DISTRIBUTION
print(y_train_clf.value_counts())
print("\nPercentage")
print((y_train_clf.value_counts(normalize=True)*100).round(2))

# LOGISTIC REGRESSION
logistic_model = LogisticRegression(class_weight="balanced",C=1.0,max_iter=1000,random_state=RANDOM_STATE)
logistic_model.fit(X_train_clf_scaled,y_train_clf)
print("Training Completed")

# PREDICTIONS
clf_predictions = logistic_model.predict(X_test_clf_scaled)
clf_probabilities = logistic_model.predict_proba(X_test_clf_scaled)[:,1]

# METRICS
accuracy = accuracy_score(y_test_clf,clf_predictions)
precision = precision_score(y_test_clf,clf_predictions,zero_division=0)
recall = recall_score(y_test_clf,clf_predictions,zero_division=0)
f1 = f1_score(y_test_clf,clf_predictions,zero_division=0)
auc = roc_auc_score(y_test_clf,clf_probabilities)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

# CONFUSION MATRIX
cm = confusion_matrix(y_test_clf,clf_predictions)
print(cm)

# CLASSIFICATION REPORT
print("\nClassification Report")
print(classification_report( y_test_clf,clf_predictions,digits=4,zero_division=0))

# SAVE CONFUSION MATRIX
cm_df = pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Pred 0","Pred 1"])
cm_df.to_csv(OUTPUT_DIR/"Confusion_Matrix.csv")

# ROC CURVE
fpr,tpr,thresholds = roc_curve(y_test_clf,clf_probabilities)
plt.figure(figsize=(7,6))
plt.plot(fpr,tpr,label=f"AUC = {auc:.4f}")
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"ROC_Curve.png",dpi=300)
plt.close()

# SAVE PREDICTIONS
prediction_df = pd.DataFrame({"Actual":y_test_clf,"Predicted":clf_predictions,"Probability":clf_probabilities})
prediction_df.to_csv(OUTPUT_DIR/"Classification_Predictions.csv",index=False)

# SAVE MODEL
joblib.dump(logistic_model,OUTPUT_DIR/"LogisticRegression.pkl")
print("\nLogistic Regression Saved")
print("ROC Curve Saved")
print("Prediction File Saved")

# THRESHOLD ANALYSIS + REGULARIZATION + BOOTSTRAP
thresholds = np.arange(0.30, 0.71, 0.10)
threshold_results = []
for threshold in thresholds:
    predictions = (clf_probabilities >= threshold).astype(int)
    precision = precision_score( y_test_clf,predictions,zero_division=0)
    recall = recall_score(y_test_clf,predictions,zero_division=0)
    f1 = f1_score(y_test_clf,predictions,zero_division=0)
    threshold_results.append([threshold,precision,recall,f1])
threshold_df = pd.DataFrame(threshold_results,
                            columns=["Threshold","Precision","Recall","F1"])
print(threshold_df)
threshold_df.to_csv(OUTPUT_DIR/"Threshold_Analysis.csv",index=False)
best_row = threshold_df.iloc[threshold_df["F1"].idxmax()]
best_threshold = best_row["Threshold"]
print("\nBest Threshold")
print(best_row)

# STRONG REGULARIZATION
logistic_regularized = LogisticRegression(class_weight="balanced",C=0.01,max_iter=1000,random_state=RANDOM_STATE)
logistic_regularized.fit(X_train_clf_scaled,y_train_clf)
regularized_predictions = logistic_regularized.predict(X_test_clf_scaled)
regularized_probabilities = logistic_regularized.predict_proba( X_test_clf_scaled)[:,1]
precision_regularized = precision_score(y_test_clf,regularized_predictions, zero_division=0)
recall_regularized = recall_score(y_test_clf,regularized_predictions,zero_division=0)
f1_regularized = f1_score(y_test_clf,regularized_predictions,zero_division=0)
auc_regularized = roc_auc_score(y_test_clf,regularized_probabilities)
print(f"Precision : {precision_regularized:.4f}")
print(f"Recall    : {recall_regularized:.4f}")
print(f"F1 Score  : {f1_regularized:.4f}")
print(f"ROC AUC   : {auc_regularized:.4f}")
comparison = pd.DataFrame({
    "Model":["Logistic C=1.0", "Logistic C=0.01"],
    "Precision":[ precision,precision_regularized],
    "Recall":[recall,recall_regularized],
    "F1":[f1,f1_regularized],
    "AUC":[auc, auc_regularized]})
print("\nRegularization Comparison")
print(comparison)
comparison.to_csv(OUTPUT_DIR/"Regularization_Comparison.csv",index=False)

# BOOTSTRAP AUC CONFIDENCE INTERVAL
iterations = 500
auc_difference = []
rng = np.random.default_rng(RANDOM_STATE)
for _ in range(iterations):
    sample = rng.choice(len(y_test_clf),size=len(y_test_clf),replace=True)
    y_sample = y_test_clf.iloc[sample]
    prob1 = clf_probabilities[sample]
    prob2 = regularized_probabilities[sample]
    auc1 = roc_auc_score(y_sample,prob1)
    auc2 = roc_auc_score(y_sample,prob2)
    auc_difference.append(auc1 - auc2)
auc_difference = np.array(auc_difference)
mean_difference = auc_difference.mean()
lower = np.percentile(auc_difference,2.5)
upper = np.percentile(auc_difference,97.5)
print(f"\nMean Difference : {mean_difference:.6f}")
print(f"95% CI Lower    : {lower:.6f}")
print(f"95% CI Upper    : {upper:.6f}")
bootstrap = pd.DataFrame({
    "Mean Difference":[mean_difference], "Lower CI":[lower],"Upper CI":[upper]})
bootstrap.to_csv(OUTPUT_DIR/"Bootstrap_AUC.csv",index=False)

# SAVE MODELS
joblib.dump(logistic_regularized,OUTPUT_DIR/"LogisticRegression_C001.pkl")
print("\nRegularized Model Saved")
print("Bootstrap Results Saved")
print("Threshold Analysis Saved")
