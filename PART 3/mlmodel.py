import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (train_test_split,StratifiedKFold,cross_val_score,GridSearchCV)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, roc_auc_score,roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = Path("../PART 1/cleaned_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print("Dataset Loaded Successfully")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# VALIDATION
required_columns = ["TransactionAmt","isFraud"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"{col} not found.")
print("Validation Successful")

# TARGETS
y = df["isFraud"].astype(int)
drop_columns = ["isFraud","TransactionAmt"]
if "TransactionID" in df.columns:
    drop_columns.append("TransactionID" )
X = df.drop(columns=drop_columns,errors="ignore")
print(X.shape)

# FEATURE TYPES
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
print("\nFeature Summary")
print(f"Numeric     : {len(numeric_features)}")
print(f"Categorical : {len(categorical_features)}")

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# PREPROCESSOR
numeric_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),
                                   ("scaler",StandardScaler())])
categorical_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                       ("encoder",OneHotEncoder(handle_unknown="ignore",drop="first"))])
preprocessor = ColumnTransformer(transformers=[("numeric",numeric_pipeline,numeric_features),
                                               ("categorical",categorical_pipeline,categorical_features)])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()
print("Preprocessing Completed")
print(f"Training Shape : {X_train_processed.shape}")
print(f"Testing Shape  : {X_test_processed.shape}")
print(f"Total Features : {len(feature_names)}")

# DECISION TREE MODELS
decision_tree_results = []
# DEFAULT DECISION TREE

default_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
default_tree.fit(X_train_processed,y_train)
default_train_accuracy = accuracy_score(y_train,default_tree.predict(X_train_processed))
default_test_accuracy = accuracy_score(y_test,default_tree.predict(X_test_processed))
default_auc = roc_auc_score( y_test, default_tree.predict_proba(X_test_processed)[:,1])
print("\nDefault Decision Tree")
print(f"Training Accuracy : {default_train_accuracy:.4f}")
print(f"Testing Accuracy  : {default_test_accuracy:.4f}")
print(f"ROC-AUC           : {default_auc:.4f}")
decision_tree_results.append({
    "Model":"Default Decision Tree",
    "Train Accuracy":default_train_accuracy,
    "Test Accuracy":default_test_accuracy,
    "ROC AUC":default_auc})

# CONTROLLED DECISION TREE
controlled_tree = DecisionTreeClassifier(max_depth=5,min_samples_split=20,random_state=RANDOM_STATE)
controlled_tree.fit(X_train_processed,y_train)
controlled_train_accuracy = accuracy_score(y_train,controlled_tree.predict(X_train_processed))
controlled_test_accuracy = accuracy_score(y_test,controlled_tree.predict(X_test_processed))
controlled_auc = roc_auc_score(y_test,controlled_tree.predict_proba(X_test_processed)[:,1])
print("\nControlled Decision Tree")
print(f"Training Accuracy : {controlled_train_accuracy:.4f}")
print(f"Testing Accuracy  : {controlled_test_accuracy:.4f}")
print(f"ROC-AUC           : {controlled_auc:.4f}")
decision_tree_results.append({
    "Model":"Controlled Decision Tree",
    "Train Accuracy":controlled_train_accuracy,
    "Test Accuracy":controlled_test_accuracy,
    "ROC AUC":controlled_auc})

# GINI DECISION TREE
gini_tree = DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=RANDOM_STATE)
gini_tree.fit(X_train_processed,y_train)
gini_accuracy = accuracy_score(y_test,gini_tree.predict(X_test_processed))
gini_auc = roc_auc_score(y_test,gini_tree.predict_proba(X_test_processed)[:,1])
print(f"Gini Accuracy : {gini_accuracy:.4f}")
print(f"Gini ROC-AUC  : {gini_auc:.4f}")

# ENTROPY DECISION TREE
entropy_tree = DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=RANDOM_STATE)
entropy_tree.fit( X_train_processed,y_train)
entropy_accuracy = accuracy_score(y_test,entropy_tree.predict(X_test_processed))
entropy_auc = roc_auc_score(y_test,entropy_tree.predict_proba(X_test_processed)[:,1])
print(f"Entropy Accuracy : {entropy_accuracy:.4f}")
print(f"Entropy ROC-AUC  : {entropy_auc:.4f}")
decision_tree_results.append({
    "Model":"Gini Decision Tree",
    "Train Accuracy":accuracy_score(y_train,gini_tree.predict(X_train_processed) ),
    "Test Accuracy":gini_accuracy,
    "ROC AUC":gini_auc})
decision_tree_results.append({
    "Model":"Entropy Decision Tree", 
    "Train Accuracy":accuracy_score(y_train,entropy_tree.predict(X_train_processed)),
    "Test Accuracy":entropy_accuracy,
    "ROC AUC":entropy_auc})

# RESULTS TABLE
decision_tree_results = pd.DataFrame(decision_tree_results)
print("\nDecision Tree Comparison")
print(decision_tree_results)
decision_tree_results.to_csv(OUTPUT_DIR /"decision_tree_comparison.csv",index=False)

# SAVE BEST TREE
best_tree = decision_tree_results.sort_values(by="ROC AUC",ascending=False).iloc[0]["Model"]
if best_tree == "Default Decision Tree":
    joblib.dump(default_tree,OUTPUT_DIR /"best_decision_tree.pkl")
elif best_tree == "Controlled Decision Tree":
    joblib.dump(controlled_tree,OUTPUT_DIR /"best_decision_tree.pkl")
elif best_tree == "Gini Decision Tree":
    joblib.dump(gini_tree,OUTPUT_DIR /"best_decision_tree.pkl")
else:
    joblib.dump(entropy_tree,OUTPUT_DIR /"best_decision_tree.pkl" )
print("\nBest Decision Tree Saved")

# RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=RANDOM_STATE,n_jobs=-1)
rf_model.fit(X_train_processed,y_train)

# PREDICTIONS
rf_train_predictions = rf_model.predict(X_train_processed)
rf_test_predictions = rf_model.predict(X_test_processed)
rf_probabilities = rf_model.predict_proba(X_test_processed)[:,1]

# METRICS
rf_train_accuracy = accuracy_score(y_train,rf_train_predictions)
rf_test_accuracy = accuracy_score(y_test,rf_test_predictions)
rf_auc = roc_auc_score(y_test,rf_probabilities)
print("\nRandom Forest Performance")
print(f"Training Accuracy : {rf_train_accuracy:.4f}")
print(f"Testing Accuracy  : {rf_test_accuracy:.4f}")
print(f"ROC AUC           : {rf_auc:.4f}")

# FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature":feature_names,
    "Importance":rf_model.feature_importances_})
importance = importance.sort_values(
    by="Importance",
    ascending=False)
print("\nTop 5 Important Features")

print(importance.head(5))
importance.to_csv(OUTPUT_DIR/"random_forest_feature_importance.csv",index=False)

# FEATURE ABLATION
least_features = importance.tail(5)["Feature"].tolist()
print("\nRemoving Features")
for f in least_features:
    print(f)
feature_index = {name:index
                 for index,name
                 in enumerate(feature_names)}
remove_index = [feature_index[f]
                for f in least_features]
X_train_reduced = np.delete( X_train_processed,remove_index,axis=1)
X_test_reduced = np.delete(X_test_processed,remove_index,axis=1)
rf_reduced = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=RANDOM_STATE,n_jobs=-1)
rf_reduced.fit(X_train_reduced,y_train)
reduced_prob = rf_reduced.predict_proba(X_test_reduced)[:,1]
reduced_auc = roc_auc_score(y_test,reduced_prob)
print("\nFeature Ablation")
print(f"Original AUC : {rf_auc:.4f}")
print(f"Reduced  AUC : {reduced_auc:.4f}")
ablation = pd.DataFrame({
    "Model":["Full Random Forest","Reduced Random Forest"],
    "ROC_AUC":[ rf_auc,reduced_auc]})
ablation.to_csv(OUTPUT_DIR/"feature_ablation_results.csv",index=False)

# SAVE RANDOM FOREST
joblib.dump(rf_model,OUTPUT_DIR/"RandomForest.pkl")
print("\nRandom Forest Saved")

# GRADIENT BOOSTING + CROSS VALIDATION
# GRADIENT BOOSTING
gradient_boosting = GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=RANDOM_STATE)
gradient_boosting.fit(X_train_processed,y_train)
gb_train_predictions = gradient_boosting.predict(X_train_processed)
gb_test_predictions = gradient_boosting.predict(X_test_processed)
gb_probabilities = gradient_boosting.predict_proba(X_test_processed)[:,1]
gb_train_accuracy = accuracy_score(y_train,gb_train_predictions)
gb_test_accuracy = accuracy_score(y_test,gb_test_predictions)
gb_auc = roc_auc_score(y_test,gb_probabilities)
print(f"Training Accuracy : {gb_train_accuracy:.4f}")
print(f"Testing Accuracy  : {gb_test_accuracy:.4f}")
print(f"ROC AUC           : {gb_auc:.4f}")

# SAVE GRADIENT BOOSTING MODEL

joblib.dump(gradient_boosting,OUTPUT_DIR /"GradientBoosting.pkl")

# 5-FOLD CROSS VALIDATION
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
models = {
    "Logistic Regression": LogisticRegression( class_weight="balanced",max_iter=1000,random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5,min_samples_split=20, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier( n_estimators=100,max_depth=10,random_state=RANDOM_STATE,n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier( n_estimators=100,learning_rate=0.1,max_depth=3,random_state=RANDOM_STATE)}
cv_results = []
for name, estimator in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor),("model", estimator)])
    scores = cross_val_score( pipeline,X,y, cv=cv,scoring="roc_auc",n_jobs=-1)
    mean_auc = scores.mean()
    std_auc = scores.std()
    cv_results.append({ "Model": name,"Mean AUC": mean_auc, "Std AUC": std_auc})
    print(f"{name}")
    print(f"Mean AUC : {mean_auc:.4f}")
    print(f"Std AUC  : {std_auc:.4f}")

# SAVE RESULTS
cv_results = pd.DataFrame(cv_results)
cv_results = cv_results.sort_values(by="Mean AUC",ascending=False)
print(cv_results)
cv_results.to_csv(OUTPUT_DIR /"cross_validation_results.csv",index=False)
print("\nGradient Boosting Model Saved")
print("Cross Validation Results Saved")

# GRID SEARCH + LEARNING CURVE
rf_pipeline = Pipeline([("preprocessor", preprocessor),("randomforestclassifier",
                                                        RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=-1,oob_score=True,bootstrap=True))])

# PARAMETER GRID
param_grid = {"randomforestclassifier__n_estimators":[50,100,200],
    "randomforestclassifier__max_depth":[5,10,None],
    "randomforestclassifier__min_samples_leaf":[1, 5]}
print("\nTotal Parameter Combinations :")
total = (3 *3 *2)
print(total)
print("Total Models Evaluated :", total*5)

# GRID SEARCH
grid = GridSearchCV(estimator=rf_pipeline,param_grid=param_grid,scoring="roc_auc",cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE),
    n_jobs=-1,verbose=2,refit=True)
grid.fit(X_train,y_train)
print("\nGrid Search Completed")
print("\nBest Parameters")
print(grid.best_params_)
print("\nBest CV ROC AUC")
print(f"{grid.best_score_:.4f}")
best_pipeline = grid.best_estimator_
best_rf = best_pipeline.named_steps["randomforestclassifier"]
print("\nOut-of-Bag Score")
print(f"{best_rf.oob_score_:.4f}")

# TEST PERFORMANCE
best_probability = best_pipeline.predict_proba(X_test)[:,1]
best_prediction = best_pipeline.predict(X_test)
best_accuracy = accuracy_score(y_test,best_prediction)
best_auc = roc_auc_score(y_test, best_probability)
train_prediction = best_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train,train_prediction)
gap = train_accuracy - best_accuracy
print("\nTraining Accuracy")
print(f"{train_accuracy:.4f}")
print("Testing Accuracy")
print(f"{best_accuracy:.4f}")
print("Train-Test Gap")
print(f"{gap:.4f}")
print("\nBest Model Performance")
print(f"Accuracy : {best_accuracy:.4f}")
print(f"ROC AUC  : {best_auc:.4f}")

# SAVE GRID SEARCH RESULTS
grid_results = pd.DataFrame(grid.cv_results_)
top10 = grid_results.sort_values("rank_test_score").head(10)
top10.to_csv(OUTPUT_DIR/"Top10_GridSearch.csv",index=False)
grid_results.to_csv(OUTPUT_DIR/ "GridSearch_Results.csv",index=False)
joblib.dump(best_pipeline,OUTPUT_DIR/"Best_RandomForest_GridSearch.pkl")
print("\nBest Model Saved")

# MANUAL LEARNING CURVE
fractions = [0.2,0.4,0.6,0.8,1.0]
learning_results = []
for fraction in fractions:
    n = int(len(X_train)* fraction)
    X_subset = X_train.iloc[:n]
    y_subset = y_train.iloc[:n]
    best_pipeline.fit(X_subset,y_subset)
    train_probability = best_pipeline.predict_proba(X_subset)[:,1]
    test_probability = best_pipeline.predict_proba(X_test)[:,1]
    train_auc = roc_auc_score(y_subset, train_probability)
    test_auc = roc_auc_score(y_test,test_probability)
    learning_results.append({"Training Fraction":fraction,
        "Training AUC":train_auc,
        "Testing AUC":test_auc })
learning_results = pd.DataFrame(learning_results)
print("\nLearning Curve")
print(learning_results)
learning_results.to_csv(OUTPUT_DIR/"Learning_Curve.csv",index=False)
plt.figure(figsize=(8,5))
plt.plot(learning_results["Training Fraction"],learning_results["Training AUC"],marker="o",label="Training")
plt.plot(learning_results["Training Fraction"],learning_results["Testing AUC"], marker="s",label="Testing")
plt.xlabel("Training Fraction")
plt.ylabel("ROC-AUC")
plt.title("Learning Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"Learning_Curve.png",dpi=300)
plt.close()

# SAVE BEST MODEL
MODEL_PATH = OUTPUT_DIR / "best_model.pkl"
joblib.dump(best_pipeline,MODEL_PATH)
print("\nModel Saved Successfully")
print(MODEL_PATH)

# LOAD MODEL
loaded_model = joblib.load(MODEL_PATH)
print("\n Model Reloaded Successfully")

# PREDICT TWO SAMPLE ROWS
sample_rows = X_test.iloc[:2]
predictions = loaded_model.predict(sample_rows)
probabilities = loaded_model.predict_proba(sample_rows)[:,1]
print("\nPredictions on Two Sample Rows")
prediction_df = pd.DataFrame({"Actual": y_test.iloc[:2].values,"Predicted": predictions,"Fraud Probability": probabilities})
print(prediction_df)
prediction_df.to_csv(OUTPUT_DIR/"sample_predictions.csv",index=False)

# FINAL MODEL COMPARISON
comparison = pd.DataFrame({"Model":["Decision Tree", "Random Forest","Gradient Boosting", "Best GridSearch RF"],
                           "Test Accuracy":[ controlled_test_accuracy,rf_test_accuracy,gb_test_accuracy,best_accuracy],
                           "ROC AUC":[ controlled_auc, rf_auc,gb_auc,best_auc]})
print(comparison)
comparison.to_csv(OUTPUT_DIR/"Final_Model_Comparison.csv",index=False)

# BEST MODEL
best_row = comparison.loc[comparison["ROC AUC"].idxmax()]
print("\nRecommended Model")
print(f"Model        : {best_row['Model']}")
print(f"ROC AUC      : {best_row['ROC AUC']:.4f}")
print(f"Accuracy     : {best_row['Test Accuracy']:.4f}")


# ALL OUTPUT FILES
files = ["decision_tree_comparison.csv","random_forest_feature_importance.csv","feature_ablation_results.csv","cross_validation_results.csv", "GridSearch_Results.csv","Top10_GridSearch.csv","Learning_Curve.csv",
         "Learning_Curve.png","sample_predictions.csv","Final_Model_Comparison.csv","best_model.pkl"]
for file in files:
    print(file)