import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi

# Set MLflow tracking URI to HTTP server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_CICD_experiment")

# Initialize HfApi
api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "rakeshkotha1/tourism-prediction"
repo_type = "dataset" # Corrected to 'dataset' for downloading

# Download the preprocessed data from Hugging Face
print("Downloading preprocessed data from Hugging Face...")
api.hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type=repo_type, local_dir=".")
api.hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type=repo_type, local_dir=".")
api.hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type=repo_type, local_dir=".")
api.hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type=repo_type, local_dir=".")
print("Preprocessed data downloaded successfully.")

# Load the preprocessed data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze() # .squeeze() to convert DataFrame to Series
ytest = pd.read_csv("ytest.csv").squeeze()

# Drop 'Unnamed: 0' if it exists in the loaded dataframes
if 'Unnamed: 0' in Xtrain.columns:
    Xtrain = Xtrain.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0' in Xtest.columns:
    Xtest = Xtest.drop(columns=['Unnamed: 0'])

print(f"Xtrain shape: {Xtrain.shape}, ytrain shape: {ytrain.shape}")
print(f"Xtest shape: {Xtest.shape}, ytest shape: {ytest.shape}")

# --- Diagnostic Step: Check for non-numeric columns before training ---
print("\nChecking Xtrain dtypes before model training:")
print(Xtrain.info())
non_numeric_cols_xtrain = Xtrain.select_dtypes(include=['object']).columns
if len(non_numeric_cols_xtrain) > 0:
    print(f"Found non-numeric columns in Xtrain: {list(non_numeric_cols_xtrain)}")
    print("Please ensure all features are numerical before proceeding with model training.")
    raise ValueError("Non-numeric columns detected in Xtrain. Cannot proceed with StandardScaler.")

print("\nChecking Xtest dtypes before model training:")
print(Xtest.info())
non_numeric_cols_xtest = Xtest.select_dtypes(include=['object']).columns
if len(non_numeric_cols_xtest) > 0:
    print(f"Found non-numeric columns in Xtest: {list(non_numeric_cols_xtest)}")
    print("Please ensure all features are numerical before proceeding with model training.")
    raise ValueError("Non-numeric columns detected in Xtest. Cannot proceed with StandardScaler.")
# -------------------------------------------------------------------

# Define base XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')

# Preprocessor: Scale numerical features (all features are now numerical after label encoding)
# No need for ColumnTransformer as all features are treated the same way
preprocessor = StandardScaler()

# Pipeline: Preprocessor + Classifier
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Hyperparameter grid for XGBClassifier (reduced for quicker execution)
param_grid = {
    'xgbclassifier__n_estimators': [100, 150],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.8, 0.9],
    'xgbclassifier__colsample_bytree': [0.8, 0.9],
    'xgbclassifier__gamma': [0.1, 0.2]
}

with mlflow.start_run():
    # Grid Search with cross-validation
    # Using 'roc_auc' as a scoring metric for binary classification
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets for each run
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Log best parameters and best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions on train and test sets
    y_pred_train = best_model.predict(Xtrain)
    y_proba_train = best_model.predict_proba(Xtrain)[:, 1] # Probability for the positive class
    y_pred_test = best_model.predict(Xtest)
    y_proba_test = best_model.predict_proba(Xtest)[:, 1] # Probability for the positive class

    # Calculate classification metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)

    train_recall = recall_score(ytrain, y_pred_train, zero_division=0)
    test_recall = recall_score(ytest, y_pred_test, zero_division=0)

    train_f1 = f1_score(ytrain, y_pred_train, zero_division=0)
    test_f1 = f1_score(ytest, y_pred_test, zero_division=0)

    # ROC AUC requires probabilities
    train_roc_auc = roc_auc_score(ytrain, y_proba_train)
    test_roc_auc = roc_auc_score(ytest, y_proba_test)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1_score": train_f1,
        "test_f1_score": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })
    model_path = "tourism_prediction.joblib"
    joblib.dump(best_model, model_path)
    # Log the best model with an input example
    mlflow.sklearn.log_model(best_model, "xgboost_tourism_model", input_example=Xtest.head(1))

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "rakeshkotha1/tourism-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_machine_failure_model_v1.joblib",
        path_in_repo="best_machine_failure_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
