import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import os

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("MLOps_CICD_experiment")

# Hugging Face API authentication
api = HfApi(token=os.getenv("HF_TOKEN"))
Xtrain_path = "hf://datasets/rakeshkotha1/tourism-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/rakeshkotha1/tourism-prediction/Xtest.csv"
ytrain_path = "hf://datasets/rakeshkotha1/tourism-prediction/ytrain.csv"
ytest_path = "hf://datasets/rakeshkotha1/tourism-prediction/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

numeric_features = [
    'Age',                     
    'CityTier',                
    'DurationOfPitch',         
    'NumberOfPersonVisiting',  
    'NumberOfFollowups',       
    'PreferredPropertyStar',   
    'NumberOfTrips',          
    'PitchSatisfactionScore', 
    'NumberOfChildrenVisiting', 
    'MonthlyIncome',
    'Passport',   
    'OwnCar'   
]
categorical_features = [
    'TypeofContact',   
    'Occupation',      
    'Gender',          
    'ProductPitched',  
    'MaritalStatus',   
    'Designation'      
]
# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    mlflow.set_tag("run_id", "run_id")
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.45

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "tourism_prediction.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
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
        path_or_fileobj="tourism_prediction.joblib",
        path_in_repo="tourism_prediction.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )