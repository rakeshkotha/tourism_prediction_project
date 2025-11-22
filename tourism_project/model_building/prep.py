# for creating a folder
import os
# for data manipulation
import pandas as pd
import sklearn
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/rakeshkotha1/tourism-prediction/tourism.csv"
tourism_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = 'ProdTaken'   # 1 if the customer purchased the package, else 0

# ----------------------------
# List of numerical features
# ----------------------------
numeric_features = [
    'Age',                     # Customer's age
    'CityTier',                # Development tier of the customer's city (1 > 2 > 3)
    'DurationOfPitch',         # Duration of the pitch given to the customer
    'NumberOfPersonVisiting',  # Number of people visiting with the customer
    'NumberOfFollowups',       # Number of follow-up interactions
    'PreferredPropertyStar',   # Preferred hotel star rating
    'NumberOfTrips',           # Total number of previous trips
    'PitchSatisfactionScore',  # Satisfaction score given by the customer
    'NumberOfChildrenVisiting', # Number of children who visited with the customer
    'MonthlyIncome',           # Monthly income of the customer
    'Passport',                # Whether customer has a passport (0 or 1)
    'OwnCar'                   # Whether customer owns a car (0 or 1)
]

# ----------------------------
# List of categorical features
# ----------------------------
categorical_features = [
    'TypeofContact',   # How the customer was contacted (e.g., Self Enquiry, Company Invited)
    'Occupation',      # Customer's job type
    'Gender',          # Male/Female/Others
    'ProductPitched',  # Type of product pitched (e.g., Basic, Deluxe, King)
    'MaritalStatus',   # Marital status of the customer
    'Designation'      # Job designation level (e.g., Executive, Manager, VP)
]


# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = tourism_df[numeric_features + categorical_features]

# ----------------------------
# Define target vector y
# ----------------------------
y = tourism_df[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="rakeshkotha1/tourism-prediction",
        repo_type="dataset",
    )