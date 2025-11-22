# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/rakeshkotha1/tourism-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)

# Drop 'Unnamed: 0' if it exists (artifact from previous saves)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Identify categorical columns (including 'Designation')
categorical_cols = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'ProductPitched',
    'Designation' # Added 'Designation' here
]

# Handle NaNs in categorical columns before encoding
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown') # Fill NaNs with a new category 'Unknown'

# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col]).astype(int) # Explicitly cast to int
    else:
        print(f"Warning: Categorical column '{col}' not found in DataFrame.")

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# Define repo_id for uploading processed data
repo_id = "rakeshkotha1/tourism-prediction"

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
