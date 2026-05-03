import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from sklearn.preprocessing import LabelEncoder

# Support both Google Colab secrets and environment variables
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except ImportError:
    hf_token = os.getenv("HF_TOKEN")

repo_id = "anujgrover/tourism-project"
repo_type = "dataset"

# ── Step 1: Load dataset from Hugging Face dataset space ──────────────────────
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Step 2: Data Cleaning ─────────────────────────────────────────────────────

# Drop unnecessary columns (CustomerID is just an identifier, not a predictor)
df.drop(columns=["CustomerID"], inplace=True)
print("Dropped 'CustomerID' column.")

# Report missing values
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing > 0])

# Impute missing values
# Numeric columns → fill with median
numeric_cols = df.select_dtypes(include="number").columns.tolist()
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Categorical columns → fill with mode
categorical_cols = df.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\nAfter cleaning — missing values: {df.isnull().sum().sum()}")
print(f"Cleaned dataset shape: {df.shape}")

# Encoding the categorial columns
label_encoders= LabelEncoder()
df['TypeofContact'] = label_encoders.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoders.fit_transform(df['Occupation'])
df['Gender'] = label_encoders.fit_transform(df['Gender'])
df['MaritalStatus'] = label_encoders.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoders.fit_transform(df['Designation'])
df['ProductPitched'] = label_encoders.fit_transform(df['ProductPitched'])


# ── Step 3: Split into train / test sets ──────────────────────────────────────
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y      # preserve class balance
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# Save splits locally
os.makedirs("data", exist_ok=True) # Corrected path
X_train.to_csv("data/X_train.csv", index=False) # Corrected path
X_test.to_csv("data/X_test.csv",  index=False) # Corrected path
y_train.to_csv("data/y_train.csv", index=False) # Corrected path
y_test.to_csv("data/y_test.csv",  index=False) # Corrected path
print("Splits saved locally under data/") # Updated message

# ── Step 4: Upload train/test datasets back to Hugging Face ───────────────────
api = HfApi(token=hf_token)

split_files = [
    "data/X_train.csv", # Corrected path
    "data/X_test.csv",  # Corrected path
    "data/y_train.csv", # Corrected path
    "data/y_test.csv",  # Corrected path
]

for file_path in split_files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded: {os.path.basename(file_path)}")

print("\nAll split files uploaded to Hugging Face dataset repo successfully.")
