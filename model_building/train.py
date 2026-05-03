# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Project")

api = HfApi()

Xtrain_path = "hf://datasets/anujgrover/tourism-project/Xtrain.csv"
Xtest_path = "hf://datasets/anujgrover/tourism-project/Xtest.csv"
ytrain_path = "hf://datasets/anujgrover/tourism-project/ytrain.csv"
ytest_path = "hf://datasets/anujgrover/tourism-project/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Drop unnecessary columns (CustomerID and Unnamed: 0 are identifiers, not predictors)
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)
print("Dropped 'Unnamed: 0' and 'CustomerID' columns.")

# Define target column
target_col = 'ProdTaken'

# Encoding the categorial columns
label_encoders= LabelEncoder()
df['TypeofContact'] = label_encoders.fit_transform(df['TypeofContact'])
df['Occupation'] = label_encoders.fit_transform(df['Occupation'])
df['Gender'] = label_encoders.fit_transform(df['Gender'])
df['MaritalStatus'] = label_encoders.fit_transform(df['MaritalStatus'])
df['Designation'] = label_encoders.fit_transform(df['Designation'])
df['ProductPitched'] = label_encoders.fit_transform(df['ProductPitched'])


# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numerical and categorical features based on data description and info
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore'
]
categorical_features = [
    'TypeofContact',
    'CityTier', # CityTier is an ordinal categorical feature, suitable for OneHotEncoder
    'Occupation',
    'Gender',
    'MaritalStatus',
    'ProductPitched',
    'Passport', # Binary features can be treated as categorical
    'OwnCar',   # Binary features can be treated as categorical
    'Designation'
]


# Set the class weight to handle class imbalance
# Note: This should ideally be calculated based on ytrain after splitting
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Calculated class weight: {class_weight}")

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1) 
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

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
    mlflow.sklearn.log_model(best_model, "best_xgboost_model") # Log the best model

    # Save the model locally
    model_path = "best_tourism_project_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "anujgrover/tourism-project"
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
        path_or_fileobj="best_tourism_project_model_v1.joblib",
        path_in_repo="best_tourism_project_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
