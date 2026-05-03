# Tourism Package Purchase Prediction — MLOps Pipeline

An end-to-end MLOps pipeline that predicts whether a customer will purchase the **Wellness Tourism Package** offered by "Visit with Us," a leading travel company.

## Live Demo

- **Streamlit App**: [https://huggingface.co/spaces/anujgrover/tourism-project](https://huggingface.co/spaces/anujgrover/tourism-project)

---

## Business Context

The company faces challenges in manually identifying potential buyers for new tourism packages — a process that is inconsistent, time-consuming, and error-prone. This project implements a scalable, automated ML system that:

- Predicts potential buyers before contacting them
- Integrates data preprocessing, model training, deployment, and CI/CD
- Adapts to evolving customer behaviors through continuous retraining

---

## Project Structure

```
tourism-project/
├── model_building/
│   ├── prep.py              # Data cleaning & train-test split
│   └── train.py             # Model training with hyperparameter tuning
├── deployment/
│   ├── app.py               # Streamlit prediction app
│   ├── Dockerfile           # Container configuration for HF Spaces
│   ├── hosting.py           # Push deployment files to HF Space
│   └── requirements.txt     # App dependencies
├── .github/
│   └── workflows/
│       └── pipeline.yml     # CI/CD workflow
├── requirements_pipeline.txt # Pipeline dependencies
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9 |
| ML Model | XGBoost |
| Preprocessing | scikit-learn (StandardScaler, OneHotEncoder, ColumnTransformer) |
| Hyperparameter Tuning | GridSearchCV (5-fold CV) |
| Experiment Tracking | MLflow |
| Data & Model Registry | Hugging Face Hub |
| Deployment | Streamlit + Docker |
| Hosting | Hugging Face Spaces |
| CI/CD | GitHub Actions |

---

## Dataset

The dataset contains **4,888 records** with customer demographics and interaction data:

**Customer Details**: Age, Gender, Occupation, CityTier, MaritalStatus, MonthlyIncome, Passport, OwnCar, NumberOfTrips, NumberOfPersonVisiting, NumberOfChildrenVisiting, PreferredPropertyStar, Designation

**Interaction Data**: TypeofContact, PitchSatisfactionScore, ProductPitched, NumberOfFollowups, DurationOfPitch

**Target**: `ProdTaken` — Whether the customer purchased the package (0: No, 1: Yes)

---

## Pipeline Stages

### 1. Data Registration
- Raw dataset (`tourism.csv`) is uploaded to the Hugging Face dataset repository as a single source of truth.

### 2. Data Preparation (`prep.py`)
- Drops `CustomerID` (non-predictive identifier)
- Imputes missing values: **median** for numeric, **mode** for categorical
- Splits data: 80% train / 20% test with stratified sampling
- Uploads split files to Hugging Face dataset repo

### 3. Model Training (`train.py`)
- Loads train/test splits from Hugging Face
- Applies `StandardScaler` on numeric features, `OneHotEncoder` on categorical features
- Handles class imbalance using `scale_pos_weight`
- Performs GridSearchCV over XGBoost hyperparameters
- Uses custom classification threshold of **0.45** to improve recall
- Logs all experiments to MLflow
- Saves best model to Hugging Face Model Hub

### 4. Deployment
- Streamlit app loads model from Hugging Face Model Hub
- Containerized with Docker for Hugging Face Spaces
- Interactive UI for real-time predictions

---

## Model Performance

| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.89 | 0.84 |
| Precision (Class 1) | 0.82 | 0.71 |
| Recall (Class 1) | 0.78 | 0.72 |
| F1-Score (Class 1) | 0.80 | 0.71 |

**Best Hyperparameters:**

| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 4 |
| colsample_bytree | 0.6 |
| colsample_bylevel | 0.6 |
| learning_rate | 0.1 |
| reg_lambda | 0.4 |

---

## CI/CD Pipeline (GitHub Actions)

The workflow triggers automatically on every push to `main`:

```
register-dataset → data-prep → model-training → deploy-hosting
```

Each job:
1. Checks out the repository
2. Sets up Python 3.9
3. Installs dependencies from `requirements_pipeline.txt`
4. Runs the respective script with `HF_TOKEN` from GitHub Secrets

---

## Setup & Usage

### Prerequisites
- Python 3.9+
- GitHub account with repository access
- Hugging Face account with API token

### Environment Variables / Secrets

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face API token (read/write) |
| `GH_TOKEN` | GitHub Personal Access Token (for pushing) |

### Local Development

```bash
# Clone the repository
git clone https://github.com/anujgrover/tourism-project.git
cd tourism-project

# Install dependencies
pip install -r requirements_pipeline.txt

# Run data preparation
python model_building/prep.py

# Run model training (requires MLflow server)
mlflow ui --host 0.0.0.0 --port 5000 &
python model_building/train.py

# Run the Streamlit app locally
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

---

## Links

- **GitHub Repository**: [https://github.com/anujgrover/tourism-project](https://github.com/anujgrover/tourism-project)
- **Hugging Face Space (App)**: [https://huggingface.co/spaces/anujgrover/tourism-project](https://huggingface.co/spaces/anujgrover/tourism-project)
- **Hugging Face Dataset**: [https://huggingface.co/datasets/anujgrover/tourism-project](https://huggingface.co/datasets/anujgrover/tourism-project)
- **Hugging Face Model**: [https://huggingface.co/anujgrover/tourism-project](https://huggingface.co/anujgrover/tourism-project)
