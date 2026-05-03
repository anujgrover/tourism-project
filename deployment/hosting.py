import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Support both Google Colab secrets and environment variables
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except ImportError:
    hf_token = os.getenv("HF_TOKEN")

# Hugging Face Space configuration
space_id = "anujgrover/tourism-project"
repo_type = "space"

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the space exists, create if not
try:
    api.repo_info(repo_id=space_id, repo_type=repo_type)
    print(f"Space '{space_id}' already exists. Updating files...")
except RepositoryNotFoundError:
    print(f"Space '{space_id}' not found. Creating new space...")
    create_repo(
        repo_id=space_id,
        repo_type=repo_type,
        space_sdk="docker",
        private=False,
        token=hf_token,
    )
    print(f"Space '{space_id}' created with Docker SDK.")

# Step 2: Upload all deployment files to the Hugging Face Space
api.upload_folder(
    folder_path="deployment", # Corrected path
    repo_id=space_id,
    repo_type=repo_type,
)
print(f"\nAll deployment files pushed to Hugging Face Space: https://huggingface.co/spaces/{space_id}")
