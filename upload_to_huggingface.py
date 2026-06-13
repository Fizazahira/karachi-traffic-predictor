"""
Upload trained models to Hugging Face Hub
--------------------------------------------
Run this ONCE to push your .pkl model files to a Hugging Face Model repo.

Steps before running:
1. pip install huggingface_hub
2. Get a token (Write role) from https://huggingface.co/settings/tokens
3. Set it as an environment variable:
   PowerShell:  $env:HF_TOKEN="hf_xxxxxxxxxxxx"
4. Edit REPO_ID below to your_username/your-repo-name
5. python upload_to_huggingface.py
"""

import os
from huggingface_hub import HfApi, create_repo

# -----------------------------
# CONFIG - edit these
# -----------------------------
REPO_ID = "Fizazahira/karachi-traffic-models"  # change this
REPO_TYPE = "model"  # keep as "model"

TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "HF_TOKEN environment variable not set.\n"
        "Run: $env:HF_TOKEN=\"hf_xxxxxxxxxxxx\"  (PowerShell)\n"
        "Get a token from https://huggingface.co/settings/tokens (Write role)"
    )

FILES_TO_UPLOAD = [
    "congestion_model.pkl",
    "traveltime_model.pkl",
    "route_encoder.pkl",
    "day_encoder.pkl",
    "congestion_encoder.pkl",
]

# -----------------------------
# Create repo (if it doesn't exist) and upload
# -----------------------------
api = HfApi(token=TOKEN)

create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True, token=TOKEN)

for filename in FILES_TO_UPLOAD:
    print(f"Uploading {filename} ...")
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=TOKEN,
    )

print("\nDone! Your models are now at:")
print(f"https://huggingface.co/{REPO_ID}")