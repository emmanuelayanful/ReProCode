import os
import json
from datetime import datetime, timedelta

import pandas as pd
from github import Github
from dotenv import load_dotenv
from tqdm import tqdm
import github.Auth

load_dotenv()  # loads .env file

# --- CONFIG ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DATA_PATH = "data/papers_list.csv"
OUT_PATH = "data/raw_metadata.json"

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env")

# --- SETUP ---
auth = github.Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)
papers = pd.read_csv(DATA_PATH)
metadata = {}

# --- HELPER FUNCTIONS ---
def safe_get(repo, attr, default=None):
    try:
        return getattr(repo, attr)
    except Exception:
        return default


def extract_repo_features(repo_url):
    repo_name = "/".join(repo_url.split("/")[-2:])  # owner/repo
    repo = g.get_repo(repo_name)

    # recency / last update
    updated = repo.updated_at
    now = datetime.now(updated.tzinfo)
    six_months_ago = now - timedelta(days=180)

    data = {
        "repo_name": repo_name,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
        "last_update": updated.strftime("%Y-%m-%d"),

        # checklist fields (all 8)
        "code_accessible": True,                     # we reached the repo successfully
        "has_license": repo.license is not None,
        "has_readme": False,
        "has_requirements": False,
        "has_tests": False,
        "mentions_dataset": False,
        "example_usage": False,
        "recently_updated": updated > six_months_ago,
    }

    # --- Try to fetch README ---
    try:
        readme = repo.get_readme().decoded_content.decode("utf-8").lower()
        data["has_readme"] = True
        data["mentions_dataset"] = ("dataset" in readme) or ("data/" in readme)
        data["example_usage"] = ("usage" in readme) or ("example" in readme)
    except Exception:
        # keep defaults (False)
        pass

    # --- Check for files at root ---
    try:
        files = [f.path for f in repo.get_contents("")]
        data["has_requirements"] = any("requirements" in f.lower() for f in files)
        data["has_tests"] = any("test" in f.lower() for f in files)
    except Exception:
        # keep defaults
        pass

    return data


# --- MAIN LOOP ---
for _, row in tqdm(papers.iterrows(), total=len(papers)):
    url = row["github_url"]
    pid = row["paper_id"]
    try:
        features = extract_repo_features(url)
        metadata[pid] = features
    except Exception as e:
        # if we even fail to reach the repo, treat as not accessible
        metadata[pid] = {
            "error": str(e),
            "code_accessible": False
        }

# --- SAVE RESULTS ---
with open(OUT_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to {OUT_PATH}")
