import pandas as pd
import requests
from github import Github
import yaml, os, json
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # loads .env file

# --- CONFIG ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DATA_PATH = "data/papers_list.csv"
OUT_PATH = "data/raw_metadata.json"

# --- SETUP ---
g = Github(GITHUB_TOKEN)
papers = pd.read_csv(DATA_PATH)
metadata = {}

# --- HELPER FUNCTIONS ---
def safe_get(repo, attr, default=None):
    try:
        return getattr(repo, attr)
    except Exception:
        return default

def extract_repo_features(repo_url):
    repo_name = "/".join(repo_url.split("/")[-2:])
    repo = g.get_repo(repo_name)
    data = {
        "repo_name": repo_name,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
        "last_update": repo.updated_at.strftime("%Y-%m-%d"),
        "has_license": repo.license is not None,
        "has_readme": False,
        "has_requirements": False,
        "has_tests": False,
        "mentions_dataset": False,
    }

    # --- Try to fetch README ---
    try:
        readme = repo.get_readme().decoded_content.decode("utf-8").lower()
        data["has_readme"] = True
        data["mentions_dataset"] = "dataset" in readme or "data/" in readme
        data["example_usage"] = "usage" in readme or "example" in readme
    except Exception:
        pass

    # --- Check for files ---
    try:
        files = [f.path for f in repo.get_contents("")]
        data["has_requirements"] = any("requirements" in f for f in files)
        data["has_tests"] = any("test" in f.lower() for f in files)
    except Exception:
        pass

    return data

# --- MAIN LOOP ---
for _, row in tqdm(papers.iterrows(), total=len(papers)):
    url = row["github_url"]
    try:
        features = extract_repo_features(url)
        metadata[row["paper_id"]] = features
    except Exception as e:
        metadata[row["paper_id"]] = {"error": str(e)}

# --- SAVE RESULTS ---
with open(OUT_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to {OUT_PATH}")