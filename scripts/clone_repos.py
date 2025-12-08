# scripts/clone_repos.py
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]   # ReProCode/
DATA_DIR = BASE_DIR / "data"
PAPERS_CSV = DATA_DIR / "papers_list.csv"
REPOS_DIR = BASE_DIR / "repos"
REPOS_DIR.mkdir(exist_ok=True)


def clone_one(github_url: str):
    github_url = github_url.rstrip("/")
    owner = github_url.split("/")[-2]
    repo = github_url.split("/")[-1]
    dest = REPOS_DIR / f"{owner}__{repo}"

    if dest.exists():
        print(f"[SKIP] {dest} already exists")
        return

    print(f"[CLONE] {github_url} -> {dest}")
    subprocess.run(["git", "clone", github_url, str(dest)], check=False)


def main():
    df = pd.read_csv(PAPERS_CSV)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        clone_one(row["github_url"])


if __name__ == "__main__":
    main()
