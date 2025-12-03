# scripts/run_all_in_docker.py
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = BASE_DIR / "repos"

PAPERS_CSV = DATA_DIR / "papers_list.csv"
PLANS_DIR = DATA_DIR / "plans"
LOGS_DIR = DATA_DIR / "runtime_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE = "reprocode-base"


def run_one(paper_id: str, github_url: str) -> bool:
    github_url = github_url.rstrip("/")
    owner = github_url.split("/")[-2]
    repo = github_url.split("/")[-1]
    repo_name = f"{owner}/{repo}"

    repo_root = REPOS_DIR / f"{owner}__{repo}"
    plan_path = PLANS_DIR / f"{paper_id}.json"

    if not repo_root.exists():
        print(f"[WARN] Missing repo dir {repo_root}, skip {paper_id}")
        return False
    if not plan_path.exists():
        print(f"[WARN] Missing plan {plan_path}, skip {paper_id}")
        return False

    container_log_path = "/workspace/reprocode_log.json"
    host_log_path = LOGS_DIR / f"{paper_id}.json"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{repo_root}:/workspace/repo",
        "-v", f"{plan_path}:/workspace/plan.json",
        IMAGE,
        "python", "/workspace/run_plan.py",
        "--plan", "/workspace/plan.json",
        "--repo", "/workspace/repo",
        "--log_out", container_log_path,
    ]

    print(f"[INFO] Running Docker for {paper_id} ({repo_name})")
    subprocess.run(cmd, check=False)

    inner_log = repo_root / "reprocode_log.json"
    if inner_log.exists():
        inner_log.replace(host_log_path)
        print(f"[OK] Saved runtime log -> {host_log_path}")
        return True
    else:
        print(f"[WARN] No log produced for {paper_id}")
        return False


def main():
    df = pd.read_csv(PAPERS_CSV)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper_id = str(row["paper_id"])
        github_url = row["github_url"]
        run_one(paper_id, github_url)


if __name__ == "__main__":
    main()
