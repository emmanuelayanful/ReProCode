# scripts/run_all_in_docker.py
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = BASE_DIR / "repos"
PLANS_DIR = DATA_DIR / "plans"
PAPERS_CSV = DATA_DIR / "papers_list.csv"

RUNTIME_LOGS_DIR = DATA_DIR / "runtime_logs"
RUNTIME_LOGS_DIR.mkdir(exist_ok=True)

IMAGE_NAME = "reprocode-base"


def main():
    df = pd.read_csv(PAPERS_CSV)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper_id = str(row["paper_id"])
        github_url = row["github_url"].rstrip("/")
        owner = github_url.split("/")[-2]
        repo = github_url.split("/")[-1]
        repo_name = f"{owner}/{repo}"

        print(f"[INFO] Running Docker for {paper_id} ({repo_name})")

        repo_dir = REPOS_DIR / f"{owner}__{repo}"
        if not repo_dir.exists():
            print(f"[WARN] Repo dir {repo_dir} does not exist, skipping {paper_id}")
            continue

        plan_path = PLANS_DIR / f"{paper_id}.json"
        if not plan_path.exists():
            print(f"[WARN] Plan file {plan_path} does not exist, skipping {paper_id}")
            continue

        host_log_path = RUNTIME_LOGS_DIR / f"{paper_id}.json"

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            # mount repo as /mnt/repo_source inside container (read-only)
            "-v",
            f"{repo_dir}:/mnt/repo_source:ro",
            # mount plan as /workspace/plan.json (read-only)
            "-v",
            f"{plan_path}:/workspace/plan.json:ro",
            # mount runtime logs dir so container can write logs visible on host
            "-v",
            f"{RUNTIME_LOGS_DIR}:/workspace/logs",
            # Resource limits
            "--memory=4g",
            "--cpus=2",
            IMAGE_NAME,
            "python",
            "/workspace/run_plan.py",
            "--plan",
            "/workspace/plan.json",
            "--repo_source",
            "/mnt/repo_source",
            "--repo_dest",
            "/workspace/repo",
            "--log_out",
            f"/workspace/logs/{paper_id}.json",
        ]

        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[WARN] Docker run failed for {paper_id} with code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        if host_log_path.exists():
            print(f"[OK] Log produced for {paper_id} -> {host_log_path}")
        else:
            print(f"[WARN] No log produced for {paper_id}")


if __name__ == "__main__":
    main()

