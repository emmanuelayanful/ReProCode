# scripts/plan_with_llm.py
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .llm_local import generate_completion  # if using -m scripts.plan_with_llm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = BASE_DIR / "repos"

PAPERS_CSV = DATA_DIR / "papers_list.csv"
PLANS_DIR = DATA_DIR / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)


PLAN_SCHEMA = r"""
You must output a JSON object with this structure:

{
  "paper_id": "string",
  "repo_name": "string",
  "env": {
    "type": "conda",
    "setup_commands": ["..."]
  },
  "steps": [
    {
      "id": "string",
      "description": "string",
      "working_dir": "string",
      "commands": ["..."],
      "expected_artifacts": ["..."]
    }
  ]
}
"""

def truncate_text(text, max_chars=3000):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [TRUNCATED] ..."


def get_readme_text(repo_root: Path) -> str:
    for name in ["README.md", "README.rst", "README.txt", "Readme.md"]:
        p = repo_root / name
        if p.exists():
            return p.read_text(errors="ignore")
    return ""


def get_file_tree(repo_root: Path, max_files=100, max_depth=3) -> str:
    lines = []
    for root, dirs, files in os.walk(repo_root):
        rel_root = Path(root).relative_to(repo_root)
        depth = len(rel_root.parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        for f in files:
            lines.append(str(rel_root / f))
        if len(lines) >= max_files:
            break
    return "\n".join(lines[:max_files])


def build_prompt(paper_id, repo_name, readme, file_tree):
    return f"""
        You are a reproducibility assistant.

        Repo name: {repo_name}
        Paper id: {paper_id}

        Repo file list (relative paths):
        {file_tree}

        README:
        \"\"\"{readme}\"\"\"

        Task:
        1. Identify the main experiment or usage the authors expect users to run.
        2. Infer environment setup commands (prefer micromamba/conda).
        3. Infer the minimal sequence of shell commands to:
        - install dependencies
        - prepare/download data (if clearly documented)
        - run the main experiment
        4. Use micromamba/conda and python, do not use sudo or destructive commands.

        {PLAN_SCHEMA}

        Output:
        Return ONLY a JSON object, no backticks, no markdown, no explanation.
        """


def call_llm_to_plan(paper_id, repo_name, readme, file_tree) -> dict:
    prompt = build_prompt(paper_id, repo_name, readme, file_tree)
    raw = generate_completion(prompt)

    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1:
        raise ValueError(f"Model did not return JSON-like text for {paper_id}")

    json_str = raw[first:last + 1]
    plan = json.loads(json_str)

    plan.setdefault("paper_id", paper_id)
    plan.setdefault("repo_name", repo_name)
    plan.setdefault("env", {"type": "conda", "setup_commands": []})
    plan.setdefault("steps", [])
    return plan


def main():
    df = pd.read_csv(PAPERS_CSV)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        paper_id = str(row["paper_id"])
        github_url = row["github_url"].rstrip("/")
        owner = github_url.split("/")[-2]
        repo = github_url.split("/")[-1]
        repo_name = f"{owner}/{repo}"

        repo_root = REPOS_DIR / f"{owner}__{repo}"
        if not repo_root.exists():
            print(f"[WARN] Repo dir {repo_root} not found, skip {paper_id}")
            continue

        readme = truncate_text(get_readme_text(repo_root))
        tree = truncate_text(get_file_tree(repo_root))


        plan = call_llm_to_plan(paper_id, repo_name, readme, tree)
        out_path = PLANS_DIR / f"{paper_id}.json"
        with out_path.open("w") as f:
            json.dump(plan, f, indent=2)

        print(f"[OK] Saved plan for {paper_id} -> {out_path}")


if __name__ == "__main__":
    main()
