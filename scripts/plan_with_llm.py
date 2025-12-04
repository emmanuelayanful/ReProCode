# scripts/plan_with_llm.py
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .llm_local import generate_completion  # used with: python -m scripts.plan_with_llm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = BASE_DIR / "repos"

PAPERS_CSV = DATA_DIR / "papers_list.csv"
PLANS_DIR = DATA_DIR / "plans"
PLANS_DIR.mkdir(parents=True, exist_ok=True)

# Where to dump raw LLM outputs for debugging
RAW_LLM_DIR = DATA_DIR / "llm_raw_outputs"
RAW_LLM_DIR.mkdir(parents=True, exist_ok=True)

# Aggressive truncation
MAX_README_CHARS = 15000
MAX_TREE_CHARS = 6000


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


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[TRUNCATED]..."


def get_readme_text(repo_root: Path) -> str:
    for name in ["README.md", "README.rst", "README.txt", "Readme.md"]:
        p = repo_root / name
        if p.exists():
            return p.read_text(errors="ignore")
    return ""


def get_file_tree(repo_root: Path, max_files: int = 100, max_depth: int = 3) -> str:
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


def build_prompt(paper_id: str, repo_name: str, readme: str, file_tree: str) -> str:
    # Keep this fairly compact to save tokens
    return f"""You are a reproducibility assistant.

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
        - run the main experiment.
        4. Use micromamba/conda and python, do not use sudo or destructive commands.

        {PLAN_SCHEMA}

        Output:
        Return ONLY a single JSON object.
        - It must start with the character '{{' on the first line.
        - It must end with the matching '}}' on the last line.
        - Do not include any explanation, comments, markdown, text, or code fences.
        Only raw JSON.
        """


def call_llm_to_plan(paper_id: str, repo_name: str, readme: str, file_tree: str) -> dict:
    prompt = build_prompt(paper_id, repo_name, readme, file_tree)
    raw = generate_completion(prompt)

    # Save raw LLM output for debugging
    with (RAW_LLM_DIR / f"{paper_id}.txt").open("w") as f:
        f.write(raw)

    # Try to extract JSON from output
    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1 or last <= first:
        print(
            f"[WARN] Model did not return JSON-like text for {paper_id}. "
            f"Raw output saved to {RAW_LLM_DIR / (paper_id + '.txt')}"
        )
        # Fallback: dummy plan so pipeline can still run
        return {
            "paper_id": paper_id,
            "repo_name": repo_name,
            "env": {
                "type": "conda",
                "setup_commands": []
            },
            "steps": [
                {
                    "id": "fallback-list-files",
                    "description": "Fallback step because LLM output was invalid; just list files.",
                    "working_dir": ".",
                    "commands": ["ls"],
                    "expected_artifacts": []
                }
            ],
        }

    json_str = raw[first : last + 1]
    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(
            f"[WARN] JSON decode failed for {paper_id}: {e}. "
            f"Raw output saved to {RAW_LLM_DIR / (paper_id + '.txt')}"
        )
        # Same fallback
        return {
            "paper_id": paper_id,
            "repo_name": repo_name,
            "env": {
                "type": "conda",
                "setup_commands": []
            },
            "steps": [
                {
                    "id": "fallback-list-files",
                    "description": "Fallback step because LLM output was invalid; just list files.",
                    "working_dir": ".",
                    "commands": ["ls"],
                    "expected_artifacts": []
                }
            ],
        }

    # Minimal defaults
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

        raw_readme = get_readme_text(repo_root)
        raw_tree = get_file_tree(repo_root)

        readme = truncate_text(raw_readme, MAX_README_CHARS)
        tree = truncate_text(raw_tree, MAX_TREE_CHARS)

        plan = call_llm_to_plan(paper_id, repo_name, readme, tree)
        out_path = PLANS_DIR / f"{paper_id}.json"
        with out_path.open("w") as f:
            json.dump(plan, f, indent=2)

        print(f"[OK] Saved plan for {paper_id} -> {out_path}")


if __name__ == "__main__":
    main()
