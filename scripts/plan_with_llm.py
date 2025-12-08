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
MAX_README_CHARS = 50000
MAX_TREE_CHARS = 10000


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

def clean_raw_output(raw: str) -> str:
    """
    Remove common LLM wrappers (```json, ```), intro phrases,
    markdown formatting, and whitespace around JSON.
    Leaves only the raw JSON-like string.
    """
    if not raw:
        return ""

    text = raw.strip()

    # Remove common markdown code fences
    fences = ["```json", "```JSON", "```", "```python", "```js"]
    for f in fences:
        text = text.replace(f, "")

    # Remove typical explanation prefixes
    prefixes = [
        "Here is the JSON object:",
        "Here is the JSON:",
        "Here is the plan:",
        "JSON:",
        "Json:",
        "json:",
        "The JSON object is:",
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    return text.strip()


def attempt_repair_json(paper_id: str, repo_name: str, broken_json: str) -> dict | None:
    """
    Ask the LLM to fix invalid JSON (syntax only).
    Returns a parsed dict if successful, otherwise None.
    """
    from .llm_local import generate_completion  # local import to avoid cycles

    repair_prompt = (
        "You previously produced this JSON-like text, but it has a small syntax error:\n\n"
        "```json\n"
        + broken_json +
        "\n```\n\n"
        "Your task: output the same JSON object, but with correct JSON syntax.\n"
        "- Do not change any keys or values.\n"
        "- Only fix commas, quotes, brackets, or other syntax issues.\n"
        "- Return ONLY the corrected JSON object, no explanation, no code fences.\n"
    )

    repaired_raw = generate_completion(repair_prompt)

    # Save repaired text for debugging
    repair_path = RAW_LLM_DIR / f"{paper_id}_repaired.txt"
    with repair_path.open("w") as f:
        f.write(repaired_raw)

    cleaned = clean_raw_output(repaired_raw)

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last == -1 or last <= first:
        print(
            f"[WARN] Repair LLM output still not JSON-like for {paper_id}. "
            f"See {repair_path}"
        )
        return None

    json_str = cleaned[first : last + 1]

    try:
        plan = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(
            f"[WARN] Repair JSON decode failed for {paper_id}: {e}. "
            f"See {repair_path}"
        )
        return None

    # Ensure minimal fields
    plan.setdefault("paper_id", paper_id)
    plan.setdefault("repo_name", repo_name)
    plan.setdefault("env", {"type": "conda", "setup_commands": []})
    plan.setdefault("steps", [])
    return plan


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

        # Second LLM call: try to repair the JSON
        repaired_plan = attempt_repair_json(paper_id, repo_name, json_str)
        if repaired_plan is not None:
            print(f"[INFO] Successfully repaired JSON for {paper_id}.")
            return repaired_plan

        # If repair also fails, final fallback
        print(f"[WARN] Falling back to dummy plan for {paper_id} after failed repair.")
        return {
            "paper_id": paper_id,
            "repo_name": repo_name,
            "env": {"type": "conda", "setup_commands": []},
            "steps": [
                {
                    "id": "fallback-json-error",
                    "description": "Fallback step because JSON decoding failed, and repair also failed.",
                    "working_dir": ".",
                    "commands": ["ls"],
                    "expected_artifacts": [],
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
