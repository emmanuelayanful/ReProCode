# run_plan.py
import argparse
import json
import subprocess
import time
from pathlib import Path
import os


def run_command(cmd, cwd, timeout=3600, env=None):
    start = time.time()
    print(f"[RUN] {cmd} (cwd={cwd})", flush=True)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        end = time.time()
        return {
            "command": cmd,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_sec": end - start,
        }
    except subprocess.TimeoutExpired as e:
        end = time.time()
        return {
            "command": cmd,
            "cwd": str(cwd),
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": f"TIMEOUT: {e}",
            "duration_sec": end - start,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo_source", required=True, help="Read-only source of the repo")
    parser.add_argument("--repo_dest", required=True, help="Writable destination for the repo")
    parser.add_argument("--log_out", default="/workspace/reprocode_log.json")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    repo_source = Path(args.repo_source).resolve()
    repo_dest = Path(args.repo_dest).resolve()

    # 0) Copy files
    print(f"[SETUP] Copying {repo_source} -> {repo_dest}...", flush=True)
    if repo_dest.exists():
        import shutil
        shutil.rmtree(repo_dest)
    
    # We use cp -r equivalent to ensure we get a fresh copy
    # shutil.copytree is the standard way
    import shutil
    try:
        shutil.copytree(repo_source, repo_dest, symlinks=True)
        print("[SETUP] Copy complete.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to copy repo: {e}", flush=True)
        # We might want to abort or log this failure
        # For now, let's just proceed but it will likely fail if dir is missing

    # Set root to dest
    repo_root = repo_dest

    with plan_path.open() as f:
        plan = json.load(f)

    logs = {
        "paper_id": plan.get("paper_id"),
        "repo_name": plan.get("repo_name"),
        "env_logs": [],
        "step_logs": [],
    }

    # 1) Environment setup
    env_cfg = plan.get("env", {})
    setup_cmds = env_cfg.get("setup_commands", [])
    for cmd in setup_cmds:
        logs["env_logs"].append(
            run_command(cmd, cwd=repo_root, timeout=1800)
        )

    # 2) Steps
    for step in plan.get("steps", []):
        step_id = step.get("id")
        step_desc = step.get("description")
        rel_workdir = step.get("working_dir", ".") or "."

        # Build candidate working dir and fall back if it doesn't exist
        candidate_dir = (repo_root / rel_workdir).resolve()
        if not candidate_dir.exists():
            print(
                f"[WARN] working_dir '{rel_workdir}' does not exist for step '{step_id}'. "
                f"Falling back to repo root '{repo_root}'.",
                flush=True,
            )
            candidate_dir = repo_root

        step_cmd_logs = []
        for cmd in step.get("commands", []):
            if not cmd:
                continue
            entry = run_command(cmd, cwd=candidate_dir, timeout=3600)
            step_cmd_logs.append(entry)

        # Check artifacts existence (still relative to repo root for now)
        artifacts = {}
        for relpath in step.get("expected_artifacts", []):
            p = repo_root / relpath
            artifacts[relpath] = p.exists()

        logs["step_logs"].append(
            {
                "id": step_id,
                "description": step_desc,
                "commands": step_cmd_logs,
                "artifacts": artifacts,
            }
        )

    # Write consolidated log
    log_path = Path(args.log_out)
    with log_path.open("w") as f:
        json.dump(logs, f, indent=2)


if __name__ == "__main__":
    main()
