# run_plan.py
import argparse
import json
import subprocess
import time
from pathlib import Path


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
    parser.add_argument("--repo", required=True)
    parser.add_argument("--log_out", default="/workspace/reprocode_log.json")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    repo_root = Path(args.repo).resolve()

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
