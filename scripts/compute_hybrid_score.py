# scripts/compute_hybrid_score.py

import yaml
import json
import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "runtime_logs"
RAW_META_FILE = DATA_DIR / "raw_metadata.json"
CHECKLIST_FILE = BASE_DIR / "checklist.yaml"
OUTPUT_FILE = DATA_DIR / "final_scores.csv"

def compute_static_scores():
    """Computes static scores based on checklist criteria."""
    if not CHECKLIST_FILE.exists() or not RAW_META_FILE.exists():
        print("[WARN] Checklist or metadata not found. Defaulting static scores to 1.0")
        return {}

    with open(CHECKLIST_FILE) as f:
        criteria = yaml.safe_load(f).get("criteria", {})

    with open(RAW_META_FILE) as f:
        meta = json.load(f)

    static_scores = {}
    for pid, repo in meta.items():
        if "error" in repo:
            score = 1.0
        else:
            score = 1.0
            for key, weight in criteria.items():
                if repo.get(key, False):
                    score -= weight
        static_scores[pid] = max(0.0, min(1.0, score))
    
    return static_scores

def compute_dynamic_score(paper_id):
    """Computes dynamic score from runtime logs."""
    log_path = LOGS_DIR / f"{paper_id}.json"
    if not log_path.exists():
        print(f"[WARN] No runtime log for {paper_id}. Defaulting dynamic score to 1.0")
        return 1.0

    try:
        with open(log_path) as f:
            log = json.load(f)
    except json.JSONDecodeError:
        print(f"[ERR] Invalid JSON for {paper_id}. Defaulting dynamic score to 1.0")
        return 1.0

    # 1. Environment Penalty (20% weight)
    env_logs = log.get("env_logs", [])
    total_env_cmds = len(env_logs)
    if total_env_cmds > 0:
        failed_env_cmds = sum(1 for entry in env_logs if entry.get("returncode", 0) != 0 and entry.get("stderr", "").strip())
        env_penalty = failed_env_cmds / total_env_cmds
    else:
        env_penalty = 0.0

    # 2. Step Execution Penalty (40% weight)
    step_logs = log.get("step_logs", [])
    total_steps = len(step_logs)
    
    if total_steps > 0:
        failed_steps = sum(1 for step in step_logs 
                           if any(c.get("returncode", 0) != 0 and c.get("stderr", "").strip() for c in step.get("commands", [])))
        step_penalty = failed_steps / total_steps
    else:
        # If no steps were executed but env succeeded (or partially succeeded), 
        # it might mean there were simply no steps defined. 
        # However, usually no steps means nothing happened.
        # Let's assume if 0 steps, full penalty for step portion.
        step_penalty = 1.0

    # 3. Artifact Penalty (40% weight)
    total_artifacts = 0
    missing_artifacts = 0
    
    for step in step_logs:
        artifacts = step.get("artifacts", {})
        total_artifacts += len(artifacts)
        # artifact value is True if found, False if missing
        missing_artifacts += sum(1 for found in artifacts.values() if not found)

    if total_artifacts > 0:
        artifact_penalty = missing_artifacts / total_artifacts
    else:
        artifact_penalty = 0.0 # No artifacts expected, so no penalty

    # Dynamic Raw Score = Weighted sum of penalties
    # Env (20%) + Steps (40%) + Artifacts (40%)
    dynamic_score = (0.2 * env_penalty) + (0.4 * step_penalty) + (0.4 * artifact_penalty)
    
    print(f"DEBUG {paper_id}: Env={env_penalty:.2f} ({failed_env_cmds if 'failed_env_cmds' in locals() else '?'}/{total_env_cmds if 'total_env_cmds' in locals() else '?'}), Steps={step_penalty:.2f} ({failed_steps if 'failed_steps' in locals() else '?'}/{total_steps if 'total_steps' in locals() else '?'}), Artifacts={artifact_penalty:.2f}")
    
    return max(0.0, min(1.0, dynamic_score))

def main():
    print("Computing Static Scores...")
    static_scores = compute_static_scores()
    
    records = []
    print("Computing Dynamic & Hybrid Scores...")
    
    # Process all paper IDs found in static scores (or logs if static is empty)
    all_pids = set(static_scores.keys())
    # Also verify if there are logs for PIDs not in metadata
    if LOGS_DIR.exists():
        log_pids = {p.stem for p in LOGS_DIR.glob("*.json")}
        all_pids.update(log_pids)

    for pid in sorted(all_pids):
        s_score = static_scores.get(pid, 1.0) # Default to 1.0 if missing metadata
        d_score = compute_dynamic_score(pid)
        
        # Hybrid Formula: 0.4 * Static + 0.6 * Dynamic
        final_score = (0.4 * s_score) + (0.6 * d_score)
        
        records.append({
            "paper_id": pid,
            "static_score": round(s_score, 2),
            "dynamic_score": round(d_score, 2),
            "final_score": round(final_score, 2)
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nFinal Scores Summary:")
    print(df)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
