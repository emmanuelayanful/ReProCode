"""FastAPI service for quick and advanced reproducibility scoring.
- Quick score: collect_metadata.py + compute_score.py
- Advanced score: clone -> plan_with_llm -> run_all_in_docker -> compute_hybrid_score.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PAPERS_CSV = DATA_DIR / "papers_list.csv"
REPRO_SCORES = DATA_DIR / "repro_scores.csv"
FINAL_SCORES = DATA_DIR / "final_scores.csv"
FRONTEND_DIR = Path(__file__).parent / "frontend"

DEFAULT_MODEL = "Qwen/Qwen1.5-1.8B-Chat"  # small enough for MPS/CPU
PYTHON = sys.executable

app = FastAPI(title="ReProCode Scoring API", version="0.1.0")

# in-memory job tracker (ephemeral)
_jobs: dict[str, "Job"] = {}


def run_cmd(cmd: list[str], env_extra: Optional[dict[str, str]] = None) -> None:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {' '.join(cmd)}: {e}")


def ensure_paper(github_url: str, title: Optional[str] = None) -> str:
    df = pd.read_csv(PAPERS_CSV)
    if "github_url" in df.columns:
        existing = df[df["github_url"].str.rstrip("/") == github_url.rstrip("/")]
        if not existing.empty:
            return str(existing.iloc[0]["paper_id"])

    # generate next PXX id
    max_num = 0
    for pid in df["paper_id"].astype(str):
        if pid.startswith("P") and pid[1:].isdigit():
            max_num = max(max_num, int(pid[1:]))
    new_pid = f"P{max_num + 1:02d}"
    new_row = {
        "paper_id": new_pid,
        "title": title or new_pid,
        "github_url": github_url.rstrip("/"),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(PAPERS_CSV, index=False)
    return new_pid


def read_static_score(paper_id: str) -> Optional[float]:
    if not REPRO_SCORES.exists():
        return None
    df = pd.read_csv(REPRO_SCORES)
    row = df[df["paper_id"] == paper_id]
    if row.empty:
        return None
    return float(row.iloc[0]["score"])


def read_hybrid_scores(paper_id: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not FINAL_SCORES.exists():
        return None, None, None
    df = pd.read_csv(FINAL_SCORES)
    row = df[df["paper_id"] == paper_id]
    if row.empty:
        return None, None, None
    s = row.iloc[0].get("static_score")
    d = row.iloc[0].get("dynamic_score")
    f = row.iloc[0].get("final_score")
    return (float(s) if pd.notna(s) else None,
            float(d) if pd.notna(d) else None,
            float(f) if pd.notna(f) else None)


@dataclass
class Job:
    paper_id: str
    status: str
    error: Optional[str] = None


def quick_flow(paper_id: str) -> dict:
    run_cmd([PYTHON, "scripts/collect_metadata.py"])
    run_cmd([PYTHON, "scripts/compute_score.py"])
    static = read_static_score(paper_id)
    return {"paper_id": paper_id, "static_score": static}


def advanced_flow(job_id: str, paper_id: str, model_name: str) -> None:
    try:
        env_filter = {"FILTER_PAPER_ID": paper_id}
        run_cmd([PYTHON, "-m", "scripts.clone_repos"], env_extra=env_filter)
        env_plan = {"FILTER_PAPER_ID": paper_id, "HF_MODEL_NAME": model_name}
        run_cmd([PYTHON, "-m", "scripts.plan_with_llm"], env_extra=env_plan)
        run_cmd([PYTHON, "-m", "scripts.run_all_in_docker"], env_extra=env_filter)
        run_cmd([PYTHON, "scripts/compute_hybrid_score.py"])
        s, d, f = read_hybrid_scores(paper_id)
        _jobs[job_id].status = "completed"
        _jobs[job_id].error = None
    except Exception as exc:  # noqa: BLE001
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(exc)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/score/quick")
def score_quick(body: dict):
    github_url = body.get("github_url")
    title = body.get("title")
    if not github_url:
        raise HTTPException(status_code=400, detail="github_url is required")
    paper_id = ensure_paper(github_url, title)
    result = quick_flow(paper_id)
    return result


@app.post("/score/advanced")
def score_advanced(body: dict, background: BackgroundTasks):
    github_url = body.get("github_url")
    title = body.get("title")
    model_name = body.get("model_name", DEFAULT_MODEL)
    if not github_url:
        raise HTTPException(status_code=400, detail="github_url is required")
    paper_id = ensure_paper(github_url, title)
    job_id = uuid.uuid4().hex
    _jobs[job_id] = Job(paper_id=paper_id, status="queued")
    background.add_task(advanced_flow, job_id, paper_id, model_name)
    return {"job_id": job_id, "paper_id": paper_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    payload = {"job_id": job_id, "paper_id": job.paper_id, "status": job.status, "error": job.error}
    if job.status == "completed":
        static_score = read_static_score(job.paper_id)
        s, d, f = read_hybrid_scores(job.paper_id)
        payload.update({"static_score": static_score, "dynamic_score": d, "final_score": f})
    return payload


@app.get("/score/{paper_id}")
def get_score(paper_id: str):
    static = read_static_score(paper_id)
    s, d, f = read_hybrid_scores(paper_id)
    if static is None and f is None:
        raise HTTPException(status_code=404, detail="paper_id not found")
    return {"paper_id": paper_id, "static_score": static, "dynamic_score": d, "final_score": f}


# Serve static frontend if desired
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "ReProCode scoring API"}
