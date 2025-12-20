# ReProCode
### CS 557 — Artificial Intelligence  
**Authors:**  
**Munib Ahmed**  
**Emmanuel Ayanful**  



## Overview

ReProCode is a project where we measure how difficult it is to reproduce code from AI-for-Code research papers. Many AI-for-Code Generation papers release their code on GitHub, but the repositories are often missing files, have unclear setup instructions, or rely on very specific environments. Hence, it is harder to reproduce the working code. Our goal is to measure how “reproducible” each repo is using a checklist and dynamic execution logs.

We selected well-known code-generation repositories such as:

- HumanEval  
- MBPP  
- SWE-Bench  
- StarCoder  
- CodeT5  
- CodeRAG-Bench  
- CORE-Bench  

This project includes:
1. A GitHub metadata scraper  
2. A reproducibility scoring system (RDS)  
3. A small dataset of metadata and scores  


## Project Structure

ReProCode/
├── checklist.yaml                      # Weighted checklist for computing scores
├── data
│   ├── llm_raw_outputs                 # Folder containing LLM raw outputs
│   ├── papers_list.csv                 # List of repos we evaluate
│   ├── plans                           # Folder containing LLM json plans
│   ├── raw_metadata.json               # GitHub metadata
│   ├── repro_scores.csv                # RDS (difficulty scores)
│   └── runtime_logs                    # Folder containing runtime logs from the docker implementation
├── environment.yaml                    # Environment dependencies for reproducibility
├── README.md
├── repos                               # Cloned GitHub repos corresponding to papers in papers_list.csv
│   ├── bigcode-project__starcoder
│   ├── code-rag-bench__code-rag-bench
│   ├── google-research__google-research
│   ├── openai__human-eval
│   ├── salesforce__CodeT5
│   ├── siegelz__core-bench
│   └── swe-bench__swe-bench
├── requirements.txt
├── results                             # Results from scoring
│   ├── feature_importances.csv
│   ├── model_metrics.json
│   └── test_predictions.csv
└── scripts
    ├── clone_repos.py
    ├── collect_metadata.py             # GitHub scraper
    ├── compute_hybrid_score.py         # Score calculator using runtime logs and checklist
    ├── compute_score.py                # Score calculator using checklist
    ├── docker_plan.json                # Plan sample to be followed by LLM
    ├── Dockerfile.reprocode            # Setup docker environment
    ├── llm_local.py                    # HuggingFace LLM run locally
    ├── plan_with_llm.py                # Generate docker plans using the LLM
    ├── run_all_in_docker.py            # Run all docker plans in the docker environment
    └── run_plan.py                     # Run a given docker plan in the docker environment


## Reproducibility Static Score (Checklist)

We calculate the Reproducibility Difficulty Score (RDS) based on these weighted features:

```yaml
criteria:
  code_accessible: 0.1
  has_readme: 0.15
  has_requirements: 0.15
  has_tests: 0.1
  mentions_dataset: 0.15
  has_license: 0.1
  recently_updated: 0.1
  example_usage: 0.15
```

- **0.0 = easy to reproduce**  
- **1.0 = very difficult to reproduce**

Repositories lose points as they satisfy more checklist items which means the difficulty decreases to reproduce the code.




## How the Project Works

### 1. Metadata Collection

We extract GitHub metadata such as:

- Stars, forks, open issues  
- Has README  
- Has requirements file  
- Has tests  
- Has license  
- Example usage in the README  
- Mentions dataset  
- Whether the repo was recently updated  

To run:

```bash
python -m scripts.collect_metadata
```

Output = `data/raw_metadata.json`



### 2. Computing the RDS Score

Each repo starts with a difficulty score of 1.0, then subtracts weights for each checklist item that is present.

To run:

```bash
python scripts/compute_score.py
```

Output = `data/repro_scores.csv`




### 3. Dynamic Scoring (Docker + LLM Plans)

We use an LLM to read the README and produce a step-by-step plan (create env, install deps, run commands, check outputs). Then we execute the plan inside a controlled Docker image (`Dockerfile.reprocode`) and log everything.

Logs are saved under `data/runtime_logs/{paper_id}.json`.

The **Dynamic Score** is calculated from three components (failure fraction penalties):

#### 1. Environment Penalty (20%)
Based on `env_logs` (environment commands: conda/pip/install/etc.).
- `failed_env_cmds`: Commands with nonzero return code and non-empty stderr.
- `env_penalty` = `failed_env_cmds / total_env_cmds` (0 = everything OK, 1 = everything failed).

#### 2. Step Execution Penalty (40%)
Based on logical steps in `step_logs` (e.g., “train model”, “run evaluation”).
A step counts as failed if **any** command in it fails.
- `failed_steps`: Number of steps with ≥1 failed command.
- `step_penalty` = `failed_steps / total_steps`
- If no steps, `step_penalty = 1.0` (nothing ran).

#### 3. Artifact Penalty (40%)
For each step, we check for expected artifacts (files/logs/models).
- `missing_artifacts`: Number of artifacts marked False.
- `artifact_penalty` = `missing_artifacts / total_artifacts`
- If no artifacts defined, `artifact_penalty = 0.0`.

#### Formula
```python
dynamic_score = 0.2 * env_penalty + 0.4 * step_penalty + 0.4 * artifact_penalty
```
The result is clipped to [0, 1] (0 = everything ran & produced artifacts, 1 = everything failed).


### 4. Hybrid Score

Hybrid score combines static + dynamic scores.

For each repo `paper_id`:
- `s_score` = static_score (checklist)
- `d_score` = dynamic_score (Docker run)

**Final Score Formula**:
```python
final_score = 0.4 * s_score + 0.6 * d_score
```

**Intuition**:
- **Static (40%)** — “Repo looks reproducible on paper.”
- **Dynamic (60%)** — “Repo actually behaves reproducibly in Docker.”

**Interpretation**:
- **0.0** → Very easy: good documentation + successful execution + expected artifacts.
- **1.0** → Very hard: poor documentation and/or commands fail, missing outputs.



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/emmanuelayanful/ReProCode.git
cd ReProCode
```

### 2. Create and activate a virtual environment and install dependencies

```bash
conda create --file environment.yaml --name reprocode
conda activate reprocode
```

### 3. Add your GitHub token and HuggingFace model name

Create a `.env` file:
```
GITHUB_TOKEN=ghp_yourtokenhere
HF_MODEL_NAME=yourmodelhere
```

### 4. How to run the Docker part
Build the docker image with
```bash
docker build -f scripts/Dockerfile.reprocode -t reprocode-base .
```
Create a directory called repo to clone all paper repositories from the root directory with
```bash
mkdir repo
python -m scripts.clone_repos
```

Generate execution plans using an LLM with
```bash
python -m scripts.plan_with_llm
```

Execute all the plans in the docker with
```bash
python -m scripts.run_all_in_docker
```

Compute Scores with the runtime logs with
```bash
python -m scripts.compute_hybrid_score
```

### 5. Run the web API + frontend
The FastAPI server serves both the JSON API and the bundled frontend (under `/`).

```bash
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in your browser to use the Quick and Advanced scoring UI.


## Example Results

Some example difficulty scores:

| paper_id | static_score | dynamic_score | final_score |
| :--- | :--- | :--- | :--- |
| P01 | 0.0 | 0.55 | 0.33 |
| P02 | 0.1 | 0.67 | 0.44 |
| P03 | 0.3 | 0.6 | 0.48 |
| P04 | 0.15 | 0.43 | 0.32 |
| P05 | 0.1 | 0.49 | 0.33 |
| P06 | 0.55 | 1.0 | 0.82 |
| P07 | 0.2 | 0.75 | 0.53 |
| P08 | 0.3 | 0.9 | 0.66 |
| P09 | 0.45 | 0.87 | 0.7 |
| P10 | 0.3 | 0.0 | 0.12 |
| P11 | 0.45 | 0.53 | 0.5 |
| P12 | 0.25 | 0.9 | 0.64 |
| P13 | 0.15 | 0.72 | 0.49 |
| P14 | 0.15 | 0.67 | 0.46 |
| P15 | 0.3 | 0.87 | 0.64 |
| P16 | 0.2 | 0.38 | 0.31 |

## Purpose of This Project

This project helped us practice:

- API usage  
- Data collection and cleaning  
- Feature engineering  
- Understanding Docker
- Understanding LLMs
- Understanding GitHub
- Understanding reproducibility issues  

Again, this is a prototype, not a full benchmark.


## Authors

- **Emmanuel Ayanful** — scoring design, docker setup, implementation and running  
- **Munib Ahmed** — metadata collection, integration, debugging  

We both contributed equally to research, testing, and writing and plan continuing to extend the project over the Thanksgiving break. 
