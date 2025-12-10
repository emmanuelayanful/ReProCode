# ReProCode
### CS 557 — Artificial Intelligence  
**Authors:**  
**Munib Ahmed**  
**Emmanuel Ayanful**  



## Overview

ReProCode is a project where we measure how difficult it is to reproduce code from AI-for-Code research papers. Many AI-for-Code Generation papers release their code on GitHub, but the repositories are often missing files, have unclear setup instructions, or rely on very specific environments. Hence, it is harder to reproduce the working code. Our goal is to measure how “reproducible” each repo is using a checklist and then train a small regression model to predict that difficulty.

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
3. A regression model that predicts RDS  
4. A small dataset of metadata and scores  


## Project Structure

```
ReProCode/
├── checklist.yaml                      # Weighted checklist for computing scores
├── data
│   ├── llm_raw_outputs                 # Folder containing LLM raw outputs
│   ├── papers_list.csv                 # List of repos we evaluate
│   ├── plans                           # Folder containg LLM json plans
│   ├── raw_metadata.json               # GitHub metadata
│   ├── repro_scores.csv                # RDS (difficulty scores)
│   └── runtime_logs                    # Folder containing runtime logs from the docker implementation
├── environment.yaml                    # Environment dependencies for reproducibility
├── models
│   └── rds_regressor.pkl               # Trained Random Forest Regressor model
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
├── results
│   ├── feature_importances.csv
│   ├── model_metrics.json
│   └── test_predictions.csv
└── scripts
    ├── clone_repos.py
    ├── collect_metadata.py             # GitHub scraper
    ├── compute_score.py                # Score calculator
    ├── docker_plan.json                # Plan sample to be followed by LLM
    ├── Dockerfile.reprocode            # Setup docker environment
    ├── llm_local.py                    # HugginFace LLM run locally
    ├── plan_with_llm.py                # Generate docker plans using thr LLM
    ├── run_all_in_docker.py            # Run all docker plans in the docker environment
    ├── run_plan.py                     # Run a given docker plan in the docker environment
    └── train_model.py                  # Model training script
```



## Reproducibility Checklist (RDS Criteria)

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
python scripts/collect_metadata.py
```

Output = `data/raw_metadata.json`



### 2. Computing the RDS Score

Each repo starts with a difficulty score of 1.0, then subtracts weights for each checklist item that is present.

To run:

```bash
python scripts/compute_score.py
```

Output = `data/repro_scores.csv`



### 3. Machine Learning Model

We train a small regression model (Random Forest Regressor) to predict the RDS based on repository metadata.

To run:

```bash
python scripts/train_model.py
```

This will print:

- Training/test split size  
- MAE and R² score  
- Feature importances  

The trained model is saved to:

```
models/rds_regressor.pkl
```



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/emmanuelayanful/ReProCode.git
cd ReProCode
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .env
source .env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your GitHub token

Create a `.env` file:

```
GITHUB_TOKEN=ghp_yourtokenhere
```

### 5. How to run the Docker part
Build the docker image with
```bash
docker build -f scripts/Dockerfile.reprocode -t reprocode-base .
```
Create a directory called repo to clone all ppaer repositories from the root directory with
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

### 6. Run the web API + frontend
The FastAPI server serves both the JSON API and the bundled frontend (under `/`).

```bash
uvicorn app.main:app --reload
```

Then open http://localhost:8000 in your browser to use the Quick and Advanced scoring UI.


## Example Results

Some example difficulty scores:

| Repo          | RDS  |
| ------------- | ---- |
| CORE-Bench    | 0.00 |
| HumanEval     | 0.10 |
| MBPP          | 0.30 |
| SWE-Bench     | 0.15 |
| StarCoder     | 0.10 |
| CodeT5        | 0.55 |
| CodeRAG-Bench | 0.20 |

Model results on our small dataset:

- Mean Absolute Error: **0.194**  
- R²: **-0.009** (this result is expected because the dataset is very small)

Top predictive features:

- open issues  
- stars  
- forks  
- example usage  


## Purpose of This Project

This project helped us practice:

- API usage  
- Data collection and cleaning  
- Feature engineering  
- Building a small ML model  
- Understanding reproducibility issues  

Again, this is a prototype, not a full benchmark.


## Authors

- **Emmanuel Ayanful** — metadata collection, scoring design  
- **Munib Ahmed** — model training, integration, debugging  

We both contributed equally to research, testing, and writing and plan continuing to extend the project over the Thanksgiving break. 
