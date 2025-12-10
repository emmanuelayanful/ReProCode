"""Benchmark several regressors for predicting reproducibility difficulty (RDS).

Usage:
    python scripts/benchmark_models.py
Outputs:
    results/model_comparison.csv  # cross-val metrics (lower MAE is better)
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RAW_META_PATH = DATA_DIR / "raw_metadata.json"
SCORES_PATH = DATA_DIR / "repro_scores.csv"


def load_dataset():
    with RAW_META_PATH.open() as f:
        meta = json.load(f)
    scores = pd.read_csv(SCORES_PATH)

    rows = []
    for pid, info in meta.items():
        if "error" in info:
            continue
        rows.append(
            {
                "paper_id": pid,
                "stars": info.get("stars", 0),
                "forks": info.get("forks", 0),
                "open_issues": info.get("open_issues", 0),
                "has_readme": int(info.get("has_readme", False)),
                "has_requirements": int(info.get("has_requirements", False)),
                "has_tests": int(info.get("has_tests", False)),
                "has_license": int(info.get("has_license", False)),
                "mentions_dataset": int(info.get("mentions_dataset", False)),
                "example_usage": int(info.get("example_usage", False)),
                "recently_updated": int(info.get("recently_updated", False)),
            }
        )

    X = pd.DataFrame(rows)
    if X.empty:
        raise RuntimeError("No valid metadata rows found to benchmark.")

    df = X.merge(scores, on="paper_id", how="inner")
    if df.empty:
        raise RuntimeError("No overlap between metadata and scores to benchmark.")

    feature_cols = [c for c in df.columns if c not in ["paper_id", "score"]]
    return df, feature_cols


def main():
    print("Loading data...")
    df, feature_cols = load_dataset()
    X = df[feature_cols]
    y = df["score"]

    # Fill missing numerics
    X = X.fillna(0)

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "hist_gb": HistGradientBoostingRegressor(random_state=42),
        "gbrt": GradientBoostingRegressor(random_state=42),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            random_state=42,
        ),
    }

    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
    }

    n_samples = len(df)
    cv_folds = min(5, n_samples) if n_samples > 1 else 1

    records = []
    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_res = cross_validate(
            model,
            X,
            y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=None,
        )

        r2_vals = pd.Series(cv_res["test_r2"]).dropna()
        r2_mean = round(r2_vals.mean(), 4) if not r2_vals.empty else None
        r2_std = round(r2_vals.std(), 4) if not r2_vals.empty else None

        records.append(
            {
                "model": name,
                "mae_mean": round(-cv_res["test_mae"].mean(), 4),
                "mae_std": round(cv_res["test_mae"].std(), 4),
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "n_folds": len(cv_res["test_mae"]),
            }
        )

    out_df = pd.DataFrame(records).sort_values(by="mae_mean")
    out_path = RESULTS_DIR / "model_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print("\nModel comparison (lower MAE is better):")
    print(out_df.to_string(index=False))
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
