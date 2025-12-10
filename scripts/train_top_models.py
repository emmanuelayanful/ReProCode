"""Train and evaluate the top three models for RDS prediction.

Models: ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
Outputs:
  - models/{name}.pkl (joblib dump with model + feature_cols)
  - results/top_models_metrics.csv (per-model metrics)
  - results/top_models_metrics.json (per-model metrics)
Usage:
  python scripts/train_top_models.py
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

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
        raise RuntimeError("No valid metadata rows found to train on.")

    df = X.merge(scores, on="paper_id", how="inner")
    if df.empty:
        raise RuntimeError("No overlap between metadata and scores to train on.")

    feature_cols = [c for c in df.columns if c not in ["paper_id", "score"]]
    return df, feature_cols


def train_and_eval(model_name: str, model, X_train, y_train, X_test, y_test, feature_cols):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model artifact
    model_path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)

    return {
        "model": model_name,
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "model_path": str(model_path),
    }


def main():
    print("Loading data...")
    df, feature_cols = load_dataset()
    X = df[feature_cols].fillna(0)
    y = df["score"]

    # Simple hold-out split; CV is handled separately in benchmark script
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    X_train, y_train = train_df[feature_cols], train_df["score"]
    X_test, y_test = test_df[feature_cols], test_df["score"]

    models = {
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
        "hist_gb": HistGradientBoostingRegressor(random_state=42),
    }

    records = []
    for name, model in models.items():
        print(f"Training {name}...")
        rec = train_and_eval(name, model, X_train, y_train, X_test, y_test, feature_cols)
        records.append(rec)

    results_df = pd.DataFrame(records).sort_values(by="mae")

    csv_path = RESULTS_DIR / "top_models_metrics.csv"
    json_path = RESULTS_DIR / "top_models_metrics.json"

    results_df.to_csv(csv_path, index=False)
    with json_path.open("w") as f:
        json.dump(records, f, indent=2)

    print("\nModel metrics (lower MAE is better):")
    print(results_df.to_string(index=False))
    print(f"\n[Saved] {csv_path}\n[Saved] {json_path}")


if __name__ == "__main__":
    main()
