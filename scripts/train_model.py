import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------
# 1. Configuration & Setup
# -----------------------
RESULTS_DIR = "results"
MODELS_DIR = "models"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# 2. Load metadata + scores
# -----------------------
print("Loading data...")
with open("data/raw_metadata.json") as f:
    meta = json.load(f)

scores = pd.read_csv("data/repro_scores.csv")  # columns: paper_id, score

rows = []
for pid, info in meta.items():
    # Skip repos where we couldn't fetch metadata
    if "error" in info:
        continue

    rows.append(
        {
            "paper_id": pid,
            "stars": info.get("stars", 0),
            "forks": info.get("forks", 0),
            "open_issues": info.get("open_issues", 0),
            # Booleans converted to 0/1
            "has_readme": int(info.get("has_readme", False)),
            "has_requirements": int(info.get("has_requirements", False)),
            "has_tests": int(info.get("has_tests", False)),
            "has_license": int(info.get("has_license", False)),
            "mentions_dataset": int(info.get("mentions_dataset", False)),
            "example_usage": int(info.get("example_usage", False)),
            # Add other features here if you expanded collect_metadata.py
            "recently_updated": int(info.get("recently_updated", False)),
        }
    )

X_meta = pd.DataFrame(rows)

if X_meta.empty:
    raise RuntimeError("No valid metadata rows found to train on.")

# Merge with scores on paper_id
df = X_meta.merge(scores, on="paper_id", how="inner")

if df.empty:
    raise RuntimeError("No overlap between metadata and scores to train on.")

print(f"Total combined records: {len(df)}")
print(df.head(3), "\n")

# -----------------------
# 3. Split features / target
# -----------------------

feature_cols = [c for c in df.columns if c not in ["paper_id", "score"]]

# NOTE: We split the whole dataframe first so we can keep 'paper_id' 
# aligned with the test set for our results CSV later.
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

X_train = train_df[feature_cols]
y_train = train_df["score"]

X_test = test_df[feature_cols]
y_test = test_df["score"]

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")

# -----------------------
# 4. Train model
# -----------------------

print("Training RandomForestRegressor...")
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=10 # Optional: limit depth to prevent overfitting on small data
)

model.fit(X_train, y_train)

# -----------------------
# 5. Evaluate model
# -----------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "mae": round(mae, 4),
    "r2_score": round(r2, 4),
    "n_train": len(X_train),
    "n_test": len(X_test)
}

print("Model evaluation:")
print(f"  Mean Absolute Error (MAE): {mae:.3f}")
print(f"  R^2 score:                 {r2:.3f}\n")

# Print and collect feature importances
print("Feature importances (higher = more important):")
importances = []
for name, imp in sorted(
    zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {name:18s} {imp:.3f}")
    importances.append({"feature": name, "importance": imp})

# -----------------------
# 6. Save Results & Model
# -----------------------

# A. Save the trained model
model_path = os.path.join(MODELS_DIR, "rds_regressor.pkl")
joblib.dump(
    {
        "model": model,
        "feature_cols": feature_cols,
    },
    model_path,
)
print(f"\n[Saved] Model -> {model_path}")

# B. Save Metrics to JSON
metrics_path = os.path.join(RESULTS_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"[Saved] Metrics -> {metrics_path}")

# C. Save Feature Importances to CSV
fi_path = os.path.join(RESULTS_DIR, "feature_importances.csv")
pd.DataFrame(importances).to_csv(fi_path, index=False)
print(f"[Saved] Feature Importances -> {fi_path}")

# D. Save Predictions (Test Set) for analysis
# We create a new dataframe with ID, Actual, Predicted, and Error
results_df = test_df.copy()
results_df["predicted_score"] = y_pred
results_df["error"] = results_df["predicted_score"] - results_df["score"]
results_df = results_df[["paper_id", "score", "predicted_score", "error"] + feature_cols]

preds_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
results_df.to_csv(preds_path, index=False)
print(f"[Saved] Predictions -> {preds_path}")