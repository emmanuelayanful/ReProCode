import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------
# 1. Load metadata + scores
# -----------------------

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
            # Booleans as 0/1
            "has_readme": int(info.get("has_readme", False)),
            "has_requirements": int(info.get("has_requirements", False)),
            "has_tests": int(info.get("has_tests", False)),
            "has_license": int(info.get("has_license", False)),
            "mentions_dataset": int(info.get("mentions_dataset", False)),
            "example_usage": int(info.get("example_usage", False)),
            # later: add recently_updated, code_accessible, etc.
        }
    )

X_meta = pd.DataFrame(rows)

if X_meta.empty:
    raise RuntimeError("No valid metadata rows found to train on.")

# Merge with scores on paper_id
df = X_meta.merge(scores, on="paper_id", how="inner")

if df.empty:
    raise RuntimeError("No overlap between metadata and scores to train on.")

print("Training data:")
print(df.head(), "\n")

# -----------------------
# 2. Split features / target
# -----------------------

feature_cols = [c for c in df.columns if c not in ["paper_id", "score"]]
X = df[feature_cols]
y = df["score"]  # 0 = easier, 1 = harder (your convention)

# With few repos, this is just a sanity split
test_size = 0.3 if len(df) > 5 else 0.4

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}\n")

# -----------------------
# 3. Train model
# -----------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

# -----------------------
# 4. Evaluate model
# -----------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model evaluation:")
print(f"  Mean Absolute Error (MAE): {mae:.3f}")
print(f"  R^2 score:                 {r2:.3f}\n")

print("Feature importances (higher = more important):")
for name, imp in sorted(
    zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {name:18s} {imp:.3f}")
print()

# -----------------------
# 5. Save model to disk
# -----------------------

os.makedirs("models", exist_ok=True)
model_path = "models/rds_regressor.pkl"

joblib.dump(
    {
        "model": model,
        "feature_cols": feature_cols,
    },
    model_path,
)

print(f"Saved trained model to: {model_path}")
