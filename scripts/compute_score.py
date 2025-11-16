import yaml, json, pandas as pd

with open("checklist.yaml") as f:
    checklist = yaml.safe_load(f)["criteria"]

with open("data/raw_metadata.json") as f:
    meta = json.load(f)

records = []
for pid, repo in meta.items():
    if "error" in repo:
        score = 0
    else:
        score = 0
        for key, weight in checklist.items():
            if repo.get(key, False):
                score += weight
    records.append({"paper_id": pid, "score": round(score, 2)})

df = pd.DataFrame(records)
df.to_csv("data/repro_scores.csv", index=False)
print(df)
