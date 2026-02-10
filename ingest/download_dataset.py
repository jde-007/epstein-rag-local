from datasets import load_dataset
from tqdm import tqdm
import json
import os

os.makedirs("data", exist_ok=True)

print("Downloading dataset...")

dataset = load_dataset(
    "teyler/epstein-files-20k",
    split="train"
)

print("Total records:", len(dataset))

docs = []
for row in tqdm(dataset):
    docs.append({
        "text": row["text"],
        "file": row.get("file_name", "unknown")
    })

with open("data/raw.json", "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("✅ Dataset downloaded and saved to data/raw.json")
