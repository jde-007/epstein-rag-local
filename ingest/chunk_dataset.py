import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("data/cleaned.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

def h(t): 
    return hashlib.sha256(t.lower().encode()).hexdigest()

seen = set()
chunks = []

for d in docs:
    parts = splitter.split_text(d["text"])
    for i, p in enumerate(parts):
        key = h(p)
        if key in seen:
            continue
        seen.add(key)
        chunks.append({
            "text": p,
            "metadata": {
                "source": d["file"],
                "chunk": i
            }
        })

with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Chunks:", len(chunks))
