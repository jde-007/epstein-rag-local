import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --------------------------------------------------
# Reset Chroma DB
# --------------------------------------------------
if Path(CHROMA_DIR).exists():
    print("🧹 Removing existing Chroma DB...")
    shutil.rmtree(CHROMA_DIR)

# --------------------------------------------------
# Load chunks
# --------------------------------------------------
print("📂 Loading chunks...")

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

total_chunks = len(chunks)
print(f"✅ Loaded {total_chunks} chunks")

# --------------------------------------------------
# Load embedding model (Ollama - GPU accelerated)
# --------------------------------------------------
print(f"🧠 Loading Ollama embedding model: {EMBED_MODEL}")
print(f"   Server: {OLLAMA_BASE_URL}")

embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=EMBED_MODEL
)

print("✅ Embedding model ready")

# --------------------------------------------------
# Create Chroma DB
# --------------------------------------------------
print("📦 Initializing Chroma vector store...")

db = Chroma(
    collection_name="epstein",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# --------------------------------------------------
# Embed in batches
# --------------------------------------------------
BATCH = 500  # Smaller batches for network calls
print(f"🚀 Starting embedding in batches of {BATCH}")

for i in range(0, total_chunks, BATCH):
    end = min(i + BATCH, total_chunks)
    pct = (end / total_chunks) * 100
    print(f"🔹 Embedding chunks {i + 1} → {end} / {total_chunks} ({pct:.1f}%)")

    db.add_texts(
        texts=[c["text"] for c in chunks[i:end]],
        metadatas=[c["metadata"] for c in chunks[i:end]]
    )

print("🎉 Chroma embedding complete")
print("📁 Vector DB saved at:", CHROMA_DIR)
