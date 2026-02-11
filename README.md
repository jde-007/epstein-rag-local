# Epstein Files RAG (Local Ollama Fork)

A RAG pipeline for querying the 'Epstein Files 20K' dataset, modified to run **completely locally** using Ollama instead of cloud APIs.

**Original repo:** [AnkitNayak-eth/EpsteinFiles-RAG](https://github.com/AnkitNayak-eth/EpsteinFiles-RAG)

## Changes from Original

- Replaced Groq (cloud LLM) with **Ollama** (local)
- No API keys required
- All queries stay on your machine
- Configurable model via environment variables

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- ~4GB disk space for embeddings
- ~8GB RAM recommended

## Quick Start

### 1. Install Ollama and pull a model

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen2.5:7b
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/jde-007/epstein-rag-local.git
cd epstein-rag-local
pip install -r requirements.txt
```

### 3. Configure (optional)

```bash
cp .env.example .env
# Edit .env to change model or Ollama URL
```

Default config:
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=qwen2.5:7b`

### 4. Run the pipeline

```bash
# Download dataset (~2M lines)
python ingest/download_dataset.py

# Clean and reconstruct documents
python ingest/clean_dataset.py

# Chunk into semantic pieces
python ingest/chunk_dataset.py

# Embed into ChromaDB (takes a while)
python ingest/embed_chunks.py
```

### 5. Start the API and UI

```bash
# Terminal 1: API server
uvicorn api.main:app --reload

# Terminal 2: Streamlit UI
streamlit run app.py
```

- API: http://127.0.0.1:8000
- UI: http://127.0.0.1:8501

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model to use for generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |

Recommended generation models:
- `qwen2.5:7b` - Good balance of speed and quality
- `llama3.2:3b` - Faster, lighter
- `qwen2.5:32b` - Better quality, slower

## Architecture

```
Raw Dataset (HuggingFace)
        ↓
Cleaning & Reconstruction
        ↓
Semantic Chunking
        ↓
Vector Embeddings (Ollama GPU)
        ↓
ChromaDB Vector Store
        ↓
Retriever (MMR)
        ↓
Ollama LLM (local GPU)
        ↓
Grounded Answer
```

Embedding models (speed vs quality):
- `nomic-embed-text` - Fast, good quality (recommended)
- `mxbai-embed-large` - Better quality, slower
- `bge-m3` - Best quality, slowest

## API Endpoints

- `GET /health` - Check status and config
- `POST /ask?question=...` - Query the documents

## Data Source

Dataset: [teyler/epstein-files-20k](https://huggingface.co/datasets/teyler/epstein-files-20k)

⚠️ **Disclaimer:** This dataset contains unverified, potentially incomplete or inaccurate information. It should be used for research and educational purposes only.

## License

MIT (same as original)

## Credits

- Original implementation: [Ankit Kumar Nayak](https://github.com/AnkitNayak-eth)
- Local Ollama adaptation: [jde-007](https://github.com/jde-007)
