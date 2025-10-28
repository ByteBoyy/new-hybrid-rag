# Wikipedia RAG System 🚀

A fast, efficient Retrieval-Augmented Generation (RAG) system built over 7 million Wikipedia articles. Search semantically and get AI-generated answers with source citations.

## ✨ Features

- **🔍 Semantic Search**: FAISS-powered vector search over 7M Wikipedia articles
- **💡 Dual Mode Support**: 
  - Standard (768D, full quality)
  - Ultra (192D, 50% smaller, 97%+ quality)
- **🤖 Multiple AI Backends**:
  - OpenAI (GPT-4 Mini)
  - Ollama (Free local models)
- **⚡ Fast**: Sub-second query times
- **📊 Smart Filtering**: Lightweight relevance scoring
- **🎯 Optional Keyword Boost**: BM25 for keyword-heavy queries

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd hybrid-rag

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create .env file
touch .env

# Add your OpenAI API key (optional - only if using OpenAI)
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### 3. Start the API

```bash
# Start FastAPI server
python -m uvicorn src.api.app:app --reload

# Or with custom host/port
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: **http://localhost:8000**

Swagger docs: **http://localhost:8000/docs**

---

## 🦙 Using Ollama (Free Local Models)

### Setup Ollama

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# 2. Pull a model (choose one)
ollama pull qwen2.5:0.5b    # Fast & small (350 MB)
ollama pull qwen2.5:1.5b    # Better quality (900 MB)
ollama pull llama2          # Best quality (4 GB)

# 3. Start Ollama server
ollama serve
```

### Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| qwen2.5:0.5b | 350 MB | ⚡⚡⚡ | ⭐⭐ | Quick testing |
| qwen2.5:1.5b | 900 MB | ⚡⚡ | ⭐⭐⭐ | Most use cases |
| llama2 | 4 GB | ⚡ | ⭐⭐⭐⭐ | Complex queries |

---

## 📡 API Usage

### Query with OpenAI

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is photosynthesis?",
    "mode": "ultra",
    "top_k": 5,
    "use_lexical": false
  }'
```

### Query with Ollama (Free!)

```bash
curl -X POST "http://localhost:8000/query/ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is photosynthesis?",
    "mode": "ultra",
    "top_k": 5,
    "use_lexical": false
  }'
```

### Python Example

```python
import requests

# Query with Ollama
response = requests.post(
    "http://localhost:8000/query/ollama",
    json={
        "query": "What causes rain?",
        "mode": "ultra",
        "top_k": 5
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])}")
```

---

## 🎛️ API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *required* | Your search query |
| `mode` | string | `"ultra"` | `"standard"` or `"ultra"` |
| `top_k` | int | `5` | Number of results (1-20) |
| `use_lexical` | bool | `false` | Add BM25 keyword boost |

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Default mode
DEFAULT_MODE = "ultra"  # or "standard"

# Retrieval settings
TOP_K = 10              # Retrieve more, filter to best 5
FINAL_K = 5             # Final results to return

# Semantic vs Lexical weights
SEMANTIC_WEIGHT = 1.0   # Primary method
LEXICAL_WEIGHT = 0.3    # Keyword boost weight

# Ultra mode settings
ULTRA_NPROBE = 20       # Higher = better quality, slower (10-50)

# Generation settings
OPENAI_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
```

---

## 📊 Performance

### Query Times

| Step | Time |
|------|------|
| Semantic Retrieval | 0.2-0.3s |
| Relevance Filter | 0.05s |
| Ollama Generation | 0.3-0.5s |
| **Total** | **0.5-0.8s** ⚡ |

### Cost Comparison

| Backend | Cost per 1,000 queries | Speed |
|---------|------------------------|-------|
| Ollama | **$0** (Free!) | 0.5-0.8s |
| OpenAI | ~$0.50 | 0.8-1.5s |

**💰 Save $500 per 1M queries with Ollama!**

---

## 🛠️ Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to docs |
| `/query` | POST | Query with OpenAI |
| `/query/ollama` | POST | Query with Ollama (free) |
| `/health` | GET | Health check |
| `/config` | GET | Current configuration |
| `/stats` | GET | Detailed statistics |

---

## 🏗️ Project Structure

```
hybrid-rag/
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI application
│   ├── retrieval/
│   │   ├── semantic_retriever.py   # FAISS search
│   │   ├── lexical_retriever.py    # BM25 search
│   │   └── hybrid_retriever.py     # Combined retrieval
│   ├── evaluation/
│   │   └── relevance_filter.py     # Relevance scoring
│   ├── generation/
│   │   ├── answer_generator.py     # OpenAI generation
│   │   └── ollama_answer_generator.py  # Ollama generation
│   ├── config.py                   # Configuration
│   └── models.py                   # Pydantic models
├── requirements.txt
├── .env
└── README.md
```

---

## 🔍 Troubleshooting

### Ollama Connection Error

```bash
# Make sure Ollama is running
ollama serve

# Check status
curl http://localhost:11434/api/tags
```

### Model Not Found

```bash
# Pull the model first
ollama pull qwen2.5:0.5b

# List installed models
ollama list
```

### Index Not Found

Make sure index files are in the correct location:
- Ultra mode: Check `ULTRA_DIR` in `src/config.py`
- Standard mode: Check `DATA_DIR` in `src/config.py`

### OpenAI API Error

```bash
# Check your API key in .env
cat .env

# Make sure OPENAI_API_KEY is set
export OPENAI_API_KEY="your-key-here"
```

---

## 📚 Learn More

- **Ollama**: https://ollama.ai
- **FAISS**: https://github.com/facebookresearch/faiss
- **BGE Embeddings**: https://huggingface.co/BAAI/bge-base-en
- **FastAPI**: https://fastapi.tiangolo.com

---

## 🎯 Use Cases

- 📖 **Question Answering**: Get accurate answers from Wikipedia
- 🔍 **Research**: Find relevant articles quickly
- 💡 **Education**: Learn about any topic with sources
- 🤖 **Chatbots**: Build knowledge-grounded assistants
- 📊 **Analysis**: Extract information at scale

---

## 🚀 What's Next?

1. **Try both backends**: Compare OpenAI vs Ollama quality
2. **Experiment with modes**: Test standard vs ultra performance
3. **Tune parameters**: Adjust `top_k`, `use_lexical`, etc.
4. **Try different models**: Test various Ollama models
5. **Build applications**: Integrate into your projects
