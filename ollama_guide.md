# Ollama Setup Guide

Use **free local models** instead of OpenAI! Fast, private, and no API costs.

## 🚀 Quick Setup

### **1. Install Ollama**

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai/download

### **2. Pull Model**
```bash
# Pull qwen2.5:0.5b (small, fast, 0.5B parameters)
ollama pull qwen2.5:0.5b

# Or try other models:
# ollama pull qwen2.5:1.5b    # Better quality
# ollama pull llama2           # 7B parameters
# ollama pull mistral          # 7B parameters
```

### **3. Start Ollama Server**
```bash
ollama serve
```

### **4. Start RAG API**
```bash
# In another terminal
python -m uvicorn src.api.app:app --reload
```

### **5. Query with Ollama**
```bash
curl -X POST "http://localhost:8000/query/ollama" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is photosynthesis?",
    "mode": "ultra",
    "top_k": 5
  }'
```

---

## 📊 Comparison: OpenAI vs Ollama

| Feature | OpenAI | Ollama |
|---------|--------|--------|
| **Cost** | $0.15-$2.50 per 1M tokens | FREE |
| **Speed** | 1-2s | 0.5-1s (local) |
| **Quality** | Excellent (GPT-4) | Good (depends on model) |
| **Privacy** | Data sent to OpenAI | 100% local |
| **Internet** | Required | Not required |
| **Setup** | API key needed | Download model |

---

## 🎯 Model Recommendations

### **For Speed (qwen2.5:0.5b)** ⚡
```bash
ollama pull qwen2.5:0.5b
```
- Size: ~350 MB
- Speed: Very fast (0.3-0.5s)
- Quality: Good for simple queries
- Best for: Quick answers, testing

### **For Balance (qwen2.5:1.5b)**
```bash
ollama pull qwen2.5:1.5b
```
- Size: ~900 MB
- Speed: Fast (0.5-1s)
- Quality: Better reasoning
- Best for: Most use cases

### **For Quality (llama2 or mistral)**
```bash
ollama pull mistral
# or
ollama pull llama2
```
- Size: ~4 GB
- Speed: Slower (1-2s)
- Quality: Excellent
- Best for: Complex queries

---

## 🔧 Configuration

Edit `src/generation/ollama_generator.py`:

```python
# Change model
_OLLAMA_GENERATOR = OllamaGenerator(
    model="qwen2.5:1.5b",  # Change this
    base_url="http://localhost:11434"
)

# Change Ollama URL (if running elsewhere)
_OLLAMA_GENERATOR = OllamaGenerator(
    model="qwen2.5:0.5b",
    base_url="http://192.168.1.100:11434"  # Remote Ollama
)
```

Or use environment variables:
```bash
export OLLAMA_MODEL="qwen2.5:1.5b"
export OLLAMA_URL="http://localhost:11434"
```

---

## 📡 API Endpoints

### **POST /query** (OpenAI)
Uses OpenAI GPT models (requires API key).

### **POST /query/ollama** (Ollama)
Uses local Ollama models (free, no API key).

**Same request format:**
```json
{
  "query": "What is quantum physics?",
  "mode": "ultra",
  "top_k": 5,
  "use_lexical": false
}
```

**Same response format:**
```json
{
  "answer": "...",
  "sources": [...],
  "confidence": "high",
  "retrieved": [...],
  "metadata": {
    "generator": "ollama",  // ← Shows which generator
    "model": "qwen2.5:0.5b"
  }
}
```

---

## 🧪 Testing

```python
# test_ollama.py
import requests

# Test Ollama endpoint
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
print(f"Model: {result['metadata']['model']}")
```

---

## 🔍 Troubleshooting

### **Error: "Cannot connect to Ollama"**
```bash
# Make sure Ollama is running
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

### **Error: "Model not found"**
```bash
# Pull the model first
ollama pull qwen2.5:0.5b

# List available models
ollama list
```

### **Slow responses**
```bash
# Use smaller model
ollama pull qwen2.5:0.5b

# Or increase timeout in ollama_generator.py
timeout=60  # Increase from 30
```

### **Out of memory**
```bash
# Use smaller model
ollama pull qwen2.5:0.5b  # 350 MB

# Or reduce context in ollama_generator.py
text[:400]  # Reduce from 800
```

---

## 🎨 Custom Models

Want to use a different Ollama model?

```bash
# Try other models
ollama pull gemma:2b
ollama pull phi
ollama pull neural-chat

# Update in code
_OLLAMA_GENERATOR = OllamaGenerator(model="gemma:2b")
```

See all models: https://ollama.ai/library

---

## 💡 Pro Tips

### **Use Ollama for Development**
- Free
- Fast iteration
- No API limits

### **Use OpenAI for Production**
- Better quality
- More consistent
- Handles complex queries

### **Hybrid Approach**
```python
# Development
POST /query/ollama

# Production
POST /query
```

### **Cost Savings**
```
OpenAI: ~$0.50 per 1,000 queries
Ollama: $0 (free!)

1M queries saved = $500 💰
```

---

## 📈 Performance

### **Query Times**

| Step | Time |
|------|------|
| Retrieval | 0.2-0.3s |
| Filtering | 0.05s |
| Ollama (qwen2.5:0.5b) | 0.3-0.5s |
| **Total** | **0.5-0.8s** ⚡ |

Compare to:
- OpenAI: 0.5-1.2s (network latency)
- Total with OpenAI: 0.8-1.5s

**Ollama is often FASTER!**

---

## 🔐 Privacy

### **With Ollama:**
✅ All data stays on your machine
✅ No internet required
✅ No data sent to third parties

### **With OpenAI:**
⚠️ Data sent to OpenAI servers
⚠️ Internet required
⚠️ Subject to OpenAI's privacy policy

---

## 📚 Learn More

- Ollama: https://ollama.ai
- Ollama Models: https://ollama.ai/library
- Qwen2.5: https://github.com/QwenLM/Qwen2.5

---

**Enjoy free, fast, and private RAG! 🚀**