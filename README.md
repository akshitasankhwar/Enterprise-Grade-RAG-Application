# 🤖 Enterprise Agentic RAG

A production-grade Retrieval Augmented Generation (RAG) system built with **LangGraph**, **Groq**, **Qdrant**, and **FastAPI**. Deployed with a beautiful Streamlit UI.

---

## ✨ Features

### Core System
- 🧠 **LangGraph Agent**: Cyclic reasoning with Planner → Retriever → Responder nodes
- 🔍 **Semantic Search**: Local embeddings (sentence-transformers) + Qdrant vector DB
- 🎯 **Smart Reranking**: FlashRank semantic reranking for better results
- 💬 **Fast LLM**: Groq Llama 3.3 70B for ultra-fast inference
- 📚 **Multi-format Support**: PDF, HTML, DOCX, TXT document parsing

### API & Frontend
- ⚡ **FastAPI Backend**: Production-ready REST API with validation & error handling
- 🎨 **Streamlit UI**: Beautiful, responsive chat interface
- 🔄 **Session Management**: Conversation history tracking
- 📥 **Export Options**: Download chats as JSON or Markdown
- 📊 **Observability**: Query logging, health checks, info endpoints

### Deployment Ready
- 🐳 **Docker Support**: Docker & docker-compose included
- ☁️ **Cloud-Native**: Deploy to AWS, Heroku, Google Cloud
- 🆓 **100% Free**: Uses only free-tier cloud services & local alternatives

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                             │
│                  (Chat Interface)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  (/health, /query, /info, /collections)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  LangGraph   │ │    Groq      │ │   Qdrant     │
│  Agent       │ │    LLM       │ │   Vector DB  │
│              │ │              │ │              │
│ Planner      │ │ Llama 3.3    │ │ Embeddings   │
│ Retriever    │ │ 70B          │ │ + Ranking    │
│ Responder    │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────────┐   ┌──────────────────────┐
│  Sentence-Transformers│   │  Document Processing │
│  (Local Embeddings)   │   │  (Loaders + Chunking)│
└──────────────────────┘   └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Conda (recommended)
- Docker (for Qdrant)
- Groq API key (free at https://console.groq.com/keys)

### Installation

```bash
# 1. Create conda environment
conda create -n rag-env python=3.11
conda activate rag-env

# 2. Clone/Download project & navigate
cd enterprise-rag

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 5. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 6. Ingest documents (one-time setup)
python -m app.ingestion.processor data --wipe
```

### Running the System

**Terminal 1 - API:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 - UI:**
```bash
streamlit run ui/app.py
```

**Terminal 3 - Optional (CLI Client):**
```bash
python app/client.py
```

Then visit:
- 🎨 **UI**: http://localhost:8501
- 📚 **API Docs**: http://localhost:8000/docs
- 🤖 **API Health**: http://localhost:8000/health

---

## 📁 Project Structure

```
enterprise-rag/
├── app/
│   ├── agents/                 # LangGraph agent logic
│   │   ├── state.py            # Agent state definition
│   │   ├── graph.py            # Compiled LangGraph
│   │   └── nodes/              # Agent nodes
│   │       ├── planner.py       # Query classification
│   │       ├── retriever.py     # Document search
│   │       └── responder.py     # Answer generation
│   ├── ingestion/              # Document processing
│   │   ├── loaders/            # File parsers
│   │   │   ├── pdf.py
│   │   │   ├── html.py
│   │   │   ├── office.py
│   │   │   └── text.py
│   │   ├── chunking/           # Text splitting
│   │   └── processor.py        # End-to-end pipeline
│   ├── services/               # Core services
│   │   └── retrieval/
│   │       ├── embedding.py    # Embedding service
│   │       ├── qdrant_service.py # Vector DB
│   │       └── ranking_service.py # Reranking
│   ├── main.py                 # FastAPI app
│   ├── models.py               # Request/Response schemas
│   └── config.py               # Configuration
├── ui/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── true_data/              # Your documents here
│   └── noisy_data/             # Optional noisy data
├── .streamlit/
│   └── config.toml             # Streamlit config
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker setup
├── README.md                   # This file
├── QUICKSTART.md               # API quick start
└── STREAMLIT_SETUP.md          # UI setup guide
```

---

## 📚 Key Components

### 1. **Ingestion Pipeline**
Loads documents → Chunks text → Creates embeddings → Stores in Qdrant

```python
from app.ingestion.processor import DocumentProcessor

processor = DocumentProcessor()
processor.process_directory("data/true_data", wipe=True)
```

### 2. **LangGraph Agent**
Orchestrates query → classification → retrieval → response

```python
from app.agents.graph import run_agent

result = run_agent("What is machine learning?")
print(result.response)  # AI response
print(result.sources)   # Source documents
```

### 3. **FastAPI Backend**
Exposes agent via REST API with validation

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?"}'
```

### 4. **Streamlit UI**
Beautiful chat interface with sources & reasoning

```bash
streamlit run ui/app.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Welcome & documentation |
| `/health` | GET | API health check |
| `/info` | GET | API & collection info |
| `/collections` | GET | List Qdrant collections |
| `/query` | POST | Main query endpoint |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "session_id": "user_123"
  }'
```

### Response

```json
{
  "query": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence...",
  "classification": "TECHNICAL",
  "sources": [
    {
      "text": "ML enables systems to learn from data...",
      "source": "ml_basics.txt",
      "score": 0.95
    }
  ],
  "reasoning": "Classified as TECHNICAL → Retrieved 10 docs → Reranked 3 → Generated response"
}
```

---

## 🧠 How It Works

### Query Flow

```
User Query
    ↓
Planner Node
├─ Classify: CONVERSATIONAL or TECHNICAL?
    ↓
Retriever Node
├─ Embed query
├─ Search Qdrant for similar docs
├─ Return top 10 results
    ↓
Responder Node
├─ Rerank docs using FlashRank
├─ Create context from top 3 docs
├─ Call Groq LLM
├─ Generate response
    ↓
Return to User
├─ Response text
├─ Sources used
├─ Classification
└─ Reasoning
```

### Example Classifications

**CONVERSATIONAL**
- "Hello! How are you?"
- "What's your name?"
- "Tell me a joke"
→ Responds without searching documents

**TECHNICAL**
- "What does the document say about ML?"
- "Summarize the paper"
- "Compare approaches in the docs"
→ Searches documents, reranks, generates response

---

## 🛠️ Configuration

### Environment Variables (.env)

```env
# LLM
GROQ_API_KEY=gsk_your_key_here

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=documents

# Embeddings
EMBEDDINGS_MODEL=all-MiniLM-L6-v2

# Project
PROJECT_ID=local-rag
LOCATION=local
```

### Streamlit Config (.streamlit/config.toml)

```toml
[theme]
primaryColor = "#00D9FF"
backgroundColor = "#0E1117"

[server]
port = 8501
maxUploadSize = 50
```

---

## 📊 Performance

- **Embedding**: ~100ms per document
- **Search**: ~50ms for Qdrant query
- **Reranking**: ~100ms for 10 documents
- **LLM Response**: 1-3s (Groq is fast!)
- **Total End-to-End**: 2-5s per query

---

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Terminal 2: API
uvicorn app.main:app --reload

# Terminal 3: UI
streamlit run ui/app.py
```

### Docker Compose
```bash
docker-compose up -d
uvicorn app.main:app
streamlit run ui/app.py
```

### Cloud Deployment

**AWS EC2:**
```bash
# Create EC2 instance, SSH in, then:
git clone <repo>
cd enterprise-rag
pip install -r requirements.txt
docker-compose up -d
uvicorn app.main:app --host 0.0.0.0
streamlit run ui/app.py --server.address 0.0.0.0
```

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Set secrets (GROQ_API_KEY)
4. Auto-deploys on push

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot connect to Qdrant | `docker run -d -p 6333:6333 qdrant/qdrant` |
| GROQ_API_KEY not found | Add to .env file |
| API port 8000 in use | `lsof -ti:8000 \| xargs kill -9` |
| No documents found | `python -m app.ingestion.processor data` |
| Streamlit won't start | Check Python 3.11+, activate conda env |

---

## 📖 Documentation

- [API Quick Start](./QUICKSTART.md)
- [Streamlit Setup](./STREAMLIT_SETUP.md)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Groq Docs](https://console.groq.com/docs)

---

## 🤝 Contributing

Feel free to fork, modify, and improve! Some ideas:

- Add more document loaders (Excel, CSV, etc.)
- Implement multi-turn conversation memory
- Add authentication
- Create mobile app
- Add vector DB alternatives
- Implement caching layer

---

## 📝 License

MIT License - Feel free to use for personal/commercial projects

---

## 🙋 Support

Having issues?

1. Check troubleshooting section
2. Review logs: `tail -f query_log.txt`
3. Test API: `curl http://localhost:8000/health`
4. Check API docs: `http://localhost:8000/docs`

---

## 🎉 You're All Set!

You now have a **production-ready RAG system** with:

✅ Local embeddings (free, private)  
✅ Vector search (Qdrant)  
✅ Smart reranking (FlashRank)  
✅ Fast LLM (Groq)  
✅ Beautiful UI (Streamlit)  
✅ Robust API (FastAPI)  
✅ Zero cloud costs  
✅ Easy deployment  

**Start querying:** http://localhost:8501

Happy building! 🚀