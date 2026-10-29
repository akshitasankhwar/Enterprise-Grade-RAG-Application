---
title: Enterprise RAG
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Upload documents and get cited answers
---

# 🤖 Enterprise Agentic RAG

A production-grade Retrieval Augmented Generation (RAG) system built with **LangGraph**, **Groq**, **Qdrant**, and **FastAPI**, with a Streamlit UI. Users upload their own documents, ask questions, and get answers that cite the passages they came from.

---

## ✨ Features

### Core System
- 📤 **Upload-driven knowledge base**: Users add PDF, DOCX, TXT, MD, CSV and HTML files themselves, from the UI or the API
- 🧠 **LangGraph Agent**: Planner → Retriever → Responder nodes
- 🔍 **Semantic Search**: Local embeddings (sentence-transformers) + Qdrant vector DB
- 🎯 **Smart Reranking**: FlashRank reranking picks the best passages for each answer
- 💬 **Fast LLM**: Groq inference (model configurable via `GROQ_MODEL`)

### API & Frontend
- ⚡ **FastAPI Backend**: REST API for upload, document management and queries
- 🎨 **Streamlit UI**: Document library in the sidebar, chat with cited sources
- 🗂️ **Document Management**: List documents, re-upload to replace, remove from the library
- 📥 **Export Options**: Download chats as JSON or Markdown
- 📊 **Observability**: Query logging, health checks, info endpoints

### Deployment Ready
- 🐳 **Docker Support**: Qdrant via docker-compose
- 🆓 **Free to run**: Local embeddings and reranking, free-tier Groq API

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
│          (Document library + chat with citations)           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│   /upload  /documents  /query  /health  /info               │
└───────────┬─────────────────────────────────┬───────────────┘
            │ upload                          │ query
            ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────────┐
│  Document Processing │         │     LangGraph Agent      │
│  Load → Chunk →      │         │  Planner → Retriever →   │
│  Embed → Store       │         │  Responder (Groq LLM)    │
└──────────┬───────────┘         └────────────┬─────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │  Qdrant Vector DB                   │
        │  + sentence-transformers embeddings │
        │  + FlashRank reranking              │
        └─────────────────────────────────────┘
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

# 2. Go to the project folder
cd Enterprise-Grade-RAG-Application-main

# 3. Install dependencies (requirements.txt alone if you won't run the tests)
pip install -r requirements-dev.txt

# 4. Create .env from the template, then set GROQ_API_KEY in it
copy .env.example .env        # Windows (macOS/Linux: cp .env.example .env)

# 5. Start Qdrant
docker compose up -d
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

Then visit:
- 🎨 **UI**: http://localhost:8501
- 📚 **API Docs**: http://localhost:8000/docs
- 🤖 **API Health**: http://localhost:8000/health

If port 8000 is taken, run the API on another port and point the UI at it:

```bash
uvicorn app.main:app --reload --port 8001
set API_BASE_URL=http://localhost:8001 && streamlit run ui/app.py
```

**VS Code:** open this folder and use Run and Debug → **API + UI** (API on port 8001).

### Using It

1. Drop files into **Add documents** in the sidebar and click **Add to library**
2. Ask questions in the chat box
3. Match the `[1]` `[2]` citations in the answer to the numbered passages under **Sources**

---

## 📁 Project Structure

```
Enterprise-Grade-RAG-Application-main/
├── app/                        # FastAPI backend
│   ├── main.py                 # API routes and startup
│   ├── config.py               # Settings from .env
│   ├── models.py               # Request/Response schemas
│   ├── agents/                 # LangGraph agent
│   │   ├── graph.py            # Planner → Retriever → Responder
│   │   ├── state.py            # State passed between nodes
│   │   ├── llm.py              # Groq chat model
│   │   └── nodes/
│   │       ├── planner.py      # Document question or chat?
│   │       ├── retriever.py    # Search Qdrant
│   │       └── responder.py    # Rerank + cited answer
│   ├── ingestion/              # Processing for uploaded files
│   │   ├── processor.py        # Load → Chunk → Embed → Store
│   │   ├── splitter.py         # Text chunking
│   │   └── loaders/            # File parsers (pdf, html, office, text)
│   └── services/
│       ├── embedding_service.py  # sentence-transformers embeddings
│       ├── qdrant_service.py     # Vector DB
│       └── ranking_service.py    # FlashRank reranking
├── ui/
│   └── app.py                  # Streamlit UI
├── tests/
│   └── test_api.py             # API tests (pytest)
├── docs/
│   └── streamlit-ui.md         # UI guide
├── scripts/
│   └── start.sh                # Starts API and UI inside the container
├── data/uploads/               # Uploaded files (git-ignored)
├── logs/                       # query_log.txt (git-ignored, created at runtime)
├── .streamlit/config.toml      # Streamlit theme & server config
├── .env.example                # Environment variable template
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # + pytest
├── pyproject.toml              # pytest configuration
├── docker-compose.yml          # Qdrant
├── Dockerfile                  # API + UI image (Hugging Face Spaces)
└── README.md                   # This file (header = Space settings)
```

---

## 📚 Key Components

### 1. **Upload Pipeline**
Each uploaded file is saved to `data/uploads/`, then loaded → chunked → embedded → stored in Qdrant, with the file name as its source. Uploading a file with the same name again replaces its old passages; removing a document deletes its passages and the stored file.

### 2. **LangGraph Agent**
Orchestrates query → classification → retrieval → response

```python
from app.agents.graph import run_agent

result = run_agent("What does the contract say about renewal?")
print(result["response"])       # AI response
print(result["reranked_docs"])  # Passages used
```

### 3. **FastAPI Backend**
Exposes upload, document management and the agent over REST (see below).

### 4. **Streamlit UI**
Document library, chat with cited sources, export. See [docs/streamlit-ui.md](./docs/streamlit-ui.md).

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/upload` | POST | Upload files into the knowledge base |
| `/documents` | GET | List documents and their passage counts |
| `/documents/{filename}` | DELETE | Remove a document |
| `/query` | POST | Ask a question |
| `/health` | GET | API health check |
| `/info` | GET | API & collection info |
| `/` | GET | Welcome & endpoint list |

### Upload Documents

```bash
curl -X POST http://localhost:8000/upload -F "files=@report.pdf" -F "files=@notes.docx"
```

```json
{
  "files": [
    {"filename": "report.pdf", "chunks": 42},
    {"filename": "notes.docx", "chunks": 7}
  ],
  "skipped": [],
  "total_chunks": 49
}
```

### List and Remove Documents

```bash
curl http://localhost:8000/documents
curl -X DELETE http://localhost:8000/documents/notes.docx
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the payment terms?", "session_id": "user_123"}'
```

```json
{
  "query": "What are the payment terms?",
  "response": "Invoices are payable within 30 days of the invoice date [1].",
  "classification": "TECHNICAL",
  "sources": [
    {
      "text": "Payment is due within thirty (30) days of the invoice date...",
      "source": "report.pdf",
      "score": 0.97,
      "page": 4
    }
  ],
  "reasoning": "Classified as TECHNICAL → Retrieved 6 passages (similarity ≥ 0.15) → Reranked 3 docs → Generated cited response"
}
```

`[1]` in the response refers to `sources[0]`, `[2]` to `sources[1]`, and so on. `page` is the zero-based page number for PDFs; for other formats it is the section index within the file.

If nothing in the library answers the question, the response is a fixed "I couldn't find that in your documents..." message with an empty `sources` list.

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
├─ Search Qdrant for similar passages
├─ Keep up to 10 with similarity ≥ MIN_RELEVANCE
    ↓
Responder Node
├─ None kept → reply "couldn't find that in your documents"
├─ Rerank passages using FlashRank, keep top 3
├─ Number them [1] [2] [3] as context
├─ Call Groq LLM (temperature 0): answer only from the passages,
│  cite every claim, or reply NOT_FOUND
├─ NOT_FOUND → reply "couldn't find that in your documents"
    ↓
Return to User
├─ Response text with [n] citations
├─ Sources used, numbered to match
├─ Classification
└─ Reasoning
```

### Grounding

Answers to document questions come only from your library:

- **Relevance floor**: passages less similar to the question than `MIN_RELEVANCE` (cosine, default 0.15) never reach the LLM. Tuned on sample documents: real questions, including broad ones like "Summarize these documents", scored at least 0.18; clearly unrelated questions scored as low as 0.03.
- **Passage-only answers at temperature 0**, with every claim cited as `[n]`. The UI shows sources with the same numbers and greys out passages the answer didn't cite.
- **No general-knowledge fallback**: when the passages don't contain the answer, the model replies `NOT_FOUND` and the user sees "I couldn't find that in your documents" with no sources.

The floor is deliberately low: some off-topic questions still clear it when a document shares vocabulary (e.g. "parental leave" vs. an expense policy). For those, the passage-only prompt and `NOT_FOUND` reply are what keep the answer grounded.

### Example Classifications

**CONVERSATIONAL**
- "Hello! How are you?"
- "What's your name?"
→ Responds without searching documents

**TECHNICAL**
- "What does the policy say about remote work?"
- "Summarize the quarterly report"
→ Searches documents, reranks, generates a cited response (or says it isn't in the documents)

---

## 🛠️ Configuration

### Environment Variables (.env)

See [.env.example](./.env.example) for the full template.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | - (required) | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model ([available models](https://console.groq.com/docs/models)) |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant server |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `QDRANT_URL` / `QDRANT_API_KEY` | unset | Qdrant Cloud cluster URL and key; overrides host/port when set |
| `EMBEDDINGS_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (384-dim) |
| `MIN_RELEVANCE` | `0.15` | Minimum cosine similarity for a passage to reach the LLM (raise it to be stricter) |

The UI reads `API_BASE_URL` (default `http://localhost:8000`) to find the API, and `APP_PASSWORD` (unset = no sign-in) to require a password before the app opens.

### Streamlit Config (.streamlit/config.toml)

Theme colors and fonts, plus server settings such as `maxUploadSize = 50` (MB per file).

---

## 📊 Performance

Answers typically take a few seconds end to end. The first upload or question after the API starts also loads the embedding and reranking models, so it takes longer.

---

## 🧪 Tests

With Qdrant running and `GROQ_API_KEY` set:

```bash
pytest -v
```

---

## 🚀 Deployment

### Free hosting: Hugging Face Spaces + Qdrant Cloud

The `Dockerfile` runs the API and the UI in one container. The API listens only inside the container; the UI is served on port 7860.

**1. Create a free Qdrant Cloud cluster**
- Sign up at https://cloud.qdrant.io and create a free cluster
- Copy its endpoint URL (like `https://<id>.<region>.cloud.qdrant.io:6333`) and create an API key for it
- Free clusters pause after 1 week without use and are deleted after 4 weeks; reactivate them in the Qdrant console

**2. Create the Space**
- Go to https://huggingface.co/new-space, choose **Docker** → **Blank**, hardware **CPU basic** (free)
- Choose **Private** visibility so only you can open it (your documents and Groq quota stay yours)

**3. Add secrets** in the Space's **Settings → Variables and secrets**:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | Your Groq key |
| `QDRANT_URL` | Cluster endpoint URL from step 1 |
| `QDRANT_API_KEY` | Cluster API key from step 1 |
| `APP_PASSWORD` | Optional: a password the app asks for before opening (use it if the Space is public) |

**4. Push the code** from this folder. `.gitignore` keeps `.env` and `data/uploads/` out of the repository:

```bash
git init -b main
git add .
git commit -m "Deploy to Hugging Face Spaces"
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push --force space main
```

When git asks for a password, use a Hugging Face access token with **write** permission (https://huggingface.co/settings/tokens). `--force` replaces the Space's starter README; later updates are a normal `git push space main`.

**5. Wait for the build** (about 10 minutes the first time) in the Space's **Logs** tab, then open the **App** tab and add documents.

Notes:
- Uploaded files are kept on the Space's temporary disk and disappear when it restarts; their passages stay in Qdrant Cloud, so answers keep working
- A free Space sleeps after 48 hours without visitors; the first visit afterwards takes about a minute
- `scripts/start.sh` turns off Streamlit's XSRF check because Spaces show the app inside an iframe, where it blocks uploads. Keep the Space private or set `APP_PASSWORD`

**Run the same container locally:**
```bash
docker build -t enterprise-rag-space .
docker run -p 7860:7860 --env-file .env -e QDRANT_HOST=host.docker.internal enterprise-rag-space
```
Then visit http://localhost:7860.

### Docker Compose (Qdrant) + App
```bash
docker compose up -d
uvicorn app.main:app --host 0.0.0.0
streamlit run ui/app.py --server.address 0.0.0.0
```

**AWS EC2:**
```bash
# Create EC2 instance, SSH in, then:
git clone <repo>
cd Enterprise-Grade-RAG-Application-main
pip install -r requirements.txt
docker compose up -d
uvicorn app.main:app --host 0.0.0.0
streamlit run ui/app.py --server.address 0.0.0.0
```

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| `docker compose up` says "no configuration file provided" | Run it from the project folder (where `docker-compose.yml` is) |
| Cannot connect to Qdrant | Start Docker Desktop, then `docker compose up -d` |
| GROQ_API_KEY not found | Copy `.env.example` to `.env` and set the key |
| `model_not_found` from Groq | Set `GROQ_MODEL` in `.env` to a model from the Groq models list |
| API port 8000 in use | Run on another port and set `API_BASE_URL` for the UI (see Quick Start) |
| "No text found" after upload | The file has no extractable text (e.g. a scanned PDF); run OCR first |
| Streamlit won't start | Check Python 3.11+, activate conda env |

---

## 📖 Documentation

- [Streamlit UI Guide](./docs/streamlit-ui.md)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Groq Docs](https://console.groq.com/docs)

---

## 🤝 Contributing

Feel free to fork, modify, and improve! Some ideas:

- Add more document loaders (Excel, PowerPoint, etc.)
- Implement multi-turn conversation memory
- Add authentication and per-user libraries
- Add OCR for scanned PDFs

---

## 📝 License

MIT License - Feel free to use for personal/commercial projects

---

## 🙋 Support

Having issues?

1. Check troubleshooting section
2. Review logs: `query_log.txt` and the API terminal output
3. Test API: `curl http://localhost:8000/health`
4. Check API docs: `http://localhost:8000/docs`
