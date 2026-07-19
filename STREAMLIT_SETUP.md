"""
Streamlit UI Setup & Running Guide

The Streamlit UI is a beautiful frontend for your RAG system.
"""

# ============================================================================
# INSTALLATION
# ============================================================================

"""
1. Make sure you have streamlit installed (included in requirements.txt)
   pip install streamlit

2. The UI file should be at: ui/app.py
"""

# ============================================================================
# RUNNING THE UI
# ============================================================================

"""
Start the Streamlit server:

  streamlit run ui/app.py

This will open in your browser at:
  http://localhost:8501

Output will look like:
  You can now view your Streamlit app in your browser.
  
  Local URL: http://localhost:8501
"""

# ============================================================================
# FEATURES
# ============================================================================

"""
✨ Features Included:

1. Chat Interface
   - Send questions and get responses
   - See conversation history
   - Clear chat with "New Chat" button

2. Real-time Classification
   - Shows if query is CONVERSATIONAL or TECHNICAL
   - Helps understand how the agent classified your query

3. Source Documents
   - Expandable "Sources" section
   - Shows which documents were used
   - Displays relevance scores (0-1)
   - Snippet of document text

4. Agent Reasoning
   - Expandable "Reasoning" section
   - Shows the agent's thought process
   - Explains: Classification → Retrieval → Response

5. Session Management
   - Unique session ID for each conversation
   - Conversation history persists during session
   - "New Chat" button to start fresh

6. Export Options
   - Download chat as JSON (structured data)
   - Download chat as Markdown (readable format)
   - Useful for documentation & archiving

7. Sidebar Controls
   - API health status
   - Collection statistics
   - Theme selection
   - Toggle reasoning/sources display
   - Auto-scroll toggle

8. Dark/Light Theme
   - Beautiful dark theme (default)
   - Optional light theme
   - Fully customizable via streamlit_config.toml
"""

# ============================================================================
# COMPLETE STARTUP (All Services)
# ============================================================================

"""
Run these commands in separate terminals:

Terminal 1 - Qdrant (Vector Database):
  docker run -d -p 6333:6333 qdrant/qdrant

Terminal 2 - FastAPI Backend:
  uvicorn app.main:app --reload --port 8000

Terminal 3 - Streamlit UI:
  streamlit run ui/app.py

Terminal 4 (Optional) - Monitor API:
  tail -f query_log.txt

Then visit:
  - UI: http://localhost:8501
  - API Docs: http://localhost:8000/docs
  - Qdrant Dashboard: http://localhost:6333/dashboard
"""

# ============================================================================
# FOLDER STRUCTURE
# ============================================================================

"""
enterprise-rag/
├── ui/
│   ├── app.py                    ← Streamlit main app (THIS IS NEW)
│   └── __init__.py
├── app/
│   ├── main.py
│   ├── models.py
│   ├── config.py
│   ├── agents/
│   ├── ingestion/
│   └── services/
├── data/
│   ├── true_data/
│   └── noisy_data/
├── .streamlit/
│   └── config.toml              ← Streamlit config (NEW)
├── .env
├── requirements.txt
└── docker-compose.yml
"""

# ============================================================================
# CUSTOMIZATION
# ============================================================================

"""
Customize the UI by editing:

1. Colors & Theme
   → Edit .streamlit/config.toml
   → Change [theme] section colors

2. API Endpoint
   → In ui/app.py, line with: API_BASE_URL = "http://localhost:8000"
   → Change to your API URL (for production)

3. Timeout
   → In ui/app.py: API_TIMEOUT = 30
   → Increase if queries take longer

4. Page Title & Icon
   → In ui/app.py: st.set_page_config()
   → Change page_title and page_icon

5. Sidebar Content
   → Edit the "SIDEBAR" section in ui/app.py
   → Add your own info, links, settings
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Problem: "Cannot connect to API"
Solution:
  1. Make sure FastAPI is running: uvicorn app.main:app --reload
  2. Check API is on port 8000: curl http://localhost:8000/health
  3. Check API_BASE_URL in ui/app.py is correct
  4. Check firewall isn't blocking port 8000

Problem: "API request timed out"
Solution:
  1. Your query might be taking too long
  2. Increase API_TIMEOUT in ui/app.py
  3. Check FastAPI logs for errors
  4. Try a simpler query first

Problem: "Qdrant connection error"
Solution:
  1. Make sure Qdrant is running
  2. Check: docker ps | grep qdrant
  3. Make sure no documents exist → Ingest: python -m app.ingestion.processor data
  4. Restart Qdrant: docker restart <container_id>

Problem: "ModuleNotFoundError: streamlit"
Solution:
  1. Activate conda environment: conda activate rag-env
  2. Install streamlit: pip install streamlit
  3. Run from project root: streamlit run ui/app.py

Problem: "Port 8501 already in use"
Solution:
  1. Another Streamlit app is running
  2. Kill it: lsof -ti:8501 | xargs kill -9
  3. Or use different port: streamlit run ui/app.py --server.port 8502
"""

# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
Optimize for faster responses:

1. Reduce query timeout if API is fast
   → API_TIMEOUT = 15

2. Cache API responses (Streamlit caches by default)
   → Streamlit automatically caches function results

3. Use smaller chunk sizes (faster retrieval)
   → Edit in app/ingestion/chunking/splitter.py

4. Limit sources displayed
   → Edit response display section in ui/app.py

5. Pre-warm the model
   → Send a test query before opening UI
"""

# ============================================================================
# DEPLOYMENT
# ============================================================================

"""
To deploy the UI (Production):

1. AWS EC2:
   streamlit run ui/app.py --server.address 0.0.0.0

2. Heroku:
   Create Procfile: web: streamlit run ui/app.py

3. Docker:
   Create Dockerfile to run Streamlit in container

4. Streamlit Cloud:
   Connect GitHub repo to Streamlit Cloud
   → Auto-deploys on push
   → Free tier available

5. Production Settings:
   → Disable toolbar: [client] toolbarMode = "viewer"
   → Hide source: [client] showErrorDetails = false
   → Add authentication (custom middleware)
"""

if __name__ == "__main__":
    print(__doc__)
