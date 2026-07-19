"""
Enterprise Agentic RAG - Streamlit UI

A beautiful, production-ready chat interface for the RAG system.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, List, Dict
import time

# Page configuration
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main { padding: 0rem 1rem; }
    
    /* Chat messages */
    .chat-message {
        display: flex;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .chat-message.user {
        background-color: #264653;
        justify-content: flex-end;
    }
    
    .chat-message.assistant {
        background-color: #2a9d8f;
    }
    
    /* Sources styling */
    .sources-container {
        background-color: #e76f51;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Reasoning */
    .reasoning-container {
        background-color: #f4a261;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30


def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def get_api_info() -> Optional[Dict]:
    """Get API information"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/info",
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def query_api(query: str, session_id: Optional[str] = None) -> Optional[Dict]:
    """Send query to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "session_id": session_id
            },
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        st.info("Make sure to run: `uvicorn app.main:app --reload`")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🤖 Enterprise RAG")
    
    # API Status
    st.subheader("Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if check_api_health():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # API Info
    st.subheader("Collection Info")
    api_info = get_api_info()
    
    if api_info:
        qdrant_info = api_info.get("qdrant", {})
        st.metric("📚 Documents", qdrant_info.get("point_count", 0))
        st.metric("📐 Vector Size", qdrant_info.get("vector_size", 0))
    else:
        st.info("⚠️ Could not fetch collection info")
    
    # Settings
    st.subheader("Settings")
    
    theme = st.radio(
        "Theme",
        ["Dark", "Light"],
        index=0
    )
    
    show_reasoning = st.checkbox("Show Reasoning", value=True)
    show_sources = st.checkbox("Show Sources", value=True)
    auto_scroll = st.checkbox("Auto Scroll", value=True)
    
    # Session management
    st.subheader("Session")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Export"):
            st.session_state.export_chat = True
    
    # About
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    **Enterprise Agentic RAG**
    
    A production-grade RAG system with:
    - 🧠 LangGraph agent
    - 🔍 Semantic search
    - 🎯 Document reranking
    - 💬 Groq LLM
    
    Built with FastAPI + Streamlit
    """)


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🚀 Enterprise Agentic RAG")
st.markdown("*Ask anything about your documents*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().timestamp()}"

if "export_chat" not in st.session_state:
    st.session_state.export_chat = False

# ============================================================================
# CHAT HISTORY
# ============================================================================

st.subheader("💬 Conversation")

# Display chat history
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        
        else:  # assistant
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
                
                # Show classification
                if "classification" in message:
                    st.caption(f"🏷️ {message['classification']}")
                
                # Show sources if enabled
                if show_sources and "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}** ({source.get('score', 0):.4f})")
                            st.markdown(f"*{source.get('source', 'Unknown')}*")
                            st.write(source.get('text', 'No text available')[:300] + "...")
                
                # Show reasoning if enabled
                if show_reasoning and "reasoning" in message:
                    with st.expander("💭 Reasoning", expanded=False):
                        st.caption(message["reasoning"])

# ============================================================================
# INPUT
# ============================================================================

st.markdown("---")

# Input form
col1, col2 = st.columns([0.9, 0.1])

with col1:
    user_input = st.chat_input(
        "Ask a question...",
        key="chat_input"
    )

with col2:
    send_button = st.button("Send", key="send_btn", use_container_width=True)

# Process input
if user_input or send_button:
    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message immediately
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.write(user_input)
        
        # Show loading spinner
        with st.spinner("🧠 Thinking..."):
            # Query API
            response = query_api(user_input, st.session_state.session_id)
        
        if response:
            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": response.get("response", "No response"),
                "classification": response.get("classification", ""),
                "sources": response.get("sources", []),
                "reasoning": response.get("reasoning", "")
            }
            st.session_state.messages.append(assistant_message)
            
            # Display assistant message
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(response.get("response", "No response"))
                    st.caption(f"🏷️ {response.get('classification', '')}")
                    
                    # Show sources
                    if show_sources and response.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(response.get("sources", []), 1):
                                st.markdown(f"**Source {i}** ({source.get('score', 0):.4f})")
                                st.markdown(f"*{source.get('source', 'Unknown')}*")
                                st.write(source.get('text', '')[:300] + "...")
                    
                    # Show reasoning
                    if show_reasoning:
                        with st.expander("💭 Reasoning", expanded=False):
                            st.caption(response.get("reasoning", ""))
            
            # Auto scroll
            if auto_scroll:
                st.rerun()

# ============================================================================
# EXPORT CHAT
# ============================================================================

if st.session_state.export_chat:
    st.markdown("---")
    st.subheader("📥 Export Conversation")
    
    # Create export data
    export_data = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📄 Download as JSON",
            data=json_str,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Markdown export
        md_content = f"# Chat Export\n\n"
        md_content += f"**Session ID:** {st.session_state.session_id}\n"
        md_content += f"**Exported:** {datetime.now().isoformat()}\n\n"
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                md_content += f"**User:** {msg['content']}\n\n"
            else:
                md_content += f"**Assistant:** {msg['content']}\n\n"
                if msg.get("sources"):
                    md_content += "**Sources:**\n"
                    for src in msg["sources"]:
                        md_content += f"- {src.get('source', 'Unknown')}\n"
                    md_content += "\n"
        
        st.download_button(
            label="📝 Download as Markdown",
            data=md_content,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    if st.button("✅ Done"):
        st.session_state.export_chat = False
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>
        Enterprise Agentic RAG | Built with FastAPI + Streamlit + LangGraph
        <br>
        For support, check the documentation or API logs
    </small>
</div>
""", unsafe_allow_html=True)