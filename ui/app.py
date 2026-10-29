"""
Enterprise Agentic RAG - Streamlit UI

Upload documents to the library, then ask questions answered from them.
"""

import hmac
import html
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG",
    page_icon=":material/menu_book:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 120  # The first answer also loads the embedding and reranking models
UPLOAD_TIMEOUT = 300  # Embedding large files can take a while

# Optional shared password for public deployments (unset = no sign-in)
APP_PASSWORD = os.getenv("APP_PASSWORD")

# File types the backend can ingest
SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md", "csv", "html", "htm"]

# Starter questions that work for most document sets
SUGGESTED_QUESTIONS = [
    "Summarize the main points of these documents",
    "List the key dates and deadlines mentioned",
    "What decisions or action items are mentioned?",
]

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/menu_book:"

# Styling - quiet paper-and-ink interface; cited passages carry a highlighter mark
st.markdown("""
<style>
    :root {
        --ink: #172033;
        --muted: #5D6679;
        --rule: #DCE1E8;
        --accent: #2B4C9B;
        --highlight: rgba(252, 230, 138, 0.9);
        --ok: #2E8A5F;
        --down: #B93A26;
    }

    [data-testid="stHeader"] { background: transparent; }

    /* Landing */
    .lede { color: var(--muted); font-size: 1.05rem; line-height: 1.6; max-width: 34rem; margin: -0.25rem 0 1.25rem; }

    /* Sidebar */
    .wordmark { font-family: "Literata", serif; font-size: 1.4rem; font-weight: 600; color: var(--ink); margin: 0; }
    .wordmark-sub { color: var(--muted); font-size: 0.85rem; margin: 0.15rem 0 0; }
    .side-heading { font-weight: 600; font-size: 0.95rem; color: var(--ink); margin: 1.4rem 0 0.35rem; }
    .side-heading .count { color: var(--muted); font-weight: 400; margin-left: 0.35rem; }
    .doc-row { display: flex; flex-direction: column; min-width: 0; }
    .doc-name { font-size: 0.9rem; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .doc-meta { font-size: 0.78rem; color: var(--muted); }
    .note { font-size: 0.85rem; color: var(--muted); line-height: 1.5; margin: 0; }
    .status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; color: var(--muted); margin-top: 1.5rem; }
    .status .dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; background: var(--ok); flex: none; }
    .status.down .dot { background: var(--down); }

    /* Answers read like a document */
    [class*="st-key-answer-"] [data-testid="stMarkdownContainer"] p,
    [class*="st-key-answer-"] [data-testid="stMarkdownContainer"] li {
        font-family: "Literata", serif; font-size: 1.02rem; line-height: 1.7;
    }
    .answer-note { font-size: 0.8rem; color: var(--muted); margin: 0.1rem 0 0.4rem; }

    /* Sources: the passages an answer used, marked like a highlighted printout */
    .source { border-left: 3px solid var(--highlight); padding: 0.1rem 0 0.1rem 0.85rem; margin: 0.4rem 0 1rem; }
    .source-head { display: flex; align-items: baseline; gap: 0.6rem; flex-wrap: wrap; font-size: 0.8rem; color: var(--muted); margin-bottom: 0.3rem; }
    .source-num { font-weight: 600; color: var(--accent); }
    .source-file { font-weight: 600; color: var(--ink); }
    .source-text { font-family: "Literata", serif; font-size: 0.92rem; line-height: 1.7; color: var(--ink); margin: 0; }
    .source.uncited { border-left-color: var(--rule); }
    .source.uncited .source-text { color: var(--muted); }
    .source-text mark {
        color: inherit; padding: 0 0.12em;
        background: linear-gradient(transparent 10%, var(--highlight) 10%, var(--highlight) 90%, transparent 90%);
        -webkit-box-decoration-break: clone; box-decoration-break: clone;
    }
    .steps { margin: 0.1rem 0 0; padding-left: 1.2rem; font-size: 0.85rem; color: var(--muted); }
    .steps li { margin: 0.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API CLIENT
# ============================================================================

def error_message(response: requests.Response) -> str:
    """Pull the error text out of an API error response"""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    return body.get("error") or body.get("detail") or f"HTTP {response.status_code}"


def check_api_health() -> bool:
    """Check if API is running"""
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_documents() -> Optional[Dict]:
    """Get the documents in the knowledge base"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        return response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None


def query_api(query: str, session_id: Optional[str] = None) -> Optional[Dict]:
    """Ask the agent a question"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "session_id": session_id},
            timeout=API_TIMEOUT
        )
    except requests.exceptions.Timeout:
        st.error("The answer took too long. Try a shorter or more specific question.")
        return None
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the API at {API_BASE_URL}. Start it, then ask again.")
        return None

    if response.status_code != 200:
        st.error(f"Couldn't answer: {error_message(response)}")
        return None
    return response.json()


def upload_files_api(files) -> Optional[Dict]:
    """Send files to API to be added to the knowledge base"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=[
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in files
            ],
            timeout=UPLOAD_TIMEOUT
        )
    except requests.exceptions.Timeout:
        st.error("Indexing took too long. Try fewer or smaller files.")
        return None
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the API at {API_BASE_URL}. Start it, then add the files again.")
        return None

    if response.status_code != 200:
        st.error(f"Couldn't add files: {error_message(response)}")
        return None
    return response.json()


def delete_document_api(filename: str) -> bool:
    """Remove a document from the knowledge base"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{requests.utils.quote(filename, safe='')}",
            timeout=30
        )
    except requests.exceptions.RequestException:
        st.error(f"Can't reach the API at {API_BASE_URL}.")
        return False

    if response.status_code != 200:
        st.error(f"Couldn't remove {filename}: {error_message(response)}")
        return False
    return True


# ============================================================================
# RENDERING
# ============================================================================

def plural(count: int, word: str) -> str:
    """'1 passage', '3 passages'"""
    return f"{count} {word}{'' if count == 1 else 's'}"


def md_escape(text: str) -> str:
    """Escape characters that markdown would treat as formatting"""
    for char in "\\*_`[]#":
        text = text.replace(char, f"\\{char}")
    return text


def cited_numbers(message: Dict) -> List[int]:
    """Source numbers the answer cites, e.g. [1], [2][3] or [1, 3]"""
    count = len(message.get("sources") or [])
    cited = set()
    for group in re.findall(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", message.get("content", "")):
        cited.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(n for n in cited if 1 <= n <= count)


def answer_note(message: Dict, cited: List[int]) -> str:
    """Plain-language note on where an answer came from"""
    if message.get("classification") == "CONVERSATIONAL":
        return "General reply. No documents searched."
    if not message.get("sources"):
        return "Nothing relevant found in your library."
    if cited:
        return f"Answered from {plural(len(cited), 'passage')} in your library."
    return "This answer doesn't cite a passage. Check the sources below."


def passage_text(text: str, limit: int = 420) -> str:
    """Readable one-paragraph snippet of a stored passage"""
    # Drop Markdown heading, list and bold markers that come through from .md uploads
    text = re.sub(r"(^|\s)(#{1,6}|[-*+])\s+", r"\1", text)
    text = text.replace("**", "")
    # Collapse whitespace so blank lines can't break the HTML block
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def render_sources(sources: List[Dict], cited: List[int]):
    """Show each passage as [n] with its file and page, matching the [n] citations in the answer"""
    cards = []
    for number, source in enumerate(sources, 1):
        filename = source.get("source", "Unknown")
        # Only PDFs have real pages; other loaders count paragraphs or whole files
        page = source.get("page")
        page_html = f"<span>page {page + 1}</span>" if page is not None and filename.lower().endswith(".pdf") else ""

        # Passages the answer didn't use lose the highlight
        is_cited = not cited or number in cited
        text = html.escape(passage_text(source.get("text", "")))

        cards.append(
            f'<div class="source{"" if is_cited else " uncited"}"><div class="source-head">'
            f'<span class="source-num">[{number}]</span>'
            f'<span class="source-file">{html.escape(filename)}</span>{page_html}'
            f'{"" if is_cited else "<span>not cited</span>"}'
            f'</div><div class="source-text">{f"<mark>{text}</mark>" if is_cited else text}</div></div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_steps(reasoning: str):
    """Show the agent's pipeline steps in order"""
    steps = [step.strip() for step in reasoning.split("→") if step.strip()]
    items = "".join(f"<li>{html.escape(step)}</li>" for step in steps)
    st.markdown(f'<ol class="steps">{items}</ol>', unsafe_allow_html=True)


def render_answer(message: Dict, key: str):
    """Render an assistant message with its sources and reasoning"""
    cited = cited_numbers(message)

    with st.container(key=f"answer-{key}"):
        st.markdown(message["content"])
    st.markdown(f'<div class="answer-note">{html.escape(answer_note(message, cited))}</div>', unsafe_allow_html=True)

    if show_sources and message.get("sources"):
        with st.expander(f"Sources ({len(message['sources'])})"):
            render_sources(message["sources"], cited)

    if show_steps and message.get("reasoning"):
        with st.expander("How this answer was found"):
            render_steps(message["reasoning"])


def export_json() -> str:
    """Conversation as JSON"""
    return json.dumps({
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }, indent=2)


def export_markdown() -> str:
    """Conversation as readable Markdown"""
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
    return md_content


def require_password():
    """Ask for APP_PASSWORD (when set) before showing anything else"""
    if not APP_PASSWORD or st.session_state.get("signed_in"):
        return

    st.title("Enterprise RAG", anchor=False)
    with st.form("sign_in"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(password.encode(), APP_PASSWORD.encode()):
                st.session_state.signed_in = True
                st.rerun()
            st.error("Wrong password. Try again.")
    st.stop()


require_password()


# ============================================================================
# SESSION STATE
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().timestamp()}"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Feedback from the previous run (uploads and removals trigger a rerun)
upload_result = st.session_state.pop("upload_result", None)
if upload_result:
    for f in upload_result["files"]:
        if f["chunks"]:
            st.toast(f"Added {f['filename']} ({plural(f['chunks'], 'passage')})", icon=":material/check_circle:")
        else:
            st.toast(f"No text found in {f['filename']}. Scanned PDFs need OCR first.", icon=":material/warning:")
    for name in upload_result.get("skipped", []):
        st.toast(f"Skipped {name}: unsupported file type", icon=":material/block:")

removed_file = st.session_state.pop("removed_file", None)
if removed_file:
    st.toast(f"Removed {removed_file}", icon=":material/delete:")


# ============================================================================
# SIDEBAR - LIBRARY
# ============================================================================

api_online = check_api_health()
library = get_documents() if api_online else None
documents = library["documents"] if library else []

with st.sidebar:
    st.markdown(
        '<div class="wordmark">Enterprise RAG</div>'
        '<div class="wordmark-sub">Answers from your own documents</div>',
        unsafe_allow_html=True
    )

    # Add documents
    st.markdown('<div class="side-heading">Add documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Add documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
        disabled=not api_online
    )

    if uploaded_files:
        if st.button(
            f"Add {plural(len(uploaded_files), 'file')} to library",
            type="primary",
            icon=":material/upload:",
            width="stretch"
        ):
            with st.spinner("Reading and indexing…"):
                result = upload_files_api(uploaded_files)

            if result:
                # Keep the result across the rerun, clear the uploader, refresh the library
                st.session_state.upload_result = result
                st.session_state.uploader_key += 1
                st.rerun()

    # Library
    st.markdown(
        f'<div class="side-heading">Library<span class="count">{plural(len(documents), "file")}</span></div>',
        unsafe_allow_html=True
    )

    if not api_online:
        st.markdown('<div class="note">The library is unavailable while the API is offline.</div>', unsafe_allow_html=True)
    elif not documents:
        st.markdown('<div class="note">No documents yet. Files you add appear here.</div>', unsafe_allow_html=True)

    for doc in documents:
        name_col, action_col = st.columns([0.82, 0.18], vertical_alignment="center", gap="small")

        with name_col:
            name = html.escape(doc["filename"])
            st.markdown(
                f'<div class="doc-row"><span class="doc-name" title="{name}">{name}</span>'
                f'<span class="doc-meta">{plural(doc["chunks"], "passage")}</span></div>',
                unsafe_allow_html=True
            )

        with action_col:
            with st.popover(":material/delete:", help=f"Remove {doc['filename']}"):
                st.markdown(f"Remove **{md_escape(doc['filename'])}** from the library? You can add it again later.")
                if st.button("Remove", key=f"remove_{doc['filename']}", type="primary", width="stretch"):
                    if delete_document_api(doc["filename"]):
                        st.session_state.removed_file = doc["filename"]
                        st.rerun()

    # Display options
    st.markdown('<div class="side-heading">Display</div>', unsafe_allow_html=True)
    show_sources = st.toggle("Show sources", value=True)
    show_steps = st.toggle("Show how answers were found", value=False)

    # Conversation
    if st.session_state.messages:
        st.markdown('<div class="side-heading">Conversation</div>', unsafe_allow_html=True)
        col_new, col_export = st.columns(2)

        with col_new:
            if st.button("New chat", icon=":material/add_comment:", width="stretch"):
                st.session_state.messages = []
                st.rerun()

        with col_export:
            with st.popover("Export", icon=":material/download:", width="stretch"):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "Download JSON",
                    data=export_json(),
                    file_name=f"chat_{stamp}.json",
                    mime="application/json",
                    width="stretch"
                )
                st.download_button(
                    "Download Markdown",
                    data=export_markdown(),
                    file_name=f"chat_{stamp}.md",
                    mime="text/markdown",
                    width="stretch"
                )

    # Connection status
    if api_online:
        st.markdown('<div class="status"><span class="dot"></span>Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="status down"><span class="dot"></span>API offline at {html.escape(API_BASE_URL)}</div>',
            unsafe_allow_html=True
        )


# ============================================================================
# MAIN - CONVERSATION
# ============================================================================

prompt = st.chat_input("Ask a question about your documents", disabled=not api_online)
if not prompt:
    prompt = st.session_state.pop("queued_prompt", None)

# Landing view until the first question
if not st.session_state.messages and not prompt:
    st.title("Ask your documents", anchor=False)

    if not api_online:
        st.markdown(
            '<div class="lede">The app can\'t reach its API right now. Start the API, then refresh this page.</div>',
            unsafe_allow_html=True
        )
        st.code("uvicorn app.main:app --port 8001", language="bash")
    elif not documents:
        st.markdown(
            '<div class="lede">Add PDF, Word, text, Markdown, CSV or HTML files to the library in the sidebar. '
            'Then ask questions here, and each answer shows the passages it came from.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="lede">{plural(len(documents), "document")} in your library. '
            'Ask anything about them, and each answer shows the passages it used.</div>',
            unsafe_allow_html=True
        )
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggest_{question}", icon=":material/chat_bubble:"):
                st.session_state.queued_prompt = question
                st.rerun()

# Chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            render_answer(message, str(i))

# New question
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching your documents…"):
            response = query_api(prompt, st.session_state.session_id)

    if response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("response", "No response"),
            "classification": response.get("classification", ""),
            "sources": response.get("sources", []),
            "reasoning": response.get("reasoning", "")
        })
        st.rerun()
    else:
        # Drop the unanswered question; the error above explains what happened
        st.session_state.messages.pop()
