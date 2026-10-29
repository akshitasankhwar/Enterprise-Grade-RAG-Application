# Streamlit UI Guide

The Streamlit UI (`ui/app.py`) is where users build their document library and ask questions about it.

## Running

Start the API first, then the UI:

```bash
uvicorn app.main:app --reload
streamlit run ui/app.py
```

Open http://localhost:8501. If the API runs on a port other than 8000, set `API_BASE_URL` before starting Streamlit (VS Code's **API + UI** launch config does this for port 8001).

## Layout

**Sidebar (Library)**
- **Add documents**: drop PDF, DOCX, TXT, MD, CSV or HTML files (up to 50 MB each), then click **Add to library**. A toast confirms each file and how many passages it produced.
- **Library**: every document in the knowledge base with its passage count. The trash icon removes a document after you confirm.
- **Display**: toggle **Show sources** and **Show how answers were found**.
- **Conversation**: **New chat** clears the chat; **Export** downloads it as JSON or Markdown.
- A status line at the bottom shows whether the UI can reach the API.

**Main area**
- Before the first question: a short introduction, plus suggested questions once the library has documents.
- Answers cite their passages as `[1]`, `[2]`, … Expandable **Sources** shows the passages with the same numbers, plus their file and page (PDFs). Cited passages are highlighted; passages the answer didn't use are greyed out and marked "not cited".
- A note under each answer says where it came from: "Answered from 2 passages in your library", "Nothing relevant found in your library" (the documents don't cover the question), or "General reply. No documents searched." (small talk).

## Customization

| What | Where |
|------|-------|
| Colors, fonts, corner radius | `[theme]` in `.streamlit/config.toml` |
| Max upload size | `maxUploadSize` in `.streamlit/config.toml` |
| API address | `API_BASE_URL` environment variable |
| Require a password to open the app | `APP_PASSWORD` environment variable (unset = no sign-in) |
| Answer / upload timeouts | `API_TIMEOUT`, `UPLOAD_TIMEOUT` in `ui/app.py` |
| Suggested questions | `SUGGESTED_QUESTIONS` in `ui/app.py` |
| Passage and highlight styling | the `<style>` block at the top of `ui/app.py` |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Sidebar says "API offline" | Start the API (`uvicorn app.main:app --reload`) and check `API_BASE_URL` |
| "Couldn't add files" | Check the API terminal for the error; make sure Qdrant is running |
| "No text found in file.pdf" | The PDF is scanned images; run OCR on it first |
| "I couldn't find that in your documents" | Check the document appears in the Library and rephrase using words from it; for stricter or looser matching adjust `MIN_RELEVANCE` in `.env` |
| Port 8501 already in use | `streamlit run ui/app.py --server.port 8502` |
