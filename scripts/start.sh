#!/usr/bin/env bash
# Start the API (container-internal only) and the Streamlit UI (public port) together.
set -e

uvicorn app.main:app --host 127.0.0.1 --port 8000 &

# Wait for the API so the UI doesn't open on "API offline"
python - <<'EOF'
import time, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        break
    except Exception:
        time.sleep(1)
EOF

# Spaces show the app inside an iframe, where Streamlit's XSRF check rejects file uploads
exec streamlit run ui/app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.enableXsrfProtection false
