# Enterprise RAG - one container running the FastAPI backend and the Streamlit UI.
# Built for Hugging Face Spaces (Docker SDK), works with any Docker host.

FROM python:3.11-slim

# Spaces run containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    API_BASE_URL=http://127.0.0.1:8000
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding and reranking models into the image so cold starts don't download them
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
 && python -c "from flashrank import Ranker; Ranker()"

COPY --chown=user . .
RUN mkdir -p data/uploads

EXPOSE 7860
CMD ["bash", "scripts/start.sh"]
