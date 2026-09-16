FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.* \
    && pip install -r requirements.txt

COPY --chown=app:app docengine ./docengine
COPY --chown=app:app app.py .

USER app
# Pre-download the embedding model so the first request is fast
RUN python -c "from docengine.embedder import load_model; load_model()"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
