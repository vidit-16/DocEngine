FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only torch keeps the image around 1GB smaller than the default CUDA wheels.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.*" \
    && pip install -r requirements.txt

COPY --chown=app:app src ./src
COPY --chown=app:app app.py .

USER app
# Bake the embedding model into the image so the first upload does not download it.
RUN python -c "from src.embedder import get_model; get_model()"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
