FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal files first for caching
COPY requirements.txt ./

# Install only runtime deps (optimize: consider a runtime requirements file)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask==3.0.3 flask-cors==5.0.0 && \
    python - <<'PY'
import nltk
try:
    nltk.download('punkt', quiet=True)
    # Some NLTK builds split tokenizers as punkt_tab; download if available
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print('NLTK download warning:', e)
PY

# Copy source
COPY src ./src
COPY config ./config
COPY qdrant_config ./qdrant_config
COPY README.md ./README.md

# Expose port
EXPOSE 31415

# Data dir inside container (bind via volume)
RUN mkdir -p /app/data

ENV CITEWEAVE_API_HOST=0.0.0.0 \
    CITEWEAVE_API_PORT=31415 \
    CITEWEAVE_ENV=production \
    CITEWEAVE_DATA_DIR=/app/data \
    CITEWEAVE_SETTINGS_PATH=/app/data/settings.json

# Entrypoint to prep data/settings and wait for deps
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["python", "-m", "src.api.server"]


