# Diabetes Health Indicators prediction service (FastAPI + uvicorn).
#
# Build:   docker build -t diabetes-predictor .
# Run:     docker run --rm -p 9696:9696 diabetes-predictor
# Test:    curl http://localhost:9696/health
#          curl -X POST http://localhost:9696/predict \
#               -H "Content-Type: application/json" -d @sample_request.json
#
# The server listens on $PORT (Render sets it automatically) and falls back to
# 9696 when it is not set.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (better layer caching). Slim serving deps only.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Application code + model artifact
COPY scoring.py predict.py model.bin sample_request.json ./

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 9696
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
