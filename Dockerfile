FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY . .

RUN echo 'import sys\nimport pysqlite3\nsys.modules["sqlite3"] = pysqlite3' > /app/sqlite_override.py

RUN chmod -R 755 /app \
    && chown -R root:root /app

LABEL org.opencontainers.image.authors="Kirsten Dizon" \
    org.opencontainers.image.url="https://github.com/Solchararat/multimodal-rag" \
    org.opencontainers.image.documentation="https://github.com/Solchararat/multimodal-rag/blob/main/README.md" \
    org.opencontainers.image.source="https://github.com/Solchararat/multimodal-rag" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.licenses="MIT"

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

RUN ls -la /app && \
    ls -la /app/db || echo "No db directory" && \
    find /app -name "*.sqlite3*" || echo "No SQLite files found"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]