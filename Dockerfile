# ── Build stage: install dependencies into a virtualenv ───────────────────────
FROM python:3.11-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements-serve.txt .
# CPU-only PyTorch wheels are ~5x smaller than the default CUDA build
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

# ── Runtime stage: only the venv and the code ─────────────────────────────────
FROM python:3.11-slim

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY src/    ./src/
COPY app/    ./app/
COPY models/ ./models/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Shell form so $PORT is expanded (Cloud Run and Render inject it)
CMD exec gunicorn app.app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 --access-logfile -
