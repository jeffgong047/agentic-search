# Agentic Search Service Dockerfile
# Multi-stage build for smaller image size

# ====================================================================
# Stage 1: Builder
# ====================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
# Copy requirements first for layer caching
COPY requirements.txt requirements_api.txt requirements_production.txt requirements_benchmark.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements_api.txt -r requirements_production.txt -r requirements_benchmark.txt

# ====================================================================
# Stage 2: Runtime
# ====================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user (optional, for security)
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    AUTO_LOAD_MOCK=true \
    DEBUG_MODE=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

