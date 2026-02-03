# Smaller Python image for Railway (free tier has 4.8GB limit)
FROM python:3.10-slim-bookworm

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY python/requirements.txt ./requirements.txt

# Install CPU-only PyTorch and other dependencies
# This significantly reduces image size compared to CUDA version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt && \
    pip cache purge && \
    rm -rf ~/.cache/pip

# Copy application code only (not models - they're downloaded at runtime)
COPY python/*.py ./

# Create models directory
RUN mkdir -p models

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Download models and start
CMD python download_models.py ; python app.py
