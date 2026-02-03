# Hugging Face Spaces Dockerfile
FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY requirements.txt ./requirements.txt

# Install CPU-only PyTorch and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt && \
    pip cache purge && \
    rm -rf ~/.cache/pip

# Copy application code
COPY *.py ./

# Create models directory
RUN mkdir -p models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Hugging Face Spaces uses port 7860
ENV PORT=7860

# Create a non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Download models and start
CMD python download_models.py ; python app.py
