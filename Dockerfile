# ==========================================
# Control IVR Pipeline - GPU Docker Image
# ==========================================
# Based on PyTorch with CUDA support

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Labels
LABEL description="Control IVR Audio Compliance Pipeline (GPU)"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing libs
    libsndfile1 \
    ffmpeg \
    # PostgreSQL client
    libpq-dev \
    # Build tools (for some pip packages)
    build-essential \
    gcc \
    # Useful utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY run_cron.sh .
COPY config/ ./config/
COPY database/ ./database/
COPY modules/ ./modules/
COPY pipeline/ ./pipeline/
COPY services/ ./services/
COPY storage/ ./storage/

# Make cron script executable
RUN chmod +x run_cron.sh

# Create directories for output and logs
RUN mkdir -p /app/output /app/logs /app/models

VOLUME ["/app/output", "/app/logs", "/app/models"]

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Default command - run the cron script
ENTRYPOINT ["/bin/bash"]
CMD ["./run_cron.sh"]

