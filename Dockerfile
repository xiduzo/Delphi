# Dockerfile for Delphi: Training and API Service
# 
# Build: docker build -t delphi .
# Run: docker run -p 8888:8888 delphi
# Run with GPU: docker run --gpus all -p 8888:8888 delphi
#
# Environment variables:
#   - DATASET: Dataset name (default: ukb_simulated_data)
#   - MAX_ITERS: Maximum training iterations (default: 2000)
#   - EVAL_INTERVAL: Evaluation interval (default: 2000)
#   - API_PORT: API server port (default: 8888)

FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash delphi && \
    mkdir -p /app /workspace && \
    chown -R delphi:delphi /app /workspace

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY --chown=delphi:delphi requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY --chown=delphi:delphi . .

# Create necessary directories
RUN mkdir -p out delphi data/ukb_simulated_data && \
    chown -R delphi:delphi out delphi data

# Create startup script
RUN cat > /app/start-api.sh << 'EOFSCRIPT' && chmod +x /app/start-api.sh
#!/bin/bash
set -e

echo "=========================================="
echo "Delphi API Service Starting..."
echo "=========================================="

# Check if checkpoint exists, if not, train the model
if [ ! -f "out/ckpt.pt" ]; then
    echo "No checkpoint found. Starting training..."
    echo "Using dataset: ${DATASET:-ukb_simulated_data}"
    echo "Max iterations: ${MAX_ITERS:-2000}"
    echo "Eval interval: ${EVAL_INTERVAL:-2000}"
    
    if python train.py \
        --out_dir=out \
        --dataset=${DATASET:-ukb_simulated_data} \
        --max_iters=${MAX_ITERS:-2000} \
        --eval_interval=${EVAL_INTERVAL:-2000}; then
        echo "✓ Training completed successfully."
        
        # Ensure checkpoint is in delphi/ directory
        if [ -f "out/ckpt.pt" ]; then
            echo "Moving checkpoint from out/ to delphi/"
            cp out/ckpt.pt delphi/ckpt.pt
            echo "✓ Checkpoint moved successfully"
        elif [ -f "delphi/ckpt.pt" ]; then
            echo "Moving checkpoint from delphi/ to out/"
            cp delphi/ckpt.pt out/ckpt.pt
            echo "✓ Checkpoint moved successfully"
        else
            echo "ERROR: Checkpoint not found after training!"
            exit 1
        fi
    else
        echo "ERROR: Training failed or was interrupted."
        exit 1
    fi
else
    echo "✓ Checkpoint found at out/ckpt.pt. Skipping training."
fi

# Verify required files exist
if [ ! -f "out/ckpt.pt" ]; then
    echo "ERROR: Checkpoint file not found at out/ckpt.pt"
    exit 1
fi

if [ ! -f "data/ukb_simulated_data/labels.csv" ]; then
    echo "WARNING: Labels file not found at data/ukb_simulated_data/labels.csv"
    echo "API may fail to start without labels file."
fi

# Start the API server
echo "=========================================="
echo "Starting API server on port ${API_PORT:-8888}..."
echo "=========================================="
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port ${API_PORT:-8888} \
    --workers 1 \
    --log-level info
EOFSCRIPT

# Switch to non-root user
USER delphi

# Expose API port
EXPOSE 8888

# Health check (uses default port 8888)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Default command: train (if needed) and start API
CMD ["/app/start-api.sh"]

