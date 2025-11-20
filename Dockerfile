# Dockerfile for Delphi: Training and API Service
# Build: docker build -t delphi .
# Run: docker run -p 8000:8000 delphi
# Run with GPU: docker run --gpus all -p 8000:8000 delphi

FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create output directory for checkpoints
RUN mkdir -p out delphi

# Create a startup script that trains and then runs ONLY the API (no frontend)
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if checkpoint exists, if not, train the model\n\
if [ ! -f "delphi/ckpt.pt" ]; then\n\
    echo "No checkpoint found. Starting training..."\n\
    echo "Using dataset: ${DATASET:-ukb_simulated_data}"\n\
    python train.py --out_dir=delphi --dataset=${DATASET:-ukb_simulated_data} --max_iters=${MAX_ITERS:-10000} --eval_interval=${EVAL_INTERVAL:-2000}\n\
    echo "Training completed. Ensuring checkpoint is in the right location..."\n\
    # Ensure checkpoint is in delphi/ directory (train.py saves to out_dir)\n\
    if [ -f "delphi/ckpt.pt" ]; then\n\
        echo "Checkpoint already in delphi/ckpt.pt"\n\
    elif [ -f "out/ckpt.pt" ]; then\n\
        echo "Moving checkpoint from out/ to delphi/"\n\
        cp out/ckpt.pt delphi/ckpt.pt\n\
    fi\n\
else\n\
    echo "Checkpoint found at delphi/ckpt.pt. Skipping training."\n\
fi\n\
\n\
# Start ONLY the API server (no frontend)\n\
echo "Starting API server on port ${API_PORT:-8000}..."\n\
uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8000}\n\
' > /app/start-api.sh && chmod +x /app/start-api.sh

# Expose API port
EXPOSE 8000

# Default command: train (if needed) and start API only (no frontend)
CMD ["/app/start-api.sh"]

