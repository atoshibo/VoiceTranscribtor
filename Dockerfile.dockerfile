FROM nvidia/cuda:12.3.2-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy server code
COPY . /app

# Cache models on a volume (recommended)
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HUGGINGFACE_HUB_CACHE=/models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
