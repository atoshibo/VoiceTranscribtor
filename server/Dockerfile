FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps (web service doesn't need GPU dependencies)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --default-timeout=300 --retries=10 -r requirements.txt

# Copy server code
COPY main.py /app/
COPY templates/ /app/templates/

ENV SESSIONS_DIR=/data/sessions

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
