# ECG Emotion Recognition — Dockerfile
# Works locally (port 8082) and on Hugging Face Spaces (port 7860)

FROM python:3.10-slim

# Install system dependencies needed by PyTorch and NeuroKit2
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (HF Spaces uses 7860, local uses 8082)
EXPOSE 7860
EXPOSE 8082

# Use PORT env variable so same image works locally and on HF Spaces
ENV PORT=8082

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
