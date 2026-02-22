FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy app files
COPY app_chainlit.py .
COPY chainlit.md .
COPY .chainlit/ .chainlit/
COPY MedQuAD_combined.csv .
COPY embeddings_cache.npy .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# GROQ_API_KEY is injected by HF Spaces as an environment variable (set in Space Secrets)
CMD ["chainlit", "run", "app_chainlit.py", "--host", "0.0.0.0", "--port", "7860"]
