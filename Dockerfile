FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install uv for faster installs, then install dependencies
RUN pip install uv && uv pip install --system .

# Default port
EXPOSE 8080

# Create startup script that handles GCP credentials
# Supports both:
# 1. GCP_CREDENTIALS_BASE64 - base64-encoded ADC JSON (for Railway with personal creds)
# 2. GOOGLE_APPLICATION_CREDENTIALS - direct path to credentials file
RUN echo '#!/bin/bash\n\
if [ -n "$GCP_CREDENTIALS_BASE64" ]; then\n\
    echo "Decoding GCP credentials from GCP_CREDENTIALS_BASE64..."\n\
    echo "$GCP_CREDENTIALS_BASE64" | base64 -d > /app/gcp-credentials.json\n\
    export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json\n\
fi\n\
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]

