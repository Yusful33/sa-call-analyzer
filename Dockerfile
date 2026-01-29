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

# Set default credential path - script will write credentials here if GCP_CREDENTIALS_BASE64 is set
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json

# Create startup script that decodes GCP credentials from base64 if provided
RUN printf '#!/bin/bash\n\
set -e\n\
\n\
echo "=== Startup Debug ===" \n\
echo "GCP_CREDENTIALS_BASE64 set: $([ -n \"$GCP_CREDENTIALS_BASE64\" ] && echo YES || echo NO)"\n\
echo "GCP_CREDENTIALS_BASE64 length: ${#GCP_CREDENTIALS_BASE64}"\n\
echo "First 50 chars: ${GCP_CREDENTIALS_BASE64:0:50}"\n\
\n\
if [ -n "$GCP_CREDENTIALS_BASE64" ]; then\n\
    echo "Decoding GCP credentials..."\n\
    echo "$GCP_CREDENTIALS_BASE64" | base64 -d > /app/gcp-credentials.json 2>&1 || echo "Base64 decode failed!"\n\
    if [ -f /app/gcp-credentials.json ]; then\n\
        echo "Credentials file created:"\n\
        ls -la /app/gcp-credentials.json\n\
        echo "File contents (first 100 chars):"\n\
        head -c 100 /app/gcp-credentials.json\n\
        echo ""\n\
    else\n\
        echo "ERROR: Credentials file was not created!"\n\
    fi\n\
else\n\
    echo "ERROR: GCP_CREDENTIALS_BASE64 is not set or empty"\n\
    echo "Please set GCP_CREDENTIALS_BASE64 in Railway environment variables"\n\
fi\n\
\n\
echo "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"\n\
echo "=== Starting uvicorn ===" \n\
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]

