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
echo "=== Startup ===" \n\
echo "GCP_CREDENTIALS_BASE64 length: ${#GCP_CREDENTIALS_BASE64}"\n\
\n\
if [ -n "$GCP_CREDENTIALS_BASE64" ]; then\n\
    echo "Decoding GCP credentials..."\n\
    echo "$GCP_CREDENTIALS_BASE64" | base64 -d > /app/gcp-credentials.json\n\
    echo "Credentials written to /app/gcp-credentials.json"\n\
    ls -la /app/gcp-credentials.json\n\
else\n\
    echo "WARNING: GCP_CREDENTIALS_BASE64 not set"\n\
fi\n\
\n\
echo "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"\n\
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]

