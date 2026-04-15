FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install dependencies with pip/uv caches disabled so wheel extract does not fill /root/.cache
# (common on Docker Desktop ARM when the VM disk is tight). After install, delete caches.
# If the build still fails with errno 28: docker builder prune -af && docker system prune -f
# or raise Docker Desktop → Settings → Resources → disk image size.
RUN pip install --no-cache-dir uv && \
    UV_NO_CACHE=1 uv pip install --system --no-cache . && \
    uv cache clean 2>/dev/null || true && \
    rm -rf /root/.cache/pip /root/.cache/uv /root/.local/share/uv 2>/dev/null || true

# Disable CrewAI's interactive tracing prompt (adds 20s delay per crew)
RUN mkdir -p /root/.config/crewai && \
    echo '{"tracing_enabled": false, "traces_viewed": true}' > /root/.config/crewai/settings.json

# Default port
EXPOSE 8080

# Set default credential path - script will write credentials here if GCP_CREDENTIALS_BASE64 is set
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json

# Create startup script that handles GCP credentials for both local dev and Railway
RUN printf '#!/bin/bash\n\
set -e\n\
\n\
echo "=== GCP Credentials Setup ===" \n\
\n\
# Path where docker-compose mounts local ADC credentials\n\
ADC_PATH="/root/.config/gcloud/application_default_credentials.json"\n\
\n\
if [ -n "$GCP_CREDENTIALS_BASE64" ] && [ ${#GCP_CREDENTIALS_BASE64} -gt 10 ]; then\n\
    # Railway production: decode base64 credentials\n\
    echo "Using GCP_CREDENTIALS_BASE64 (Railway mode)"\n\
    echo "$GCP_CREDENTIALS_BASE64" | base64 -d > /app/gcp-credentials.json 2>&1 || echo "Base64 decode failed!"\n\
    if [ -f /app/gcp-credentials.json ]; then\n\
        echo "✅ Credentials decoded to /app/gcp-credentials.json"\n\
        export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-credentials.json\n\
    fi\n\
elif [ -f "$ADC_PATH" ]; then\n\
    # Local development: use mounted ADC credentials\n\
    echo "Using mounted ADC credentials (local development mode)"\n\
    echo "✅ Found ADC at $ADC_PATH"\n\
    # Unset the env var so Google libraries use default ADC path\n\
    unset GOOGLE_APPLICATION_CREDENTIALS\n\
else\n\
    echo "⚠️  No GCP credentials found!"\n\
    echo "   For Railway: Set GCP_CREDENTIALS_BASE64 environment variable"\n\
    echo "   For local Docker: Run '\''gcloud auth application-default login'\'' on host"\n\
    # Unset so BigQuery client shows helpful error\n\
    unset GOOGLE_APPLICATION_CREDENTIALS\n\
fi\n\
\n\
echo "=== Starting uvicorn ===" \n\
# Unset proxy base URLs so LLM calls go direct to providers\n\
unset OPENAI_BASE_URL\n\
# Use python -m so we do not rely on a console_scripts entrypoint on PATH\n\
exec python3 -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]

