# LiteLLM Proxy - Railway Deployment
FROM ghcr.io/berriai/litellm:main-latest

# Copy config file
COPY config.yaml /app/config.yaml

# Railway provides PORT env var, LiteLLM needs to listen on it
# Default to 4000 if PORT not set
ENV PORT=4000

# Use shell form to allow variable expansion
CMD litellm --config /app/config.yaml --port $PORT --host 0.0.0.0
