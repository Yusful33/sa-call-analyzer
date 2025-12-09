FROM ghcr.io/berriai/litellm:main-latest

# Copy config file into the container
COPY config.yaml /app/config.yaml

# Expose port
EXPOSE 4000

# Run LiteLLM with config
CMD ["--config", "/app/config.yaml", "--port", "4000"]
