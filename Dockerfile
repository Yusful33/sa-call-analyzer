FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy all application code first (needed for pyproject.toml build)
COPY . .

# Install uv for faster installs, then install dependencies
RUN pip install uv && uv pip install --system .

# Expose port (Railway will set PORT env var, default to 8080)
EXPOSE 8080

# Run the application with configurable port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

