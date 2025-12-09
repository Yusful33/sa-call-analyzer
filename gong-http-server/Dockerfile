# Build on top of the existing Gong MCP server image (Alpine-based)
FROM yusufarize/gong-mcp:latest

# Install Python for the HTTP wrapper (Alpine uses apk)
RUN apk add --no-cache python3 py3-pip

# Install Python dependencies
RUN pip3 install --break-system-packages fastapi uvicorn

# Copy HTTP wrapper
COPY server.py /app/http_server.py

# The MCP server is at /app/dist/index.js (Node.js)
# Our HTTP wrapper will call it via subprocess

EXPOSE 8080

# Run the HTTP wrapper (which communicates with MCP server internally)
CMD ["python3", "-m", "uvicorn", "http_server:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app"]

