# Multi-architecture Dockerfile for Gong MCP HTTP Server
# Builds for both ARM64 (Mac M1/M2) and AMD64 (Railway/cloud)

FROM node:20-alpine AS mcp-builder

# Copy and build the local MCP server
WORKDIR /mcp
COPY mcp-server/package.json mcp-server/tsconfig.json ./
RUN npm install
COPY mcp-server/src ./src
RUN npm run build

# Final image with Python HTTP wrapper
FROM node:20-alpine

# Install Python for the HTTP wrapper
RUN apk add --no-cache python3 py3-pip

# Copy the built MCP server from builder stage
COPY --from=mcp-builder /mcp/dist /app/dist
COPY --from=mcp-builder /mcp/node_modules /app/node_modules
COPY --from=mcp-builder /mcp/package.json /app/package.json

# Install Python dependencies
RUN pip3 install --break-system-packages fastapi uvicorn pydantic

# Copy HTTP wrapper
WORKDIR /app
COPY server.py /app/http_server.py

EXPOSE 8080

# Run the HTTP wrapper
CMD ["python3", "-m", "uvicorn", "http_server:app", "--host", "0.0.0.0", "--port", "8080"]
