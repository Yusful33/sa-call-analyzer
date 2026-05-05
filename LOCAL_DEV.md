# Stillness Local Development Guide

This guide covers running Stillness locally for development.

## Prerequisites

- **Python 3.10+** with [uv](https://docs.astral.sh/uv/) for dependency management
- **Node.js 18+** with npm
- **Docker** (optional, for Docker Compose setup)
- **Google Cloud SDK** (optional, for BigQuery features)

### Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Minimum required: ANTHROPIC_API_KEY
```

### 2. Start Development

**Option A: Using the dev script (recommended)**
```bash
chmod +x dev.sh
./dev.sh
```

**Option B: Using Make**
```bash
make setup  # First time only
make dev    # Start all services
```

**Option C: Manual**
```bash
# Terminal 1: API
cd apps/api && uv sync --extra hypothesis && uv run python main.py

# Terminal 2: Web
cd apps/web && npm install && npm run dev
```

### 3. Access the Application

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | http://localhost:3000 | Next.js frontend |
| **API** | http://localhost:8080 | FastAPI backend |
| **API Docs** | http://localhost:8080/docs | Interactive OpenAPI docs |

## Environment Variables

### Required

| Variable | Description | Get it from |
|----------|-------------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key | [console.anthropic.com](https://console.anthropic.com/) |

### Optional (Feature-specific)

| Variable | Description | Required for |
|----------|-------------|--------------|
| `GONG_ACCESS_KEY` | Gong API access key | Gong transcript fetch |
| `GONG_SECRET_KEY` | Gong API secret | Gong transcript fetch |
| `BRAVE_API_KEY` | Brave Search API | Hypothesis research |
| `ARIZE_API_KEY` | Arize AX API key | Tracing/observability |
| `ARIZE_SPACE_ID` | Arize AX space ID | Tracing/observability |

### BigQuery (for Prospect Overview)

BigQuery features require Google Cloud credentials. Choose one method:

**Option 1: Application Default Credentials (recommended for local dev)**
```bash
gcloud auth application-default login
```

**Option 2: Service Account**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Docker Compose Setup

For a full containerized setup including LiteLLM proxy:

```bash
# Start all services
make dev-docker

# Or manually
docker compose -f infra/docker-compose.yml up

# View logs
make logs

# Stop services
make down
```

Docker Compose starts:
- **app** (FastAPI) → http://localhost:8080
- **web** (Next.js) → http://localhost:3000
- **litellm** (LLM proxy) → http://localhost:4000
- **gong-mcp** (Gong MCP server) → http://localhost:8081

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│   Next.js Web   │────▶│   FastAPI API   │
│   :3000         │     │   :8080         │
│                 │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
              │           │ │         │ │           │
              │  Claude   │ │ BigQuery│ │   Gong    │
              │  (LLM)    │ │         │ │   API     │
              │           │ │         │ │           │
              └───────────┘ └─────────┘ └───────────┘
```

## Development Commands

```bash
# Full setup (first time)
make setup

# Start development
make dev           # API + Web locally
make dev-docker    # All services via Docker

# Individual services
make dev-api       # API only
make dev-web       # Web only

# Docker operations
make up            # Start detached
make down          # Stop
make logs          # Tail logs
make build         # Rebuild images

# Utilities
make check         # Health check
make test          # Run tests
make clean         # Clean artifacts
```

## Troubleshooting

### API won't start

1. Check Python version: `python --version` (need 3.10+)
2. Check uv is installed: `uv --version`
3. Reinstall deps: `cd apps/api && uv sync --extra hypothesis`

### Web won't connect to API

1. Ensure API is running on port 8080
2. Check no other service is using port 8080
3. The web app auto-proxies `/api/*` to `http://localhost:8080` in development

### BigQuery errors

1. Run `gcloud auth application-default login`
2. Or set `GOOGLE_APPLICATION_CREDENTIALS` env var
3. Ensure your account has BigQuery access to the project

### Docker build fails (disk space)

```bash
docker builder prune -af
docker system prune -f
```

## Testing

```bash
# Run API tests
make test

# Or directly
cd apps/api && uv run pytest
```

## Hot Reload

Both services support hot reload:
- **API**: Uses uvicorn's auto-reload
- **Web**: Uses Next.js fast refresh

## Ports Summary

| Port | Service | Description |
|------|---------|-------------|
| 3000 | Web | Next.js frontend |
| 8080 | API | FastAPI backend |
| 4000 | LiteLLM | LLM proxy (Docker only) |
| 8081 | Gong MCP | Gong server (Docker only) |

## See Also

- [README.md](README.md) - Project overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [docs/deploy-vercel.md](docs/deploy-vercel.md) - Vercel deployment
