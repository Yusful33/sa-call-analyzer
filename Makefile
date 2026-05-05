# Stillness Local Development Makefile
# Run `make help` to see all available commands

.PHONY: help dev dev-docker dev-api dev-web setup install clean logs stop

# Default target
help:
	@echo "Stillness Local Development"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup     - First-time setup (install deps, copy .env)"
	@echo "  make dev       - Start all services (API + Web, no Docker)"
	@echo "  make dev-docker - Start all services via Docker Compose"
	@echo ""
	@echo "Individual Services:"
	@echo "  make dev-api   - Start FastAPI backend only (port 8080)"
	@echo "  make dev-web   - Start Next.js frontend only (port 3000)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make up        - Start Docker services (detached)"
	@echo "  make down      - Stop Docker services"
	@echo "  make logs      - Tail Docker logs"
	@echo "  make build     - Rebuild Docker images"
	@echo ""
	@echo "Utilities:"
	@echo "  make install   - Install all dependencies"
	@echo "  make clean     - Clean build artifacts and caches"
	@echo "  make test      - Run tests"
	@echo ""
	@echo "Environment:"
	@echo "  Copy .env.example to .env and configure required variables"
	@echo "  Minimum: ANTHROPIC_API_KEY"
	@echo "  Optional: GONG_ACCESS_KEY, GONG_SECRET_KEY, BRAVE_API_KEY"

# ============================================================================
# Setup & Installation
# ============================================================================

setup: .env install
	@echo "✅ Setup complete! Run 'make dev' to start."

.env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "📝 Created .env from .env.example"; \
		echo "⚠️  Edit .env and add your API keys before running"; \
	fi

install: install-api install-web
	@echo "✅ All dependencies installed"

install-api:
	@echo "📦 Installing Python dependencies..."
	cd apps/api && uv sync --extra hypothesis

install-web:
	@echo "📦 Installing Node dependencies..."
	cd apps/web && npm install

# ============================================================================
# Local Development (No Docker)
# ============================================================================

# Start all services locally (requires two terminals or use tmux/screen)
dev:
	@echo "Starting Stillness locally..."
	@echo ""
	@echo "This will start both services. Use Ctrl+C to stop."
	@echo "API: http://localhost:8080 (OpenAPI docs: http://localhost:8080/docs)"
	@echo "Web: http://localhost:3000"
	@echo ""
	@$(MAKE) -j2 dev-api dev-web

dev-api:
	@echo "🚀 Starting FastAPI backend on http://localhost:8080"
	cd apps/api && uv run python main.py

dev-web:
	@echo "🚀 Starting Next.js frontend on http://localhost:3000"
	cd apps/web && npm run dev

# ============================================================================
# Docker Compose
# ============================================================================

dev-docker:
	@echo "🐳 Starting all services via Docker Compose..."
	docker compose -f infra/docker-compose.yml up

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

stop: down

logs:
	docker compose -f infra/docker-compose.yml logs -f

logs-api:
	docker compose -f infra/docker-compose.yml logs -f app

logs-web:
	docker compose -f infra/docker-compose.yml logs -f web

build:
	docker compose -f infra/docker-compose.yml build

# ============================================================================
# Testing
# ============================================================================

test:
	cd apps/api && uv run pytest

test-api:
	cd apps/api && uv run pytest

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

clean-docker:
	docker compose -f infra/docker-compose.yml down -v --rmi local
	docker system prune -f

# ============================================================================
# Health Checks
# ============================================================================

check-api:
	@curl -s http://localhost:8080/health | python -m json.tool || echo "API not running"

check-web:
	@curl -s http://localhost:3000/api/health | python -m json.tool || echo "Web not running"

check: check-api check-web
