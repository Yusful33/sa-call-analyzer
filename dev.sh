#!/usr/bin/env bash
# Stillness Local Development Script
# Starts both API and Web services for local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for required tools
check_requirements() {
    local missing=()
    
    if ! command -v uv &> /dev/null; then
        missing+=("uv (install: curl -LsSf https://astral.sh/uv/install.sh | sh)")
    fi
    
    if ! command -v node &> /dev/null; then
        missing+=("node (install: https://nodejs.org/)")
    fi
    
    if ! command -v npm &> /dev/null; then
        missing+=("npm (comes with node)")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing required tools:${NC}"
        for tool in "${missing[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi
}

# Check for .env file
check_env() {
    if [ ! -f .env ]; then
        echo -e "${YELLOW}No .env file found. Creating from .env.example...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}⚠️  Please edit .env and add your API keys before running.${NC}"
        echo ""
        echo "Minimum required:"
        echo "  ANTHROPIC_API_KEY - Get from https://console.anthropic.com/"
        echo ""
        echo "Optional (for full features):"
        echo "  GONG_ACCESS_KEY, GONG_SECRET_KEY - For Gong integration"
        echo "  BRAVE_API_KEY - For hypothesis research"
        echo "  BigQuery credentials - For prospect overview"
        echo ""
        exit 1
    fi
    
    # Check for required API key
    source .env 2>/dev/null || true
    if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_api_key_here" ]; then
        echo -e "${RED}ANTHROPIC_API_KEY not set in .env${NC}"
        echo "Get your API key from https://console.anthropic.com/"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    
    echo "Installing Python dependencies..."
    (cd apps/api && uv sync --extra hypothesis) || {
        echo -e "${RED}Failed to install Python dependencies${NC}"
        exit 1
    }
    
    echo "Installing Node dependencies..."
    (cd apps/web && npm install) || {
        echo -e "${RED}Failed to install Node dependencies${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Start services
start_services() {
    echo ""
    echo -e "${GREEN}🚀 Starting Stillness...${NC}"
    echo ""
    echo -e "${BLUE}API:${NC} http://localhost:8080"
    echo -e "${BLUE}API Docs:${NC} http://localhost:8080/docs"
    echo -e "${BLUE}Web:${NC} http://localhost:3000"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
    echo ""
    
    trap cleanup EXIT INT TERM
    
    # Start API
    (cd apps/api && uv run python main.py) &
    API_PID=$!
    
    # Wait a moment for API to start
    sleep 2
    
    # Start Web
    (cd apps/web && npm run dev) &
    WEB_PID=$!
    
    # Wait for both processes
    wait
}

# Main
main() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════╗"
    echo "║     Stillness Local Development           ║"
    echo "╚═══════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_requirements
    check_env
    
    # Parse arguments
    case "${1:-}" in
        --install|-i)
            install_deps
            exit 0
            ;;
        --help|-h)
            echo "Usage: ./dev.sh [options]"
            echo ""
            echo "Options:"
            echo "  --install, -i  Install dependencies only"
            echo "  --help, -h     Show this help"
            echo ""
            echo "Environment:"
            echo "  Requires .env file with at least ANTHROPIC_API_KEY"
            echo "  Copy .env.example to .env and configure"
            exit 0
            ;;
    esac
    
    # Check if deps are installed
    if [ ! -d "apps/api/.venv" ] || [ ! -d "apps/web/node_modules" ]; then
        install_deps
    fi
    
    start_services
}

main "$@"
