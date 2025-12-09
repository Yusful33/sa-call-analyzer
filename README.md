# SA Call Analyzer

AI-powered call analysis using 4 specialized agents to provide actionable feedback for Solution Architects.

## Quick Start

### 1. Install Dependencies

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 2. Configure API Key

```bash
# Create .env file
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

Get your API key from [console.anthropic.com](https://console.anthropic.com/)

### 3. Run the App

```bash
uv run python main.py
```

Open http://localhost:8080 in your browser.

---

## Using the App

1. **Paste a Gong URL** or transcript text
2. Click **Analyze Call**
3. Review actionable feedback with timestamps

Analysis takes 2-5 minutes (4 AI agents working together).

---

## Docker Deployment

Run with Docker Compose (includes LiteLLM proxy):

```bash
# Make sure your .env file has ANTHROPIC_API_KEY set
# Docker Compose reads from .env automatically

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f id-pain
```

Services:
- **id-pain** (port 8080) - Main web app
- **litellm** (port 4000) - LLM proxy (routes requests to Anthropic/OpenAI)
- **gong-mcp** - Gong transcript fetcher

---

## Configuration

Your `.env` file should contain:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional
MODEL_NAME=claude-3-5-haiku-20241022
OPENAI_API_KEY=sk-your-openai-key      # If using OpenAI models
ARIZE_API_KEY=your-arize-key           # For observability
ARIZE_SPACE_ID=your-space-id
GONG_ACCESS_KEY=your-gong-key          # For Gong integration
GONG_ACCESS_KEY_SECRET=your-secret
```

**Note:** Docker Compose automatically uses LiteLLM to route requests. For local development without Docker, set `USE_LITELLM=false` to call Anthropic directly.

---

## How It Works

4 specialized AI agents analyze each call:

1. **SA Identifier** - Detects who the Solution Architect is
2. **Technical Evaluator** - Assesses technical performance
3. **Sales Methodology Expert** - Evaluates discovery & Command of Message
4. **Report Compiler** - Synthesizes actionable recommendations

Each insight includes:
- What happened (with timestamp)
- Why it matters
- Better approach
- Example phrasing

---

## Cost Estimate

| Model | Cost/Call | Quality |
|-------|-----------|---------|
| Claude 3.5 Haiku | $0.25-0.50 | ⭐⭐⭐⭐ |
| Claude 3.5 Sonnet | $1.50-2.50 | ⭐⭐⭐⭐⭐ |

---

## Documentation

- [Gong Integration Guide](docs/GONG_INTEGRATION.md)
- [Agent Details](docs/CREWAI_GUIDE.md)
- [Evaluation Rubric](docs/SA_EVALUATION_RUBRIC.md)

---

## API

```bash
# Analyze with transcript text
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript": "0:16 | John\nHello everyone..."}'

# Analyze with Gong URL
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"gong_url": "https://app.gong.io/call?id=YOUR_CALL_ID"}'

# Health check
curl http://localhost:8080/health
```
