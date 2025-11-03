# SA Call Analyzer - CrewAI Multi-Agent System

An AI-powered call analysis tool that uses **4 specialized agents** working together to provide comprehensive, actionable feedback for Solution Architects using the Command of the Message framework.

## Features

- 📝 Paste transcript directly (with or without speaker labels)
- 👥 **4 specialized AI agents** collaborate on every analysis:
  1. 🔍 SA Identifier - Detects the Solution Architect
  2. 🛠️ Technical Evaluator - Assesses technical performance
  3. 💡 Sales Methodology & Discovery Expert - Evaluates discovery and Command of Message
  4. 📝 Report Compiler - Synthesizes actionable recommendations
- 📊 Deep analysis against Command of the Message framework
- 💡 Specific, actionable feedback with timestamps and alternative phrasing
- 🔍 Multiple expert perspectives on every call
- 💰 Flexible pricing: $0.25-2.50 per call depending on model choice
- 🔌 Supports Anthropic API, LiteLLM proxy, and local models

## How It Works

Instead of a single AI analyzing your call, **4 specialized agents** work together sequentially:

1. **SA Identifier** determines who the Solution Architect is
2. **Technical Evaluator** reviews technical depth and architecture discussions
3. **Sales Methodology Expert** scores discovery quality and Command of Message pillars
4. **Report Compiler** synthesizes all insights into actionable recommendations

This multi-agent approach provides deeper, more nuanced feedback from multiple expert perspectives.

📖 **See [CREWAI_GUIDE.md](CREWAI_GUIDE.md) for detailed agent descriptions**

## Architecture Overview

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│                      http://localhost:8000                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ POST /api/analyze
                             │ { transcript, sa_name? }
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                                 │
│                      FastAPI Application                         │
├─────────────────────────────────────────────────────────────────┤
│  • Receives transcript via API                                   │
│  • Loads environment config (.env)                               │
│  • Routes to CrewAI analyzer                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   transcript_parser.py                           │
├─────────────────────────────────────────────────────────────────┤
│  • Parses transcript format (with/without labels)                │
│  • Extracts speakers and timestamps                              │
│  • Formats for LLM processing                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Parsed transcript + speakers
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    crew_analyzer.py                              │
│                   SACallAnalysisCrew                             │
├─────────────────────────────────────────────────────────────────┤
│                    🤖 Agent Orchestration                        │
│                                                                  │
│  Agent 1: 🔍 SA Identifier Agent                                │
│  ├─ Role: Identify Solution Architect                           │
│  └─ Output: SA name, confidence                                 │
│           │                                                      │
│           ▼                                                      │
│  Agent 2: 🛠️ Technical Evaluator Agent                         │
│  ├─ Role: Assess technical performance                          │
│  └─ Output: Technical scores + feedback                         │
│           │                                                      │
│           ▼                                                      │
│  Agent 3: 💡 Sales Methodology & Discovery Expert               │
│  ├─ Role: Score discovery + Command of Message                  │
│  └─ Output: Framework scores + discovery feedback               │
│           │                                                      │
│           ▼                                                      │
│  Agent 4: 📝 Report Compiler Agent                              │
│  ├─ Role: Synthesize all agent feedback                         │
│  └─ Output: Complete analysis with actionable insights          │
│           │                                                      │
│           ▼                                                      │
│  📦 Parses JSON, converts to Pydantic models                    │
│  └─ Returns: AnalysisResult                                     │
│                                                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ AnalysisResult (Pydantic model)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        models.py                                 │
│                    Pydantic Data Models                          │
├─────────────────────────────────────────────────────────────────┤
│  • AnalysisResult                                                │
│  • CommandOfMessageScore                                         │
│  • SAPerformanceMetrics                                          │
│  • ActionableInsight                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Structured JSON response
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      User Browser                                │
│                   Results displayed with:                        │
│  • Overall scores                                                │
│  • Top actionable insights with timestamps                       │
│  • Specific alternative phrasing                                 │
│  • Strengths and improvement areas                               │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure & Responsibilities

```
id-pain/
├── main.py                    # FastAPI app, API endpoints, startup
├── crew_analyzer.py           # CrewAI orchestration, 4 agents defined, returns Pydantic models
├── transcript_parser.py       # Parses transcript formats
├── models.py                  # Pydantic data models
├── .env                       # Configuration (API keys, model)
├── frontend/
│   └── index.html            # Web UI for transcript input
├── README.md                 # This file
├── CREWAI_GUIDE.md          # Detailed agent documentation
└── pyproject.toml           # Dependencies (uv)
```

### LLM Integration

The system supports multiple LLM backends:

```
┌──────────────────────────────────────────────────┐
│              crew_analyzer.py                     │
│          (reads .env configuration)               │
└───────────────────┬──────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────────┐
│  Anthropic   │        │    LiteLLM       │
│     API      │        │  (Local Proxy)   │
├──────────────┤        ├──────────────────┤
│ USE_LITELLM  │        │ USE_LITELLM      │
│   = false    │        │   = true         │
│              │        │                  │
│ Uses:        │        │ Supports:        │
│ - Haiku      │        │ - Groq (free)    │
│ - Sonnet     │        │ - GPT-4o-mini    │
│              │        │ - Ollama (local) │
│              │        │ - Any model      │
└──────────────┘        └──────────────────┘
```

### Agent Collaboration Details

Each agent in `crew_analyzer.py` is a CrewAI Agent with:
- **Role**: Specific expertise (e.g., "Technical Evaluator")
- **Goal**: What they're trying to achieve
- **Backstory**: Context that shapes their analysis
- **LLM**: Configured model (Haiku/Sonnet/LiteLLM)

Agents run **sequentially**, with later agents accessing earlier agents' analysis through **context sharing**:

```
Agent 1 Output → Agent 2 (sees Agent 1) → Agent 3 (sees 1+2) → ...
```

This creates a **collaborative intelligence** where each agent builds on previous insights.

## Command of the Message Framework

The analyzer evaluates Solution Architects on:

1. **Problem Identification** - Uncovering customer's business problems
2. **Differentiation** - Articulating unique capabilities vs. competitors
3. **Proof/Evidence** - Providing relevant case studies, metrics, demos
4. **Required Capabilities** - Tying technical features to business outcomes

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) - Modern Python package manager (replaces pip/venv)
  ```bash
  # Install uv (macOS/Linux)
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Or via Homebrew
  brew install uv
  ```
- Choose ONE of:
  - Anthropic API key ([get one here](https://console.anthropic.com/)) - Recommended
  - LiteLLM proxy running locally (free options available)
  - Any OpenAI-compatible API endpoint

### Installation

1. Clone or download this project
2. Install dependencies with uv:
   ```bash
   uv sync
   ```
   This will automatically create a virtual environment and install all dependencies.

3. Configure your model (choose one option):

   **Option A: Anthropic (Recommended - Great balance of cost/quality)**
   ```bash
   cp .env.example .env
   # Edit .env and set:
   # ANTHROPIC_API_KEY=your_key_here
   # MODEL_NAME=claude-3-5-haiku-20241022  (cheap ~$0.10/call)
   ```

   **Option B: LiteLLM with your local proxy (FREE options available)**
   ```bash
   cp .env.example .env
   # Edit .env and set:
   # USE_LITELLM=true
   # LITELLM_BASE_URL=http://localhost:4000
   # MODEL_NAME=groq/llama-3.1-70b-versatile  (or any model you configured)
   ```

   📖 **See [COST_GUIDE.md](COST_GUIDE.md) for detailed setup and cost comparison**

### Running the Application

```bash
uv run python main.py
```

Or activate the virtual environment and run directly:
```bash
source .venv/bin/activate  # On macOS/Linux
python main.py
```

Then open http://localhost:8000 in your browser.

You'll see all 4 agents listed on startup:
```
🤖 Using CrewAI Multi-Agent System (4 specialized agents)
   1. 🔍 SA Identifier
   2. 🛠️ Technical Evaluator
   3. 💡 Sales Methodology & Discovery Expert
   4. 📝 Report Compiler
```

📖 **Want to understand what each agent does?** Read [CREWAI_GUIDE.md](CREWAI_GUIDE.md)

## Usage

1. Paste your call transcript into the text area
2. (Optional) Specify who the SA is, or let AI auto-detect
3. Click "Analyze Call"
4. Review actionable feedback with specific timestamps and recommendations

## Example Transcript Format

The tool handles various formats:

**With speaker labels:**
```
0:16 | Hakan
Yeah, they're so wealthy.

0:17 | Juan
Yeah.
```

**Without labels:**
```
Yeah, they're so wealthy.
Yeah.
```

## API Endpoints

- `POST /api/analyze` - Analyze a transcript
- `GET /health` - Health check

## Cost & Performance

CrewAI runs **4+ LLM calls** per analysis (one per agent). Cost depends on your model choice:

| Model | Cost/Call | Quality | Speed | Best For |
|-------|-----------|---------|-------|----------|
| **Claude 3.5 Haiku** | **$0.25-0.50** | ⭐⭐⭐⭐ | ⚡⚡ | Regular use, cost-effective |
| Claude 3.5 Sonnet | $1.50-2.50 | ⭐⭐⭐⭐⭐ | ⚡ | Maximum insight, important calls |
| **Groq (via LiteLLM)** | **~$0.00** | ⭐⭐⭐ | ⚡⚡⚡ | Budget option, free tier |
| GPT-4o-mini (LiteLLM) | $0.10-0.25 | ⭐⭐⭐⭐ | ⚡⚡ | Good balance |

**Analysis Time:** 2-5 minutes (worth it for the depth!)

📖 **See [CREWAI_GUIDE.md](CREWAI_GUIDE.md) for detailed cost breakdown**

## Tech Stack

- **Backend**: FastAPI (Python)
- **AI**: Multiple LLM options (Claude, GPT, Llama, etc.)
- **Frontend**: HTML/JavaScript (vanilla)
- **API Gateway**: Direct or via LiteLLM proxy
