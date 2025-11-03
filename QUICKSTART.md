# Quick Start Guide - CrewAI Multi-Agent System

Get up and running with 4 AI agents analyzing your SA calls in 5 minutes!

## Setup

1. **Install uv (if you don't have it):**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or via Homebrew
   brew install uv
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Get your Anthropic API key:**
   - Go to https://console.anthropic.com/
   - Sign up or log in
   - Navigate to API Keys section
   - Create a new API key

4. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

5. **Add your API key to .env:**
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

## Running the App

```bash
uv run python main.py
```

Then open http://localhost:8000 in your browser.

## Using the Tool

1. **Paste your transcript** - Copy/paste from Gong, Zoom, or any source
2. **(Optional) Enter SA name** - Or let AI auto-detect
3. **Click "Analyze Call"** - Wait 2-5 minutes for multi-agent analysis
4. **Review feedback** - Get comprehensive insights from 4 expert perspectives

**What's happening during analysis?**
- Agent 1: Identifying the Solution Architect
- Agent 2: Evaluating technical performance
- Agent 3: Scoring sales methodology & discovery
- Agent 4: Compiling actionable recommendations

## Transcript Formats Supported

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

**Mixed format:**
```
Hakan: Yeah, they're so wealthy.
Juan: Yeah.
```

The tool will handle any of these formats automatically.

## Understanding the Results

### Command of the Message Scores (1-10)
- **Problem Identification**: Did SA uncover real business problems?
- **Differentiation**: Did SA articulate unique value vs competitors?
- **Proof/Evidence**: Did SA provide case studies, metrics, demos?
- **Required Capabilities**: Did SA tie features to business outcomes?

### Actionable Insights
Each insight includes:
- **What Happened**: Specific moment from the call (with timestamp)
- **Why It Matters**: Business impact
- **Better Approach**: Exact alternative technique
- **Example Phrasing**: Word-for-word suggestion

### Severity Levels
- 🔴 **Critical**: High impact on deal progression
- 🟡 **Important**: Moderate impact on customer perception
- 🔵 **Minor**: Small improvements for polish

## Tips for Best Results

1. **Use complete transcripts** - More data = better insights
2. **Include timestamps** - Helps with specific feedback
3. **Specify SA if known** - Saves time on auto-detection
4. **Review full context** - AI may miss nuances, use your judgment

## Troubleshooting

**"API key not configured"**
- Make sure .env file exists with your API key
- Restart the server after adding the key

**"Analysis failed"**
- Check your API key is valid
- Ensure you have credits on your Anthropic account
- Try with a shorter transcript first

**No speaker detected**
- Manually enter the SA's name
- Make sure transcript has some context (not just greetings)

**Analysis is slow (2-5 minutes)**
- Normal! Running 4 AI agents takes time
- Each agent does deep analysis
- Worth it for the comprehensive feedback
- See CREWAI_GUIDE.md for details

## Cost Estimate

CrewAI Multi-Agent Analysis (4+ LLM calls):
- **Claude 3.5 Haiku**: ~$0.25-0.50 per call - Recommended
- **Claude 3.5 Sonnet**: ~$1.50-2.50 per call - Best quality
- **Groq (via LiteLLM)**: ~$0.00 (free tier)
- **GPT-4o-mini (LiteLLM)**: ~$0.10-0.25 per call

## Support

Issues or questions? Check the main README.md or create an issue.
