# CrewAI Multi-Agent Analysis Guide

## Overview

Your SA Call Analyzer uses **CrewAI** - a multi-agent framework where **4 specialized AI agents** collaborate to provide comprehensive feedback on Solution Architect performance.

## Architecture

### CrewAI Multi-Agent Flow (4+ LLM calls)
```
User → Agent 1 (SA Identifier)
     → Agent 2 (Technical Evaluator)
     → Agent 3 (Sales Methodology & Discovery Expert)
     → Agent 4 (Report Compiler)
     → Comprehensive Results
```
- Thorough (~2-5 minutes)
- Cost: ~$0.25-0.50 per call with Haiku, ~$1.50-2.50 with Sonnet
- Exceptional quality with multiple expert perspectives

## The 4 Specialized Agents

### 1. 🔍 SA Identifier Agent
**Role**: Detective
**Expertise**: Identifying Solution Architects
**Analyzes**:
- Speaking patterns
- Technical depth of contributions
- Architecture/integration discussions
- Who answers technical questions

**Output**: SA name, confidence level, reasoning

### 2. 🛠️ Technical Evaluator Agent
**Role**: Senior Technical Architect
**Expertise**: Evaluating technical performance
**Scores** (1-10):
- Technical depth and accuracy
- Architecture discussions quality
- Demo effectiveness

**Provides**:
- Specific feedback on technical explanations
- Better ways to explain complex concepts
- Assessment of technical credibility
- Timestamps of key technical moments

### 3. 💡 Sales Methodology & Discovery Expert Agent
**Role**: Sales Coach & Command of Message Expert
**Expertise**: Sales methodology and discovery

**Evaluates TWO areas**:

#### A. Discovery & Engagement
Scores (1-10):
- Quality of discovery questions
- Active listening skills
- Customer engagement
- Pain point identification

**Provides**:
- Missed discovery opportunities
- Better questions to ask
- Active listening improvements

#### B. Command of the Message Framework
Scores each pillar (1-10):
1. **Problem Identification** - Uncovering business problems
2. **Differentiation** - Unique value vs competitors
3. **Proof/Evidence** - Case studies, metrics, demos
4. **Required Capabilities** - Features → Business outcomes

**Provides**:
- Specific examples with timestamps
- Missed opportunities for each pillar
- Alternative phrasing suggestions
- Actionable recommendations

### 4. 📝 Report Compiler Agent
**Role**: Executive Coach
**Expertise**: Synthesizing feedback into actionable insights
**Creates**:
- Overall performance score (1-10)
- All individual scores from technical and sales agents
- Top 3-5 high-impact improvements with:
  - Category (which skill area)
  - Severity (critical/important/minor)
  - Timestamp (if available)
  - What happened
  - Why it matters
  - Better approach
  - Example phrasing
- Strengths to reinforce (2-3 items)
- Improvement areas (2-3 items)
- Key call moments with timestamps

## Agent Collaboration

### How Agents Work Together

1. **Sequential Process**: Agents work one after another
2. **Context Sharing**: Later agents see earlier agents' analysis
3. **Specialization**: Each agent focuses on their domain
4. **Synthesis**: Report Compiler agent combines all insights

### Example Flow:
```
1. SA Identifier: "Hakan is the SA (high confidence)"
   ↓
2. Technical Evaluator: "Strong architecture discussion,
   but missed opportunity to explain benefits at 12:45
   Technical depth: 8/10"
   ↓
3. Sales Methodology & Discovery Expert:
   "Good discovery (7/10) but weak on differentiation (5/10)
   Missed opportunity to position against competitors at 8:30"
   ↓
4. Report Compiler: Synthesizes all feedback into
   "Top 3 Improvements with exact timestamps and alternatives:
   1. Differentiation (critical) - At 8:30, could have said...
   2. Technical explanation (important) - At 12:45, try...
   3. Discovery follow-up (important) - At 15:20, ask..."
```

## Cost & Performance

### Per-Call Costs
| Model | Cost/Call | Quality | Speed | Best For |
|-------|-----------|---------|-------|----------|
| **Claude 3.5 Haiku** | **$0.25-0.50** | ⭐⭐⭐⭐ | ⚡⚡ | Regular use, cost-effective |
| Claude 3.5 Sonnet | $1.50-2.50 | ⭐⭐⭐⭐⭐ | ⚡ | Maximum insight, important calls |
| **Groq (via LiteLLM)** | **~$0.00** | ⭐⭐⭐ | ⚡⚡⚡ | Budget option, free tier |
| GPT-4o-mini (LiteLLM) | $0.10-0.25 | ⭐⭐⭐⭐ | ⚡⚡ | Good balance |

### Why Multiple Agents?

**Single Agent Analysis**:
- One perspective
- May miss nuanced issues
- Generic feedback
- No cross-validation

**Multi-Agent Analysis**:
- 4 expert perspectives
- Technical + Sales + Discovery combined
- Cross-referenced insights
- Each agent builds on previous analysis
- More specific, actionable feedback

### Analysis Time
- **2-5 minutes** per call
- Worth it for the depth!
- Each agent does deep, specialized analysis
- Final report synthesizes all perspectives

## Output Structure

The final analysis includes:

### Overall Metrics
```json
{
  "overall_score": 7.5,
  "sa_identified": "Hakan",
  "sa_confidence": "high"
}
```

### Command of Message Scores
```json
{
  "problem_identification": 8,
  "differentiation": 6,
  "proof_evidence": 7,
  "required_capabilities": 5
}
```

### SA Performance Metrics
```json
{
  "technical_depth": 8,
  "discovery_quality": 7,
  "active_listening": 9,
  "value_articulation": 5
}
```

### Actionable Insights
Each insight includes:
- **Category**: Which skill area (e.g., "Differentiation")
- **Severity**: critical, important, or minor
- **Timestamp**: When it happened (e.g., "8:30")
- **What happened**: What the SA did/said
- **Why it matters**: Business impact
- **Better approach**: Specific alternative
- **Example phrasing**: Exact words they could use

Example:
```json
{
  "category": "Differentiation",
  "severity": "critical",
  "timestamp": "8:30",
  "what_happened": "Customer asked about competitors, SA gave generic response",
  "why_it_matters": "Missed opportunity to position unique value, customer may not see difference",
  "better_approach": "Acknowledge competitor strengths, then pivot to differentiated capabilities",
  "example_phrasing": "Yes, Competitor X does handle basic data integration. Where we differentiate is our real-time anomaly detection that saved Company Y $2M in their first quarter..."
}
```

## Tips for Best Results

### 1. Provide Complete Transcripts
- Longer transcripts = better analysis
- Include full conversation context
- Don't truncate early or late portions

### 2. Speaker Labels Help (But Aren't Required)
- With labels: More accurate SA identification
- Without labels: Agent will infer based on content

### 3. Allow Time for Analysis
- 2-5 minutes is normal
- Each agent is doing deep analysis
- Don't interrupt the process

### 4. Review Each Section
- Overall scores give the big picture
- Top insights are prioritized by impact
- Strengths show what to reinforce
- Improvement areas show focus opportunities
- Key moments highlight critical timestamps

### 5. Choose the Right Model
- **Haiku**: Great for regular call reviews (~$0.25-0.50/call)
- **Sonnet**: Best for important/complex calls (~$1.50-2.50/call)
- **Groq/LiteLLM**: Budget option, free tier available

## Model Configuration

### Using Anthropic (Recommended)

In `.env`:
```bash
ANTHROPIC_API_KEY=your_key_here
MODEL_NAME=claude-3-5-haiku-20241022
USE_LITELLM=false
```

### Using LiteLLM (Free/Local Models)

In `.env`:
```bash
USE_LITELLM=true
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=dummy
MODEL_NAME=groq/llama-3.1-70b-versatile
```

See [COST_GUIDE.md](COST_GUIDE.md) for detailed setup instructions.

## Technical Implementation

### Dependencies
- `crewai>=0.11.0` - Multi-agent framework
- `langchain-anthropic>=0.1.0` - Anthropic integration
- `langchain-openai>=0.0.5` - OpenAI/LiteLLM integration

### Key Files
- `crew_analyzer.py` - Agent definitions, orchestration, and Pydantic model conversion
- `models.py` - Pydantic data models
- `main.py` - FastAPI application
- `transcript_parser.py` - Transcript parsing utilities

### Agent Definition Example
```python
technical_evaluator = Agent(
    role='Senior Technical Architect & Evaluator',
    goal='Assess the Solution Architect\'s technical depth...',
    backstory="""You are a seasoned technical architect...""",
    llm=self.llm,
    verbose=True,
    allow_delegation=False
)
```

### Task Execution
Agents run **sequentially** with **context sharing**:
```python
analysis_crew = Crew(
    agents=[
        agents['technical_evaluator'],
        agents['sales_methodology_expert'],
        agents['report_compiler']
    ],
    tasks=[
        technical_task,
        sales_methodology_task,
        compile_task
    ],
    process=Process.sequential,
    verbose=True
)
```

## Troubleshooting

### "Analysis takes 2-5 minutes"
- **Normal!** You're running 4 AI agents
- Each agent does deep, specialized analysis
- Worth it for the comprehensive feedback

### "Costs are higher than expected"
- You're running 4+ LLM calls (one per agent)
- Use Haiku model to reduce costs (~$0.25/call)
- Consider Groq via LiteLLM for free tier
- See [COST_GUIDE.md](COST_GUIDE.md) for cost optimization

### "Analysis seems to repeat some points"
- Some overlap is intentional for cross-validation
- Each agent has a specific perspective
- Report Compiler synthesizes unique insights
- Look for the "better approach" and "example phrasing" - that's the value

### "Want more specialized agents?"
Easy to extend! You can add:
- **Competitive Intelligence Agent** - Analyzes competitor mentions
- **ROI Calculator Agent** - Quantifies business impact
- **Demo Quality Agent** - Specialized demo evaluation

To add an agent:
1. Define new agent in `crew_analyzer.py`
2. Create a new task for the agent
3. Add to the crew's agents and tasks lists
4. Update `models.py` if new metrics needed

## Why CrewAI?

### Benefits of Multi-Agent Approach

1. **Specialization**: Each agent is an expert in their domain
2. **Depth**: More thorough analysis than a single agent
3. **Cross-Validation**: Multiple perspectives catch more issues
4. **Actionable**: Specific, timestamp-based recommendations
5. **Context Sharing**: Later agents build on earlier analysis

### Real-World Impact

**Single Agent**: "The SA could improve discovery"
**Multi-Agent**: "At 8:30, when the customer mentioned budget concerns, the SA should have asked 'What's driving that budget constraint?' to uncover the deeper business problem. This would enable better positioning of ROI in the Required Capabilities section."

The difference: **Specific timestamp, exact phrasing, tied to framework**

## Future Enhancements

Potential additions:
- Parallel agent execution for faster analysis
- Trend analysis across multiple calls
- Custom agent configurations per team
- Integration with CRM systems
- Automated action item creation

Want these? Let me know!
