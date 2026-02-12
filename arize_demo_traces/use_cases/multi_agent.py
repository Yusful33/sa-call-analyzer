"""Multi-agent orchestration use-case: shared prompts, queries, agent configs, guardrails, evaluators."""

QUERIES = [
    "Research the latest trends in sustainable energy and draft a 500-word executive brief with investment recommendations.",
    "Analyze our Q3 customer churn data and produce an action plan to improve retention by 15%.",
    "Review the competitor landscape for our new product launch and create a positioning document.",
    "Investigate the root cause of the production outage last week and write a post-mortem report.",
    "Evaluate three vendor proposals for our cloud migration and produce a recommendation with cost analysis.",
    "Summarize recent regulatory changes in financial services and assess impact on our compliance posture.",
    "Research best practices for LLM deployment in healthcare and draft internal guidelines.",
    "Analyze customer feedback from the last quarter and identify the top 5 feature requests with business justification.",
]

# Agent role definitions for the multi-agent orchestration
AGENTS = [
    {
        "role": "Research Specialist",
        "goal": "Gather comprehensive, accurate information from available sources",
        "backstory": "You are an expert researcher who excels at finding relevant data, synthesizing information from multiple sources, and identifying key insights.",
    },
    {
        "role": "Analysis Expert",
        "goal": "Analyze data and research findings to extract actionable insights",
        "backstory": "You are a senior analyst skilled at interpreting data, identifying patterns, and drawing evidence-based conclusions.",
    },
    {
        "role": "Writer",
        "goal": "Produce clear, well-structured written outputs based on research and analysis",
        "backstory": "You are a professional writer who transforms complex analysis into clear, compelling documents tailored to the target audience.",
    },
    {
        "role": "Quality Reviewer",
        "goal": "Review outputs for accuracy, completeness, and quality before delivery",
        "backstory": "You are a meticulous reviewer who ensures deliverables meet high standards for factual accuracy, logical consistency, and clarity.",
    },
]

SUPERVISOR_PROMPT = """You are a supervisor agent coordinating a team of specialists. Break down the user's request into subtasks and delegate to the appropriate agents:
- Research Specialist: for information gathering
- Analysis Expert: for data analysis and insight extraction
- Writer: for drafting documents and reports
- Quality Reviewer: for final review and quality assurance

Plan the workflow and delegate tasks in the optimal order."""

RESEARCH_PROMPT = "You are a research specialist. Given the task, gather comprehensive information and present key findings in a structured format."

ANALYSIS_PROMPT = "You are an analysis expert. Given the research findings, analyze the data to extract actionable insights, identify patterns, and draw evidence-based conclusions."

WRITER_PROMPT = "You are a professional writer. Given the analysis and research, produce a clear, well-structured document that addresses the original request."

REVIEWER_PROMPT = "You are a quality reviewer. Review the following document for factual accuracy, logical consistency, completeness, and clarity. Provide a brief assessment and any corrections needed."

GUARDRAILS = [
    {
        "name": "Scope Validation",
        "system_prompt": (
            "Check if the user request is within acceptable scope for an enterprise AI assistant. "
            "Reject requests for personal advice, illegal activities, or topics outside business use. "
            "Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
    {
        "name": "Content Safety Check",
        "system_prompt": (
            "You are a content safety filter. Check if the input contains unsafe, "
            "harmful, or inappropriate content. "
            "Respond ONLY 'PASS' or 'FAIL: <reason>'."
        ),
    },
]

EVALUATORS = [
    {
        "name": "end_to_end_quality_evaluation",
        "criteria": "overall quality — whether the final output is comprehensive, accurate, and well-structured",
    },
    {
        "name": "agent_delegation_evaluation",
        "criteria": "delegation effectiveness — whether subtasks were appropriately broken down and delegated to the right specialists",
    },
]
