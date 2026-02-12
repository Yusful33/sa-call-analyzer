"""Classification / routing pipeline use-case: shared prompts, queries, categories, guardrails, evaluators."""

import random

QUERIES = [
    "I've been waiting 3 weeks for my refund and no one is responding to my emails. This is unacceptable!",
    "Can you help me configure SSO for my team? We're using Okta.",
    "Your new analytics dashboard is amazing! The real-time metrics are exactly what we needed.",
    "We're evaluating your enterprise plan — can you send over the pricing breakdown and SLA details?",
    "The API has been returning 500 errors intermittently since yesterday's deployment.",
    "I'd like to cancel my subscription effective end of this billing cycle.",
    "How do I integrate your SDK with our existing Python application?",
    "We noticed unusual activity on our account — possible unauthorized access.",
    "We need to add 50 more seats to our enterprise license. What's the process?",
    "The export feature isn't working in Chrome but works fine in Firefox.",
]

CATEGORIES = {
    "billing_refund": {
        "label": "Billing & Refund",
        "description": "Payment issues, refund requests, invoice questions",
        "priority": "high",
        "routing": "billing_team",
    },
    "technical_support": {
        "label": "Technical Support",
        "description": "Technical issues, bugs, integration help, API problems",
        "priority": "medium",
        "routing": "engineering_support",
    },
    "account_management": {
        "label": "Account Management",
        "description": "Account changes, license management, cancellations, upgrades",
        "priority": "medium",
        "routing": "account_managers",
    },
    "sales_inquiry": {
        "label": "Sales Inquiry",
        "description": "Pricing questions, plan comparisons, enterprise inquiries",
        "priority": "medium",
        "routing": "sales_team",
    },
    "security_compliance": {
        "label": "Security & Compliance",
        "description": "Security incidents, unauthorized access, compliance questions",
        "priority": "critical",
        "routing": "security_team",
    },
    "positive_feedback": {
        "label": "Positive Feedback",
        "description": "Compliments, positive reviews, feature appreciation",
        "priority": "low",
        "routing": "product_team",
    },
}

SYSTEM_PROMPT_CLASSIFY = """You are a customer support ticket classifier. Classify the following customer message into exactly ONE category.

Available categories:
- billing_refund: Payment issues, refund requests, invoice questions
- technical_support: Technical issues, bugs, integration help, API problems
- account_management: Account changes, license management, cancellations, upgrades
- sales_inquiry: Pricing questions, plan comparisons, enterprise inquiries
- security_compliance: Security incidents, unauthorized access, compliance questions
- positive_feedback: Compliments, positive reviews, feature appreciation

Return ONLY a JSON object with:
- "category": one of the category keys above
- "confidence": a float between 0.0 and 1.0
- "reasoning": a brief explanation"""

SYSTEM_PROMPT_SENTIMENT = """Analyze the sentiment of the following customer message. Return ONLY a JSON object with:
- "sentiment": one of "positive", "negative", "neutral"
- "intensity": a float between 0.0 and 1.0 (how strong the sentiment is)
- "urgency": one of "low", "medium", "high", "critical" """

SYSTEM_PROMPT_EXTRACT = """Extract key entities from the following customer message. Return ONLY a JSON object with:
- "product_mentioned": product or feature name if any, else null
- "issue_type": specific issue type if any, else null
- "action_requested": what the customer wants, else null
- "time_reference": any time-related mention, else null"""

SYSTEM_PROMPT_RESPONSE = """You are a customer support agent. Based on the classification and extracted entities, generate a helpful, empathetic response to the customer.

Classification: {classification}
Sentiment: {sentiment}
Entities: {entities}

Write a professional response that acknowledges the customer's concern and provides appropriate next steps."""

GUARDRAILS = [
    {
        "name": "PII Detection",
        "system_prompt": (
            "Check if the input contains personally identifiable information (PII) such as "
            "credit card numbers, SSNs, passwords, or API keys that should be redacted. "
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
        "name": "classification_accuracy_evaluation",
        "criteria": "classification accuracy — whether the ticket was routed to the correct category with appropriate confidence",
    },
    {
        "name": "response_appropriateness_evaluation",
        "criteria": "response appropriateness — whether the generated response matches the ticket category, sentiment, and urgency level",
    },
]


def get_expected_category(query: str) -> str:
    """Return a plausible expected category for evaluation purposes."""
    query_lower = query.lower()
    if any(k in query_lower for k in ["refund", "payment", "billing", "invoice", "charge"]):
        return "billing_refund"
    elif any(k in query_lower for k in ["error", "bug", "api", "sdk", "integrate", "not working"]):
        return "technical_support"
    elif any(k in query_lower for k in ["cancel", "seats", "license", "subscription", "upgrade"]):
        return "account_management"
    elif any(k in query_lower for k in ["pricing", "plan", "enterprise", "evaluate"]):
        return "sales_inquiry"
    elif any(k in query_lower for k in ["security", "unauthorized", "breach", "suspicious"]):
        return "security_compliance"
    elif any(k in query_lower for k in ["amazing", "great", "love", "excellent"]):
        return "positive_feedback"
    return random.choice(list(CATEGORIES.keys()))
