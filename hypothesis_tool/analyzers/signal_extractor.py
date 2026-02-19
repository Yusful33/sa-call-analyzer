"""Signal extractor - analyzes research results to extract AI/ML signals."""

import re
from enum import Enum

import anthropic

from ..clients.brave_client import SearchResult
from ..models.hypothesis import (
    AIMLSignal,
    SignalConfidence,
    CompetitiveSituation,
    CompanyResearch,
    CompetitorEvidence,
)
from ..config import get_settings


# Keywords that indicate high-confidence AI/GenAI signals (prioritize LLM/GenAI)
HIGH_CONFIDENCE_KEYWORDS = [
    # GenAI / LLM (highest priority)
    "llm",
    "large language model",
    "generative ai",
    "genai",
    "gpt-4",
    "gpt-3",
    "chatgpt",
    "claude",
    "anthropic",
    "openai api",
    "llama",
    "mistral",
    # RAG and Agents
    "rag",
    "retrieval augmented",
    "ai agent",
    "autonomous agent",
    "langchain",
    "llamaindex",
    "ai assistant",
    "chatbot",
    # LLM Infrastructure
    "prompt engineering",
    "prompt management",
    "llmops",
    "vector database",
    "pinecone",
    "weaviate",
    "chroma",
    "embeddings",
    "semantic search",
    "fine-tuning",
    "fine tuning",
    # Traditional ML (secondary)
    "mlops",
    "ml platform",
    "model deployment",
    "model serving",
    "model monitoring",
    "inference",
]

# Keywords that indicate medium-confidence signals
MEDIUM_CONFIDENCE_KEYWORDS = [
    # GenAI related
    "ai-powered",
    "ai powered",
    "copilot",
    "ai assistant",
    "natural language",
    "nlp",
    "text generation",
    "conversational ai",
    # Traditional AI/ML
    "artificial intelligence",
    "machine learning",
    "data science",
    "predictive",
    "neural network",
    "deep learning",
    "computer vision",
    "recommendation system",
]

# Competitor tools to detect
# Note: Some keywords are "ambiguous" and require AI context to avoid false positives
# High-specificity keywords (prefixed with product-specific terms) are trusted without context
# Generic keywords require AI context validation
COMPETITOR_TOOLS = {
    # LLM Observability & Evaluation (primary competitors)
    "langsmith": [
        "langsmith",
        "langchain smith",
        "langchain tracing",
        "langchain hub",
    ],
    "braintrust": [
        "braintrust ai",
        "braintrust eval",
        "braintrust.dev",
        "braintrust data",
        "braintrust",  # Generic - needs AI context validation
    ],
    "langfuse": [
        "langfuse",
        "langfuse tracing",
    ],
    "helicone": [
        "helicone",
        "helicone.ai",
    ],
    "promptlayer": [
        "promptlayer",
        "prompt layer",
    ],
    "humanloop": [
        "humanloop",
        "human loop",
    ],
    "weights_and_biases": [
        "wandb.ai",
        "wandb experiment",
        "wandb tracking",
        "weights & biases mlops",
        "weights & biases experiment",
        "weights & biases",  # Generic - needs AI context validation
        "weights and biases",  # Generic - needs AI context validation  
        "wandb",  # Generic - needs AI context validation
        "w&b",  # Generic - needs AI context validation
        "weave",  # W&B's new LLM product
    ],
    "datadog": [
        "datadog ml",
        "datadog machine learning",
        "datadog llm observability",
        "datadog llm",
    ],
    "newrelic": [
        "new relic ai",
        "new relic llm",
    ],
    "mlflow": [
        "mlflow",
        "ml flow experiment",
    ],
    "comet": [
        "comet ml",
        "comet.ml",
    ],
    "arize": [
        "arize ai",  # In case they're already using us!
        "arize phoenix",
        "phoenix tracing",
    ],
}

# Keywords that are ambiguous and could refer to non-AI entities
# These require either AI context OR absence of financial context
AMBIGUOUS_KEYWORDS = [
    "braintrust",  # Braintrust Capital (financial firm)
    "weights & biases", "weights and biases", "wandb", "w&b",  # Stock discussions
]

# Financial/investment terms that indicate a false positive for competitor detection
FINANCIAL_NEGATIVE_KEYWORDS = [
    "capital", "stock", "invest", "fund", "securities",
    "portfolio", "equity", "ipo", "shares", "trading",
    "venture", "financing", "valuation", "acquisition",
    "hedge", "asset management", "private equity", "holdings",
    "nasdaq", "nyse", "ticker", "market cap", "dividend",
]

# AI/ML context keywords that validate a competitor mention
AI_CONTEXT_KEYWORDS = [
    # GenAI specific
    "llm", "large language model", "generative ai", "genai",
    "chatgpt", "gpt", "claude", "anthropic", "openai",
    "prompt", "rag", "retrieval", "agent", "chatbot",
    "langchain", "llamaindex", "huggingface", "mistral", "llama",
    # Infrastructure
    "observability", "tracing", "evaluation", "eval",
    "embedding", "vector", "inference", "fine-tune",
    # Traditional ML
    "ai", "ml", "model", "experiment", "dataset",
    "training", "monitoring", "drift", "production model", "mlops",
]


class SignalExtractor:
    """Extracts AI/ML signals and competitive context from research."""

    def __init__(self, llm_model: str | None = None):
        settings = get_settings()
        self.llm_model = llm_model or settings.llm_model
        self.anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _check_keywords(
        self,
        text: str,
        keywords: list[str],
    ) -> list[str]:
        """Check which keywords appear in text."""
        text_lower = text.lower()
        found = []
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword)
        return found

    def _has_financial_context(self, text: str) -> bool:
        """Check if text contains financial/investment context indicating false positive.
        
        Args:
            text: Text to analyze (should be lowercase)
            
        Returns:
            True if financial context is detected
        """
        text_lower = text.lower()
        matches = sum(1 for kw in FINANCIAL_NEGATIVE_KEYWORDS if kw in text_lower)
        # Require at least 1 financial keyword to flag as financial context
        return matches >= 1

    def _has_ai_context(self, text: str) -> bool:
        """Check if text contains AI/ML context validating a competitor mention.
        
        Args:
            text: Text to analyze (should be lowercase)
            
        Returns:
            True if AI/ML context is detected
        """
        text_lower = text.lower()
        matches = sum(1 for kw in AI_CONTEXT_KEYWORDS if kw in text_lower)
        # Require at least 1 AI keyword to validate
        return matches >= 1

    def _is_ambiguous_keyword(self, keyword: str) -> bool:
        """Check if a keyword is ambiguous and needs context validation.
        
        Args:
            keyword: The competitor keyword
            
        Returns:
            True if the keyword is ambiguous
        """
        return keyword.lower() in [k.lower() for k in AMBIGUOUS_KEYWORDS]

    def extract_signals_from_search_results(
        self,
        results: list[SearchResult],
    ) -> list[AIMLSignal]:
        """Extract AI/ML signals from search results.

        Args:
            results: Search results to analyze

        Returns:
            List of detected signals
        """
        signals = []

        for result in results:
            combined_text = f"{result.title} {result.description}"

            # Check for high-confidence keywords
            high_matches = self._check_keywords(combined_text, HIGH_CONFIDENCE_KEYWORDS)
            if high_matches:
                signals.append(
                    AIMLSignal(
                        signal_type="engineering",
                        evidence=f"Found: {', '.join(high_matches)} in '{result.title}'",
                        confidence=SignalConfidence.HIGH,
                        source_url=result.url,
                    )
                )
                continue

            # Check for medium-confidence keywords
            medium_matches = self._check_keywords(
                combined_text, MEDIUM_CONFIDENCE_KEYWORDS
            )
            if medium_matches:
                signals.append(
                    AIMLSignal(
                        signal_type="marketing",
                        evidence=f"Found: {', '.join(medium_matches)} in '{result.title}'",
                        confidence=SignalConfidence.MEDIUM,
                        source_url=result.url,
                    )
                )

        return signals

    def detect_competitors(
        self,
        results: list[SearchResult],
    ) -> tuple[CompetitiveSituation, list[str], list[CompetitorEvidence]]:
        """Detect competitor tools from search results with context-aware filtering.

        Uses multi-layer filtering to avoid false positives:
        1. Checks for competitor keyword matches
        2. For ambiguous keywords (e.g., "braintrust", "wandb"):
           - Skips if financial context is detected (e.g., "Braintrust Capital")
           - Skips if no AI/ML context is present
        3. High-specificity keywords (e.g., "braintrust.dev", "wandb.ai") are trusted

        Args:
            results: Search results to analyze

        Returns:
            Tuple of (competitive situation, list of detected tools, evidence list)
        """
        detected_tools = []
        evidence_list = []
        skipped_evidence = []  # Track what we filtered out for debugging

        for result in results:
            combined_text = f"{result.title} {result.description}".lower()

            for tool_key, keywords in COMPETITOR_TOOLS.items():
                for keyword in keywords:
                    if keyword not in combined_text:
                        continue
                    
                    # Check if this is an ambiguous keyword that needs context validation
                    if self._is_ambiguous_keyword(keyword):
                        # Check for financial context (false positive indicator)
                        if self._has_financial_context(combined_text):
                            skipped_evidence.append(
                                f"Skipped '{keyword}' in '{result.title}' - financial context detected"
                            )
                            continue
                        
                        # For ambiguous keywords, require AI context to validate
                        if not self._has_ai_context(combined_text):
                            skipped_evidence.append(
                                f"Skipped '{keyword}' in '{result.title}' - no AI context"
                            )
                            continue
                    
                    # Valid match - add evidence
                    evidence_list.append(
                        CompetitorEvidence(
                            tool=tool_key,
                            keyword_matched=keyword,
                            source_title=result.title,
                            source_description=result.description[:200] + "..." if len(result.description) > 200 else result.description,
                            source_url=result.url,
                        )
                    )
                    if tool_key not in detected_tools:
                        detected_tools.append(tool_key)

        # Log skipped evidence for debugging (in future could expose this in UI)
        if skipped_evidence:
            print(f"Competitor detection filtered out {len(skipped_evidence)} false positives:")
            for skip in skipped_evidence[:5]:  # Only print first 5
                print(f"  - {skip}")

        if detected_tools:
            return CompetitiveSituation.SWITCHING, detected_tools, evidence_list

        # Check for homegrown signals
        homegrown_keywords = ["grafana", "prometheus", "custom dashboard", "internal"]
        for result in results:
            combined_text = f"{result.title} {result.description}".lower()
            for kw in homegrown_keywords:
                if kw in combined_text:
                    evidence_list.append(
                        CompetitorEvidence(
                            tool="homegrown",
                            keyword_matched=kw,
                            source_title=result.title,
                            source_description=result.description[:200] + "..." if len(result.description) > 200 else result.description,
                            source_url=result.url,
                        )
                    )
                    return CompetitiveSituation.BUILD_VS_BUY, [], evidence_list

        return CompetitiveSituation.UNKNOWN, [], []

    def calculate_overall_confidence(
        self,
        signals: list[AIMLSignal],
    ) -> SignalConfidence:
        """Calculate overall AI/ML confidence from signals.

        Args:
            signals: List of detected signals

        Returns:
            Overall confidence level
        """
        if not signals:
            return SignalConfidence.LOW

        high_count = sum(
            1 for s in signals if s.confidence == SignalConfidence.HIGH
        )
        medium_count = sum(
            1 for s in signals if s.confidence == SignalConfidence.MEDIUM
        )

        if high_count >= 2:
            return SignalConfidence.HIGH
        elif high_count >= 1 or medium_count >= 3:
            return SignalConfidence.MEDIUM
        else:
            return SignalConfidence.LOW

    async def analyze_with_llm(
        self,
        company_name: str,
        search_results: list[SearchResult],
        job_results: list[SearchResult],
        news_results: list[SearchResult],
    ) -> str:
        """Use LLM to generate a company summary from research.

        Args:
            company_name: Company name
            search_results: General AI/ML search results
            job_results: Job posting results
            news_results: News results

        Returns:
            Company summary string
        """
        # Format search results for LLM
        search_text = "\n".join(
            f"- {r.title}: {r.description}" for r in search_results[:10]
        )
        job_text = "\n".join(
            f"- {r.title}: {r.description}" for r in job_results[:5]
        )
        news_text = "\n".join(
            f"- {r.title}: {r.description}" for r in news_results[:5]
        )

        prompt = f"""Based on the following search results about {company_name}, provide a brief 2-3 sentence summary of their AI and GenAI capabilities.

Focus specifically on:
1. LLM/GenAI usage (chatbots, AI assistants, RAG systems, AI agents)
2. AI-powered products or features they've launched
3. Their AI/ML infrastructure and maturity

AI/GenAI Related Search Results:
{search_text}

Job Postings:
{job_text}

Recent News:
{news_text}

Provide a factual summary prioritizing evidence of LLM/GenAI adoption. If there's limited evidence of GenAI work, note what traditional ML or AI they might be using instead. Be specific about what you found."""

        response = self.anthropic.messages.create(
            model=self.llm_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    async def build_company_research(
        self,
        company_name: str,
        ai_ml_results: list[SearchResult],
        competitor_results: dict[str, list[SearchResult]],
        job_results: list[SearchResult],
        news_results: list[SearchResult],
        crm_data: dict | None = None,
    ) -> CompanyResearch:
        """Build complete company research from all sources.

        Args:
            company_name: Company name
            ai_ml_results: AI/ML focused search results
            competitor_results: Competitor detection results
            job_results: Job posting results
            news_results: News results
            crm_data: Optional CRM data from BigQuery

        Returns:
            Complete CompanyResearch object
        """
        # Extract signals
        signals = self.extract_signals_from_search_results(ai_ml_results)
        job_signals = self.extract_signals_from_search_results(job_results)
        signals.extend(job_signals)

        # Detect competitors
        all_competitor_results = []
        for tool, results in competitor_results.items():
            all_competitor_results.extend(results)

        competitive_situation, detected_tools, competitor_evidence = self.detect_competitors(
            all_competitor_results
        )

        # If no competitors detected from search, check if greenfield or unknown
        if competitive_situation == CompetitiveSituation.UNKNOWN:
            # If we have strong AI/ML signals but no tools, likely greenfield
            if self.calculate_overall_confidence(signals) in [
                SignalConfidence.HIGH,
                SignalConfidence.MEDIUM,
            ]:
                competitive_situation = CompetitiveSituation.GREENFIELD

        # Calculate overall confidence
        ai_ml_confidence = self.calculate_overall_confidence(signals)

        # Generate summary with LLM
        summary = await self.analyze_with_llm(
            company_name, ai_ml_results, job_results, news_results
        )

        # Build research object
        research = CompanyResearch(
            company_name=company_name,
            domain=crm_data.get("website") if crm_data else None,
            industry=crm_data.get("industry") if crm_data else None,
            employee_count=crm_data.get("number_of_employees") if crm_data else None,
            ai_ml_signals=signals,
            ai_ml_confidence=ai_ml_confidence,
            competitive_situation=competitive_situation,
            detected_tools=detected_tools,
            competitor_evidence=competitor_evidence,
            exists_in_crm=crm_data is not None,
            company_summary=summary,
        )

        return research
