"""Brave Search API client for web research."""

import asyncio
import httpx
from pydantic import BaseModel

from ..config import get_settings
from ..errors import SearchAPIError


class SearchResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    description: str
    page_age: str | None = None


class BraveSearchClient:
    """Client for Brave Search API."""

    BASE_URL = "https://api.search.brave.com/res/v1"

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.brave_api_key
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        # Track if API has failed (for graceful degradation)
        self.api_available = True
        self.last_error: str | None = None

    async def search(
        self,
        query: str,
        count: int = 10,
        freshness: str | None = None,
    ) -> list[SearchResult]:
        """Perform a web search with retry logic.

        Args:
            query: Search query
            count: Number of results (max 20)
            freshness: Filter by freshness (pd=past day, pw=past week, pm=past month)

        Returns:
            List of search results
            
        Raises:
            SearchAPIError: If the search API fails after retries
        """
        params = {
            "q": query,
            "count": min(count, 20),
        }
        if freshness:
            params["freshness"] = freshness

        last_exception = None
        
        # Manual retry logic (3 attempts)
        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.BASE_URL}/web/search",
                        headers=self.headers,
                        params=params,
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                self.api_available = True
                self.last_error = None
                
                # Parse results
                results = []
                for item in data.get("web", {}).get("results", []):
                    results.append(
                        SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            description=item.get("description", ""),
                            page_age=item.get("page_age"),
                        )
                    )
                return results
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                self.api_available = False
                error_msg = f"HTTP {e.response.status_code}"
                if e.response.status_code == 401:
                    # Don't retry auth errors
                    self.last_error = "Invalid API key"
                    raise SearchAPIError("Brave Search", "Invalid API key - check BRAVE_API_KEY")
                elif e.response.status_code == 429:
                    error_msg = "Rate limit exceeded"
                    self.last_error = error_msg
                    # Wait before retry
                    if attempt < 2:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                elif e.response.status_code >= 500:
                    error_msg = "Service temporarily unavailable"
                    self.last_error = error_msg
                    if attempt < 2:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
                        continue
                        
            except httpx.TimeoutException as e:
                last_exception = e
                self.api_available = False
                self.last_error = "Request timed out"
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                    
            except httpx.RequestError as e:
                last_exception = e
                self.api_available = False
                self.last_error = str(e)
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue

        # All retries exhausted
        if last_exception:
            if isinstance(last_exception, httpx.HTTPStatusError):
                raise SearchAPIError("Brave Search", self.last_error or f"HTTP {last_exception.response.status_code}")
            elif isinstance(last_exception, httpx.TimeoutException):
                raise SearchAPIError("Brave Search", "Request timed out after 30 seconds")
            else:
                raise SearchAPIError("Brave Search", f"Connection error: {str(last_exception)}")
        
        return []

    async def search_company_ai_ml(self, company_name: str) -> list[SearchResult]:
        """Search for AI/ML and GenAI related content about a company.

        Args:
            company_name: Company name to research

        Returns:
            List of relevant search results
        """
        # Reduced to 4 most effective queries for speed
        queries = [
            f'"{company_name}" LLM generative AI chatbot GPT',
            f'"{company_name}" AI agents RAG machine learning',
            f'"{company_name}" artificial intelligence product features',
            f'"{company_name}" AI ML engineering infrastructure',
        ]

        all_results = []
        for query in queries:
            try:
                results = await self.search(query, count=5)
                all_results.extend(results)
                await asyncio.sleep(0.2)  # Reduced delay
            except SearchAPIError:
                continue

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)

        return unique_results

    async def search_company_competitors(
        self,
        company_name: str,
        competitors: list[str] | None = None,
    ) -> dict[str, list[SearchResult]]:
        """Search for competitor tool usage at a company.

        Args:
            company_name: Company name to research
            competitors: List of competitor names to check

        Returns:
            Dict mapping competitor name to search results
        """
        if competitors is None:
            # Prioritize LLM observability competitors, then traditional ML tools
            competitors = [
                # LLM Observability & Evaluation (primary competitors)
                "LangSmith",
                "Braintrust AI",
                "Langfuse",
                "Helicone",
                "PromptLayer",
                # ML Experiment Tracking (secondary)
                "Weights & Biases",
                "MLflow",
                "Comet ML",
                # Infrastructure
                "Datadog LLM",
                "New Relic AI",
            ]

        # Only check top 5 competitors for speed
        results = {}
        for competitor in competitors[:5]:
            try:
                query = f'"{company_name}" "{competitor}"'
                search_results = await self.search(query, count=3)
                if search_results:
                    results[competitor] = search_results
                await asyncio.sleep(0.15)  # Reduced delay
            except SearchAPIError:
                continue

        return results

    async def search_company_news(
        self,
        company_name: str,
        count: int = 5,
    ) -> list[SearchResult]:
        """Search for recent AI/GenAI news about a company.

        Args:
            company_name: Company name
            count: Number of results

        Returns:
            List of news results
        """
        # Single combined query for speed
        query = f'"{company_name}" AI announcement generative AI LLM news 2024 2025'
        
        try:
            results = await self.search(query, count=count * 2, freshness="pm")
            return results
        except SearchAPIError:
            return []

    async def search_company_jobs(
        self,
        company_name: str,
        role_keywords: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for job postings related to AI/ML and GenAI.

        Args:
            company_name: Company name
            role_keywords: Keywords to search for in job titles

        Returns:
            List of job posting results
        """
        if role_keywords is None:
            # Prioritize GenAI roles first
            role_keywords = [
                # GenAI specific roles
                "LLM Engineer",
                "AI Engineer",
                "Prompt Engineer",
                "GenAI",
                # Traditional ML roles
                "ML Engineer",
                "MLOps",
                "Machine Learning Platform",
            ]

        all_results = []
        for keyword in role_keywords[:3]:  # Reduced to 3 for speed
            try:
                query = f'"{company_name}" "{keyword}" job careers hiring'
                results = await self.search(query, count=3)
                all_results.extend(results)
                await asyncio.sleep(0.15)  # Reduced delay
            except SearchAPIError:
                continue

        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)

        return unique_results

    # =========================================================================
    # Specialized Sub-Agent Search Methods for Hypothesis Formation
    # =========================================================================

    async def search_linkedin_team(self, company_name: str) -> list[SearchResult]:
        """Search for LinkedIn information about a company's AI/ML team.
        
        Hypothesis Formation: LinkedIn Research
        - How many people have "ML", "AI", "LLM" in their title?
        - Any recent hires in these roles?
        - What does the team structure suggest about AI maturity?
        
        Args:
            company_name: Company name
            
        Returns:
            List of search results about team/people
        """
        # Reduced to 2 queries for speed
        queries = [
            f'site:linkedin.com "{company_name}" "AI Engineer" OR "ML Engineer" OR "LLM"',
            f'"{company_name}" AI ML team hiring engineering',
        ]
        
        all_results = []
        for query in queries:
            try:
                results = await self.search(query, count=5)
                all_results.extend(results)
                await asyncio.sleep(0.1)
            except SearchAPIError:
                continue
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        return unique_results

    async def search_job_postings_detailed(self, company_name: str) -> list[SearchResult]:
        """Search for detailed job posting information.
        
        Hypothesis Formation: Job Postings
        - What tools/frameworks are mentioned? (LangChain, OpenAI, etc.)
        - What problems are they trying to solve?
        - Production vs research focus?
        
        Args:
            company_name: Company name
            
        Returns:
            List of job posting results with detailed info
        """
        # Reduced to 2 queries for speed
        queries = [
            f'"{company_name}" job "LangChain" OR "OpenAI" OR "LLM" OR "ML platform"',
            f'"{company_name}" hiring "AI" OR "machine learning" infrastructure',
        ]
        
        all_results = []
        for query in queries:
            try:
                results = await self.search(query, count=4)
                all_results.extend(results)
                await asyncio.sleep(0.1)
            except SearchAPIError:
                continue
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        return unique_results

    async def search_website_blog(self, company_name: str, domain: str | None = None) -> list[SearchResult]:
        """Search for company website and engineering blog content.
        
        Hypothesis Formation: Website/Blog
        - Any AI-powered product features mentioned?
        - Technical blog posts about ML/LLM challenges?
        - Press releases about AI initiatives?
        
        Args:
            company_name: Company name
            domain: Optional company domain for site-specific search
            
        Returns:
            List of results from company site/blog
        """
        # Reduced to 2 queries for speed
        queries = [
            f'"{company_name}" engineering blog AI machine learning LLM',
            f'"{company_name}" "AI-powered" product features announcement',
        ]
        
        if domain:
            queries[0] = f'site:{domain} AI LLM "machine learning"'
        
        all_results = []
        for query in queries:
            try:
                results = await self.search(query, count=4)
                all_results.extend(results)
                await asyncio.sleep(0.1)
            except SearchAPIError:
                continue
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        return unique_results

    async def search_news_funding(self, company_name: str) -> list[SearchResult]:
        """Search for news and funding information related to AI initiatives.
        
        Hypothesis Formation: News/Funding
        - Recent AI product launches?
        - Funding announcements mentioning AI?
        - Executive quotes about AI strategy?
        
        Args:
            company_name: Company name
            
        Returns:
            List of news and funding results
        """
        # Reduced to 2 queries for speed
        queries = [
            f'"{company_name}" AI announcement funding 2025 2026',
            f'"{company_name}" partnership OpenAI Anthropic "AI strategy"',
        ]
        
        all_results = []
        for query in queries:
            try:
                results = await self.search(query, count=4, freshness="pm")
                all_results.extend(results)
                await asyncio.sleep(0.1)
            except SearchAPIError:
                continue
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        return unique_results

    async def search_pain_points(self, company_name: str) -> list[SearchResult]:
        """Search for potential pain points the company might have.
        
        Hypothesis Formation: Pain Points
        - Technical challenges mentioned in blogs/talks
        - Scale/performance issues
        - Compliance/security concerns with AI
        
        Args:
            company_name: Company name
            
        Returns:
            List of results indicating potential pain points
        """
        # Reduced to 2 queries for speed
        queries = [
            f'"{company_name}" AI ML challenges scaling production monitoring',
            f'"{company_name}" LLM accuracy cost optimization compliance',
        ]
        
        all_results = []
        for query in queries:
            try:
                results = await self.search(query, count=3)
                all_results.extend(results)
                await asyncio.sleep(0.1)
            except SearchAPIError:
                continue
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        return unique_results
