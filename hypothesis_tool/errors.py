"""Custom exceptions for the AE Hypothesis Tool."""


class ResearchError(Exception):
    """Base exception for research errors."""
    
    def __init__(self, message: str, error_type: str, recoverable: bool = False):
        self.message = message
        self.error_type = error_type
        self.recoverable = recoverable
        super().__init__(message)


class CompanyNotFoundError(ResearchError):
    """Raised when no meaningful information is found about a company."""
    
    def __init__(self, company_name: str, search_attempted: bool = True):
        self.company_name = company_name
        message = (
            f"Could not find meaningful information about '{company_name}'. "
            "This could mean the company name is misspelled, too generic, or the company "
            "has limited online presence. Try adding the company domain or a more specific name."
        )
        super().__init__(message, "company_not_found", recoverable=True)


class AmbiguousCompanyError(ResearchError):
    """Raised when search results seem to reference multiple different entities."""
    
    def __init__(self, company_name: str, possible_matches: list[str] | None = None):
        self.company_name = company_name
        self.possible_matches = possible_matches or []
        matches_text = ""
        if possible_matches:
            matches_text = f" Possible matches: {', '.join(possible_matches)}."
        message = (
            f"The name '{company_name}' appears to match multiple entities.{matches_text} "
            "Please provide the company domain or a more specific name to disambiguate."
        )
        super().__init__(message, "ambiguous_company", recoverable=True)


class SearchAPIError(ResearchError):
    """Raised when the search API fails."""
    
    def __init__(self, provider: str, original_error: str):
        self.provider = provider
        self.original_error = original_error
        message = (
            f"Web search failed ({provider}): {original_error}. "
            "This is typically a temporary issue. Please try again in a few moments."
        )
        super().__init__(message, "search_api_error", recoverable=True)


class LLMError(ResearchError):
    """Raised when the LLM fails to generate hypotheses."""
    
    def __init__(self, stage: str, original_error: str):
        self.stage = stage
        self.original_error = original_error
        message = (
            f"AI analysis failed during {stage}: {original_error}. "
            "This may be due to rate limits or service issues. Please try again."
        )
        super().__init__(message, "llm_error", recoverable=True)


class InsufficientDataError(ResearchError):
    """Raised when there's not enough data to generate meaningful hypotheses."""
    
    def __init__(self, company_name: str, signals_found: int, min_required: int = 2):
        self.company_name = company_name
        self.signals_found = signals_found
        message = (
            f"Found only {signals_found} relevant signal(s) for '{company_name}', "
            f"which is insufficient for reliable hypothesis generation. "
            "The results may be generic. Consider providing more context or the company domain."
        )
        super().__init__(message, "insufficient_data", recoverable=False)
