"""Fetch main text from a URL for two-phase GenAI product extraction.

Uses Jina Reader (r.jina.ai) to get clean article text without parsing HTML.
Falls back to no fetch on failure so phase 1 result is unchanged.
"""

import httpx


async def fetch_page_text(url: str, max_chars: int = 4000, timeout: float = 15.0) -> str | None:
    """Fetch the main text content of a URL.

    Uses Jina Reader (https://jina.ai/reader/) which returns clean article text.
    No API key required for basic usage.

    Args:
        url: Full URL to fetch (e.g. https://example.com/article).
        max_chars: Maximum characters to return from the response body.
        timeout: Request timeout in seconds.

    Returns:
        First max_chars of main text, or None if fetch failed.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None
    # Jina Reader: GET https://r.jina.ai/<target_url>
    reader_url = f"https://r.jina.ai/{url}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                reader_url,
                headers={"X-Return-Format": "text"},
                timeout=timeout,
            )
            response.raise_for_status()
            text = (response.text or "").strip()
            if not text:
                return None
            return text[:max_chars] if len(text) > max_chars else text
    except Exception:
        return None
