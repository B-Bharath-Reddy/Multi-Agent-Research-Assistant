# ============================================================
# tools/wikipedia_tool.py
# ============================================================
# PURPOSE:
#   This module provides Wikipedia search capability used by the
#   Search Agent. Wikipedia is an excellent source of structured,
#   factual, encyclopedic information that complements DuckDuckGo
#   web search results.
#
# WHY WIKIPEDIA:
#   - Completely FREE — no API key required
#   - High-quality, peer-reviewed factual content
#   - Structured summaries perfect for research
#   - Covers virtually every topic
#   - The 'wikipedia' Python library makes it simple to use
#
# WHY WE NEED THIS (in addition to DuckDuckGo):
#   DuckDuckGo gives us current web pages (news, blogs, articles).
#   Wikipedia gives us authoritative, structured background knowledge.
#   Together they provide both depth (Wikipedia) and recency (web).
#   This dual-source approach makes research reports more reliable.
#
# USAGE:
#   from tools.wikipedia_tool import search_wikipedia
#   result = search_wikipedia("Quantum computing", sentences=5)
# ============================================================

import wikipedia                  # Wikipedia Python library (pip install wikipedia)
from typing import Dict, Optional  # Type hints for clarity


def search_wikipedia(query: str, sentences: int = 5) -> Dict[str, str]:
    """
    Search Wikipedia for a topic and return a structured summary.

    WHY THIS FUNCTION EXISTS:
        Wikipedia provides high-quality, encyclopedic summaries that
        are perfect for grounding research in factual background
        knowledge. The Search Agent uses this alongside DuckDuckGo
        to get both authoritative background info and current web info.

    HOW IT WORKS:
        1. Takes a search query
        2. Uses the wikipedia library to find the best matching article
        3. Extracts a summary of the specified number of sentences
        4. Also retrieves the full page URL for citation
        5. Returns everything in a structured dictionary
        6. Handles disambiguation and missing pages gracefully

    DISAMBIGUATION HANDLING:
        Sometimes a query like "Python" could mean the programming
        language OR the snake. Wikipedia raises a DisambiguationError
        in this case. We handle it by trying the first suggested option.

    Args:
        query (str): The topic to search for on Wikipedia.
                     Example: "Quantum computing"
        sentences (int): Number of sentences to include in the summary.
                         More sentences = more detail but longer text.
                         Default: 5 (from config.yaml)

    Returns:
        Dict[str, str]: A dictionary containing:
                        - 'title': The Wikipedia article title
                        - 'summary': The extracted summary text
                        - 'url': The Wikipedia page URL (for citation)
                        - 'source': Always 'wikipedia' (for tracking)

                        Example:
                        {
                          'title': 'Quantum computing',
                          'summary': 'Quantum computing is a type of...',
                          'url': 'https://en.wikipedia.org/wiki/Quantum_computing',
                          'source': 'wikipedia'
                        }

                        Returns error dict if search fails:
                        {
                          'title': 'Not Found',
                          'summary': 'No Wikipedia article found for...',
                          'url': '',
                          'source': 'wikipedia'
                        }

    Example:
        >>> result = search_wikipedia("Artificial Intelligence", sentences=3)
        >>> print(result['summary'])
        'Artificial intelligence (AI) is intelligence demonstrated by machines...'
    """

    try:
        # ----------------------------------------------------------------
        # Step 1: Set Wikipedia language to English
        # This ensures consistent results regardless of system locale
        # ----------------------------------------------------------------
        wikipedia.set_lang("en")

        # ----------------------------------------------------------------
        # Step 2: Search for the query and get the best matching page
        # wikipedia.page() fetches the full page object
        # auto_suggest=True allows Wikipedia to correct typos/variations
        # ----------------------------------------------------------------
        page = wikipedia.page(query, auto_suggest=True)

        # ----------------------------------------------------------------
        # Step 3: Extract a summary of the specified number of sentences
        # wikipedia.summary() is smarter than page.summary — it lets us
        # control exactly how many sentences we want
        # ----------------------------------------------------------------
        summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)

        # ----------------------------------------------------------------
        # Step 4: Return the structured result
        # We include the URL so it can be cited in the research report
        # ----------------------------------------------------------------
        return {
            'title': page.title,          # Official Wikipedia article title
            'summary': summary,            # Extracted summary text
            'url': page.url,              # Full Wikipedia URL for citation
            'source': 'wikipedia'         # Track that this came from Wikipedia
        }

    except wikipedia.exceptions.DisambiguationError as e:
        # ----------------------------------------------------------------
        # DisambiguationError: The query matches multiple Wikipedia articles
        # Example: "Python" could be the language or the snake
        # Solution: Try the first suggested option from the disambiguation list
        # ----------------------------------------------------------------
        try:
            # e.options contains a list of possible article titles
            # We try the first one as it's usually the most relevant
            first_option = e.options[0]
            page = wikipedia.page(first_option, auto_suggest=False)
            summary = wikipedia.summary(first_option, sentences=sentences, auto_suggest=False)

            return {
                'title': page.title,
                'summary': summary,
                'url': page.url,
                'source': 'wikipedia'
            }
        except Exception:
            # If even the first option fails, return a graceful error
            return _wikipedia_error_result(query, "Disambiguation — multiple articles found")

    except wikipedia.exceptions.PageError:
        # ----------------------------------------------------------------
        # PageError: No Wikipedia article found for this exact query
        # This is common for very specific or niche topics
        # ----------------------------------------------------------------
        return _wikipedia_error_result(query, "No Wikipedia article found")

    except Exception as e:
        # ----------------------------------------------------------------
        # Catch-all for any other errors (network issues, timeouts, etc.)
        # We never want a Wikipedia failure to crash the entire pipeline
        # ----------------------------------------------------------------
        print(f"WARNING: Wikipedia search failed for '{query}': {e}")
        return _wikipedia_error_result(query, str(e))


def _wikipedia_error_result(query: str, reason: str) -> Dict[str, str]:
    """
    Create a standardized error result dictionary for failed Wikipedia searches.

    WHY THIS FUNCTION EXISTS:
        When Wikipedia search fails, we need to return something in the
        same format as a successful result so the rest of the pipeline
        doesn't break. This helper creates a consistent error response.

    Args:
        query (str): The original search query that failed.
        reason (str): A brief description of why the search failed.

    Returns:
        Dict[str, str]: An error result dictionary in the same format
                        as a successful Wikipedia result, but with
                        an informative message in the 'summary' field.
    """
    return {
        'title': 'Not Found',
        'summary': f"No Wikipedia information available for '{query}'. Reason: {reason}",
        'url': '',
        'source': 'wikipedia'
    }


def format_wikipedia_result(result: Dict[str, str]) -> str:
    """
    Format a Wikipedia result dictionary into a readable text string.

    WHY THIS FUNCTION EXISTS:
        Similar to format_search_results() in web_search.py, this
        function converts the Wikipedia result dictionary into plain
        text that the LLM (Summarizer Agent) can easily read and
        incorporate into its summaries.

    Args:
        result (Dict[str, str]): A Wikipedia result dictionary as
                                  returned by search_wikipedia().

    Returns:
        str: A formatted string with the Wikipedia content,
             ready to be passed to the LLM as context.

    Example:
        >>> result = {'title': 'AI', 'summary': '...', 'url': 'http://...'}
        >>> print(format_wikipedia_result(result))
        Wikipedia Article: AI
        Content: ...
        Source: http://...
    """

    # Format the Wikipedia result with clear labels
    return (
        f"Wikipedia Article: {result.get('title', 'N/A')}\n"
        f"Content: {result.get('summary', 'N/A')}\n"
        f"Source: {result.get('url', 'N/A')}"
    )
