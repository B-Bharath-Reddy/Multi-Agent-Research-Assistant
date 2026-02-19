# ============================================================
# agents/summarizer_agent.py
# ============================================================
# PURPOSE:
#   This module implements the Summarizer Agent -- the THIRD agent
#   in the multi-agent research pipeline. It takes the raw search
#   results (web pages + Wikipedia) gathered by the Search Agent
#   and uses an LLM to produce clean, concise, informative summaries
#   for each sub-topic.
#
# WHY WE NEED A SUMMARIZER AGENT:
#   Raw search results are messy -- they contain ads, navigation text,
#   repeated information, irrelevant content, and inconsistent formatting.
#   The Summarizer Agent acts as an intelligent filter and distiller:
#   it reads all the raw content and extracts only the key facts,
#   insights, and information relevant to each sub-question.
#
#   Without this step, the Writer Agent would receive a massive wall
#   of unstructured text, making it hard to write a coherent report.
#   The Summarizer creates clean, structured summaries that the Writer
#   can easily synthesize into a professional research report.
#
# HOW IT FITS IN THE PIPELINE:
#   Search Agent -> [SUMMARIZER AGENT] -> Clean Summaries -> Writer Agent
#
# TECHNOLOGY USED:
#   - Groq API: Free LLM inference (LLaMA 3.3-70B) for summarization
#   - LangChain's ChatGroq: Python interface to Groq
#   - config_loader: Reads settings from config.yaml
#
# EXAMPLE:
#   Input:  Raw search results for "What is quantum computing?"
#           (3 DuckDuckGo results + 1 Wikipedia article = ~2000 words)
#   Output: A clean 150-word summary covering:
#           - Core definition and key concepts
#           - How it differs from classical computing
#           - Key facts and figures
#           - Source citations
# ============================================================

from langchain_groq import ChatGroq                                      # Groq LLM
from langchain_core.messages import HumanMessage, SystemMessage          # Message types
from utils.config_loader import load_config                              # Config loader
from typing import Dict, Any                                             # Type hints


def create_llm(config: dict) -> ChatGroq:
    """
    Create and configure a ChatGroq LLM instance for summarization.

    WHY THIS FUNCTION EXISTS:
        Each agent file has its own create_llm() helper to keep agents
        self-contained and independently testable. This function creates
        a ChatGroq instance configured with settings from config.yaml.

    WHY GROQ FOR SUMMARIZATION:
        Summarization requires reading long texts and producing concise
        outputs. Groq's LLaMA 3.3-70B is excellent at this task and
        processes text extremely fast (tokens per second is very high),
        making it ideal for summarizing multiple search results quickly.

    Args:
        config (dict): Configuration dictionary from config.yaml.
                       Needs: api.groq_api_key, llm.model,
                       llm.temperature, llm.max_tokens

    Returns:
        ChatGroq: A configured LangChain ChatGroq LLM instance.

    Example:
        >>> config = load_config()
        >>> llm = create_llm(config)
    """
    return ChatGroq(
        api_key=config["api"]["groq_api_key"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )


def run_summarizer_agent(
    search_results: Dict[str, Any],
    config: dict = None
) -> Dict[str, str]:
    """
    Run the Summarizer Agent to create clean summaries from raw search results.

    WHY THIS FUNCTION EXISTS:
        This is the core function of the Summarizer Agent. It processes
        the raw, messy search results from the Search Agent and produces
        clean, concise, informative summaries for each sub-topic.

        The summaries serve as the building blocks for the Writer Agent.
        Instead of the Writer having to process thousands of words of raw
        web content, it receives clean, focused summaries -- making the
        final report much higher quality.

    HOW IT WORKS:
        1. Load configuration (if not provided)
        2. Create a Groq LLM instance
        3. For each sub-question and its search results:
           a. Build a summarization prompt with the raw context
           b. Call the LLM to generate a focused summary
           c. Store the summary keyed by the sub-question
        4. Return the complete summaries dictionary

    SUMMARIZATION STRATEGY:
        We instruct the LLM to:
        - Focus on facts, not opinions
        - Include specific data points and statistics when available
        - Maintain a neutral, academic tone
        - Stay within the word limit (from config)
        - Cite sources when possible
        This produces summaries that are both informative and credible.

    Args:
        search_results (Dict[str, Any]): The complete results dictionary from
                                          run_search_agent(). Keys are sub-questions,
                                          values contain web_results, wikipedia,
                                          and formatted_context.
        config (dict, optional): Configuration dictionary. If None, loads
                                  from config.yaml automatically.

    Returns:
        Dict[str, str]: A dictionary mapping each sub-question to its summary.
                        Structure:
                        {
                          "What is quantum computing?": "Quantum computing is a...",
                          "What are quantum computing applications?": "Quantum computers...",
                          ...
                        }
                        Returns fallback summaries if LLM fails for any question.

    Example:
        >>> summaries = run_summarizer_agent(search_results)
        >>> for question, summary in summaries.items():
        ...     print(f"Q: {question[:50]}")
        ...     print(f"A: {summary[:100]}")
    """

    # ----------------------------------------------------------------
    # Step 1: Load configuration if not provided
    # ----------------------------------------------------------------
    if config is None:
        config = load_config()

    # ----------------------------------------------------------------
    # Step 2: Get summarizer settings from config
    # max_summary_length controls how long each summary can be
    # ----------------------------------------------------------------
    max_summary_length = (
        config.get("agents", {})
              .get("summarizer", {})
              .get("max_summary_length", 150)
    )

    # ----------------------------------------------------------------
    # Step 3: Create the LLM instance
    # ----------------------------------------------------------------
    llm = create_llm(config)

    # ----------------------------------------------------------------
    # Step 4: Initialize the summaries dictionary
    # ----------------------------------------------------------------
    summaries = {}

    total = len(search_results)
    print(f"\nSummarizer Agent starting -- {total} sub-topics to summarize...")

    # ----------------------------------------------------------------
    # Step 5: Process each sub-question's search results
    # ----------------------------------------------------------------
    for i, (question, result_data) in enumerate(search_results.items(), start=1):
        print(f"\n   [{i}/{total}] Summarizing: '{question[:55]}...'" if len(question) > 55
              else f"\n   [{i}/{total}] Summarizing: '{question}'")

        # Get the formatted context string (combined web + Wikipedia text)
        context = result_data.get("formatted_context", "")

        # ----------------------------------------------------------------
        # Step 5a: Handle case where no search results were found
        # If there's no context, we can't summarize -- use a fallback message
        # ----------------------------------------------------------------
        if not context or context.strip() == "":
            print(f"      WARNING: No context available for this question")
            summaries[question] = generate_fallback_summary(question)
            continue

        # ----------------------------------------------------------------
        # Step 5b: Build the summarization prompt
        # The system prompt defines the summarizer's role and constraints
        # ----------------------------------------------------------------
        summary = summarize_single_topic(
            question=question,
            context=context,
            max_words=max_summary_length,
            llm=llm
        )

        summaries[question] = summary
        print(f"      Summary generated ({len(summary.split())} words)")

    print(f"\nSummarizer Agent completed -- {len(summaries)} summaries created")
    return summaries


def summarize_single_topic(
    question: str,
    context: str,
    max_words: int,
    llm: ChatGroq
) -> str:
    """
    Summarize the search results for a single sub-topic using the LLM.

    WHY THIS FUNCTION EXISTS:
        This function handles the actual LLM call for summarizing one
        sub-topic. It's separated from run_summarizer_agent() to keep
        the code clean and to make individual topic summarization
        independently testable.

    HOW IT WORKS:
        1. Crafts a system prompt defining the summarizer's role
        2. Crafts a human message with the question and raw context
        3. Calls the Groq LLM to generate the summary
        4. Returns the clean summary text
        5. Falls back to a simple extraction if LLM fails

    PROMPT DESIGN:
        The system prompt is carefully designed to:
        - Set a clear role: "expert research summarizer"
        - Specify the output format: factual, concise, structured
        - Set a word limit to prevent overly long summaries
        - Instruct the LLM to focus on facts, not opinions
        - Ask for source mentions when available

    Args:
        question (str): The sub-question being summarized.
        context (str): The raw search results text (web + Wikipedia combined).
        max_words (int): Maximum word count for the summary.
        llm (ChatGroq): The configured LLM instance to use.

    Returns:
        str: A clean, concise summary of the search results.
             Falls back to a truncated version of the context if LLM fails.

    Example:
        >>> summary = summarize_single_topic(
        ...     "What is quantum computing?",
        ...     raw_context_text,
        ...     150,
        ...     llm
        ... )
        >>> print(summary[:100])
        "Quantum computing is a type of computation that harnesses quantum mechanical phenomena..."
    """

    # ----------------------------------------------------------------
    # Build the System Prompt for the Summarizer
    # This tells the LLM exactly what role to play and how to respond
    # ----------------------------------------------------------------
    system_prompt = f"""You are an expert research summarizer and academic writer.
Your task is to read raw search results and create a clear, factual, well-structured summary.

INSTRUCTIONS:
1. Write a summary of approximately {max_words} words (do not exceed {max_words + 50} words)
2. Focus ONLY on factual information -- no opinions or speculation
3. Include specific data, statistics, or examples when present in the sources
4. Maintain a neutral, academic tone suitable for a research report
5. Mention key sources or organizations when relevant
6. Structure the summary as flowing prose (not bullet points)
7. Directly address the research question in your summary
8. Do NOT include phrases like "According to the search results" or "Based on the context"
   -- write as if you are stating facts directly

OUTPUT FORMAT:
Write ONLY the summary paragraph(s). No introduction, no conclusion, no meta-commentary."""

    # ----------------------------------------------------------------
    # Build the Human Message with the question and raw context
    # We include both so the LLM knows what question to answer
    # ----------------------------------------------------------------
    human_message = f"""RESEARCH QUESTION TO SUMMARIZE:
{question}

RAW SEARCH RESULTS TO SUMMARIZE:
{context}

Please write a {max_words}-word summary that directly answers the research question above."""

    # ----------------------------------------------------------------
    # Call the LLM and handle any errors gracefully
    # ----------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])

        # Extract and clean the summary text
        summary = response.content.strip()

        # Validate the summary is not empty
        if not summary:
            return generate_fallback_summary(question)

        return summary

    except Exception as e:
        print(f"      WARNING: LLM summarization failed: {e}")
        return generate_fallback_summary(question)


def generate_fallback_summary(question: str) -> str:
    """
    Generate a fallback summary when the LLM fails or no context is available.

    WHY THIS FUNCTION EXISTS:
        If the Groq API fails or returns an empty response, we need to
        provide something meaningful rather than an empty string. This
        fallback ensures the pipeline continues and the Writer Agent
        still has something to work with for every sub-topic.

    Args:
        question (str): The sub-question that couldn't be summarized.

    Returns:
        str: A placeholder summary indicating that information was unavailable.

    Example:
        >>> fallback = generate_fallback_summary("What is quantum computing?")
        >>> print(fallback)
        "Information about 'What is quantum computing?' could not be retrieved..."
    """
    return (
        f"Information about this topic could not be fully retrieved at this time. "
        f"The research question '{question}' requires further investigation. "
        f"Please refer to authoritative sources such as academic journals, "
        f"Wikipedia, or recent news articles for comprehensive information on this topic."
    )


def format_summaries_for_writer(summaries: Dict[str, str]) -> str:
    """
    Format all summaries into a single structured document for the Writer Agent.

    WHY THIS FUNCTION EXISTS:
        The Writer Agent needs all summaries in a single, well-organized
        text document. This function combines all individual summaries
        into a structured format with clear section headers, making it
        easy for the Writer Agent to understand the structure and write
        a coherent report.

    HOW IT WORKS:
        1. Iterates through all question-summary pairs
        2. Formats each as a numbered section with the question as header
        3. Joins everything into a single document string

    Args:
        summaries (Dict[str, str]): Dictionary mapping sub-questions to summaries,
                                     as returned by run_summarizer_agent().

    Returns:
        str: A formatted document with all summaries organized by sub-topic.
             Each section has a clear header (the sub-question) and the
             summary as the body text.

    Example:
        >>> formatted = format_summaries_for_writer(summaries)
        >>> print(formatted[:300])
        "RESEARCH SUMMARIES
        ==================
        SUB-TOPIC 1: What is quantum computing?
        ..."
    """
    if not summaries:
        return "No summaries available."

    # Build the formatted document
    parts = ["RESEARCH SUMMARIES", "=" * 60, ""]

    for i, (question, summary) in enumerate(summaries.items(), start=1):
        # Add section header with the sub-question
        parts.append(f"SUB-TOPIC {i}: {question}")
        parts.append("-" * 40)
        # Add the summary text
        parts.append(summary)
        parts.append("")  # Empty line between sections

    return "\n".join(parts)
