# ============================================================
# agents/writer_agent.py
# ============================================================
# PURPOSE:
#   This module implements the Writer Agent -- the FOURTH agent
#   in the multi-agent research pipeline. It takes all the clean
#   summaries produced by the Summarizer Agent and synthesizes
#   them into a single, comprehensive, well-structured research
#   report in Markdown format.
#
# WHY WE NEED A WRITER AGENT:
#   Having individual summaries per sub-topic is not enough for a
#   professional research report. The summaries need to be:
#   - Woven together into a coherent narrative
#   - Organized into logical sections (Introduction, Analysis, etc.)
#   - Written in a consistent academic style throughout
#   - Formatted professionally with headers, structure, and flow
#
#   The Writer Agent acts like a professional technical writer who
#   takes research notes (summaries) and turns them into a polished,
#   publication-ready document. This is the most visible output of
#   the entire pipeline -- what the user actually reads.
#
# HOW IT FITS IN THE PIPELINE:
#   Summarizer Agent -> [WRITER AGENT] -> Research Report -> Critic Agent
#
# REPORT STRUCTURE (from config.yaml):
#   The report sections are configurable in config.yaml:
#   1. Introduction
#   2. Key Findings
#   3. Detailed Analysis
#   4. Current Trends
#   5. Challenges and Limitations
#   6. Future Outlook
#   7. Conclusion
#
# TECHNOLOGY USED:
#   - Groq API: Free LLM inference (LLaMA 3.3-70B) for report writing
#   - LangChain's ChatGroq: Python interface to Groq
#   - Markdown: Output format for the report (renders beautifully in Streamlit)
#   - config_loader: Reads settings from config.yaml
#
# EXAMPLE:
#   Input:  5 clean summaries about "Quantum Computing"
#   Output: A full Markdown research report with:
#           - Title: "Quantum Computing: A Comprehensive Research Report"
#           - 7 structured sections
#           - ~1500-2000 words
#           - Professional academic tone
# ============================================================

from langchain_groq import ChatGroq                                      # Groq LLM
from langchain_core.messages import HumanMessage, SystemMessage          # Message types
from utils.config_loader import load_config                              # Config loader
from agents.summarizer_agent import format_summaries_for_writer          # Format summaries
from typing import Dict, List                                            # Type hints
import datetime                                                          # For report timestamp


def create_llm(config: dict) -> ChatGroq:
    """
    Create and configure a ChatGroq LLM instance for report writing.

    WHY THIS FUNCTION EXISTS:
        Each agent has its own create_llm() to remain self-contained
        and independently testable. The Writer Agent uses the same
        Groq LLaMA model but may benefit from slightly higher temperature
        for more natural, flowing prose (though we keep it at 0.3 for
        factual accuracy as configured in config.yaml).

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


def run_writer_agent(
    topic: str,
    summaries: Dict[str, str],
    config: dict = None
) -> str:
    """
    Run the Writer Agent to synthesize summaries into a full research report.

    WHY THIS FUNCTION EXISTS:
        This is the core function of the Writer Agent. It takes the
        research topic and all the clean summaries from the Summarizer
        Agent, then uses an LLM to write a comprehensive, well-structured
        research report in Markdown format.

        The Writer Agent is the "synthesis" step -- it doesn't just
        concatenate summaries, it intelligently weaves them together
        into a coherent narrative with proper flow, transitions, and
        structure. This is what transforms raw research into a
        professional document.

    HOW IT WORKS:
        1. Load configuration (if not provided)
        2. Get the report sections from config (Introduction, Key Findings, etc.)
        3. Format all summaries into a structured input document
        4. Build a detailed writing prompt with the topic, summaries, and
           required sections
        5. Call the Groq LLM to write the full report
        6. Add metadata (title, date, word count) to the report
        7. Return the complete Markdown report

    WRITING STRATEGY:
        The LLM is instructed to:
        - Write in Markdown format (headers, bold, etc.)
        - Cover all required sections from config.yaml
        - Synthesize information across all sub-topics
        - Maintain consistent academic tone throughout
        - Include a proper introduction and conclusion
        - Reference the research summaries as source material

    Args:
        topic (str): The original research topic from the user.
                     Example: "Impact of AI on Healthcare"
        summaries (Dict[str, str]): Dictionary of sub-question -> summary pairs
                                     from the Summarizer Agent.
        config (dict, optional): Configuration dictionary. If None, loads
                                  from config.yaml automatically.

    Returns:
        str: A complete research report in Markdown format.
             Includes title, date, all sections, and word count.
             Returns a fallback report if the LLM fails.

    Example:
        >>> report = run_writer_agent("Quantum Computing", summaries)
        >>> print(report[:500])
        "# Quantum Computing: A Comprehensive Research Report
        **Generated:** February 18, 2026
        ...
        ## Introduction
        Quantum computing represents a paradigm shift..."
    """

    # ----------------------------------------------------------------
    # Step 1: Load configuration if not provided
    # ----------------------------------------------------------------
    if config is None:
        config = load_config()

    # ----------------------------------------------------------------
    # Step 2: Get the report sections from config
    # These define the structure of the final report
    # Default sections cover all standard research report components
    # ----------------------------------------------------------------
    report_sections = config.get("agents", {}).get("writer", {}).get(
        "report_sections",
        ["Introduction", "Key Findings", "Detailed Analysis",
         "Current Trends", "Challenges and Limitations",
         "Future Outlook", "Conclusion"]
    )

    # ----------------------------------------------------------------
    # Step 3: Create the LLM instance
    # ----------------------------------------------------------------
    llm = create_llm(config)

    print(f"\nWriter Agent starting -- writing report on: '{topic}'")
    print(f"   Sections to write: {', '.join(report_sections)}")

    # ----------------------------------------------------------------
    # Step 4: Format all summaries into a structured input document
    # format_summaries_for_writer() creates a clean, numbered document
    # that the LLM can easily reference while writing
    # ----------------------------------------------------------------
    formatted_summaries = format_summaries_for_writer(summaries)

    # ----------------------------------------------------------------
    # Step 5: Build the writing prompt
    # ----------------------------------------------------------------
    report = write_full_report(
        topic=topic,
        formatted_summaries=formatted_summaries,
        report_sections=report_sections,
        llm=llm
    )

    # ----------------------------------------------------------------
    # Step 6: Add metadata header to the report
    # This adds the title, generation date, and word count
    # ----------------------------------------------------------------
    report_with_metadata = add_report_metadata(
        report=report,
        topic=topic,
        num_sources=len(summaries)
    )

    word_count = len(report_with_metadata.split())
    print(f"Writer Agent completed -- report generated ({word_count} words)")

    return report_with_metadata


def write_full_report(
    topic: str,
    formatted_summaries: str,
    report_sections: List[str],
    llm: ChatGroq
) -> str:
    """
    Use the LLM to write the full research report from the summaries.

    WHY THIS FUNCTION EXISTS:
        This function handles the actual LLM call for writing the report.
        It's separated from run_writer_agent() to keep the code clean
        and to make the report writing step independently testable.

    HOW IT WORKS:
        1. Builds a detailed system prompt defining the writer's role
        2. Builds a human message with the topic, summaries, and sections
        3. Calls the Groq LLM to generate the complete report
        4. Returns the raw report text
        5. Falls back to a structured fallback report if LLM fails

    PROMPT ENGINEERING:
        The system prompt is carefully crafted to:
        - Define the role: "expert research report writer"
        - Specify Markdown formatting requirements
        - List all required sections explicitly
        - Set quality standards (academic tone, evidence-based)
        - Instruct the LLM to synthesize (not just copy) the summaries

    Args:
        topic (str): The research topic.
        formatted_summaries (str): All summaries formatted as a document.
        report_sections (List[str]): List of section names to include.
        llm (ChatGroq): The configured LLM instance.

    Returns:
        str: The complete research report in Markdown format.
             Falls back to a structured template if LLM fails.

    Example:
        >>> report = write_full_report("AI in Healthcare", summaries_doc, sections, llm)
        >>> print(report[:200])
        "## Introduction\nArtificial intelligence is transforming healthcare..."
    """

    # ----------------------------------------------------------------
    # Format the sections list for the prompt
    # Convert ["Introduction", "Key Findings", ...] to a numbered list
    # ----------------------------------------------------------------
    sections_list = "\n".join(
        f"{i+1}. ## {section}" for i, section in enumerate(report_sections)
    )

    # ----------------------------------------------------------------
    # Build the System Prompt for the Writer Agent
    # This is the most detailed prompt in the pipeline because writing
    # a full report requires the most specific instructions
    # ----------------------------------------------------------------
    system_prompt = f"""You are an expert research report writer and academic author.
Your task is to write a comprehensive, professional research report in Markdown format.

REPORT REQUIREMENTS:
1. Write a COMPLETE report covering ALL of these sections (in order):
{sections_list}

2. FORMATTING RULES:
   - Use ## for main section headers (e.g., ## Introduction)
   - Use **bold** for key terms and important concepts
   - Use proper paragraph breaks between ideas
   - Write in flowing prose -- NOT bullet points (except where lists genuinely help)
   - Each section should be 2-4 paragraphs long

3. CONTENT RULES:
   - Base the report ENTIRELY on the provided research summaries
   - Synthesize information across sub-topics -- don't just copy summaries
   - Maintain a consistent, neutral, academic tone throughout
   - Include specific facts, data, and examples from the summaries
   - Make logical connections between different aspects of the topic
   - The Introduction should provide context and scope
   - The Conclusion should synthesize key insights and implications

4. QUALITY STANDARDS:
   - Professional, publication-ready writing quality
   - Logical flow and smooth transitions between sections
   - Evidence-based claims (reference the research data)
   - Comprehensive coverage of the topic

OUTPUT: Write ONLY the report content starting from ## Introduction.
Do NOT include a title (it will be added separately).
Do NOT add any preamble or meta-commentary."""

    # ----------------------------------------------------------------
    # Build the Human Message with the topic and all summaries
    # ----------------------------------------------------------------
    human_message = f"""RESEARCH TOPIC: {topic}

RESEARCH SUMMARIES (use these as your source material):
{formatted_summaries}

Please write a comprehensive research report on "{topic}" covering all the required sections.
Base your report on the research summaries provided above."""

    # ----------------------------------------------------------------
    # Call the LLM to write the report
    # ----------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])

        report = response.content.strip()

        # Validate the report is not empty
        if not report or len(report) < 100:
            print("   WARNING: LLM returned too short a report, using fallback")
            return generate_fallback_report(topic, formatted_summaries, report_sections)

        return report

    except Exception as e:
        print(f"   WARNING: Writer Agent LLM call failed: {e}")
        return generate_fallback_report(topic, formatted_summaries, report_sections)


def add_report_metadata(report: str, topic: str, num_sources: int) -> str:
    """
    Add a professional metadata header to the research report.

    WHY THIS FUNCTION EXISTS:
        A professional research report should have a clear title,
        generation date, and source count. This function adds these
        metadata elements to the top of the report, making it look
        polished and complete. The metadata is formatted in Markdown
        so it renders beautifully in Streamlit and when saved to file.

    HOW IT WORKS:
        1. Creates a formatted title from the research topic
        2. Adds the current date and time
        3. Adds the number of sub-topics researched
        4. Adds a horizontal rule separator
        5. Prepends all this to the report content

    Args:
        report (str): The raw report content from the LLM.
        topic (str): The research topic (used for the title).
        num_sources (int): Number of sub-topics that were researched.

    Returns:
        str: The complete report with metadata header prepended.

    Example:
        >>> full_report = add_report_metadata(report, "Quantum Computing", 5)
        >>> print(full_report[:200])
        "# Quantum Computing: A Comprehensive Research Report
        **Generated by:** Multi-Agent Research Assistant
        **Date:** February 18, 2026
        ..."
    """
    # Get the current date and time for the report timestamp
    now = datetime.datetime.now()
    date_str = now.strftime("%B %d, %Y")      # e.g., "February 18, 2026"
    time_str = now.strftime("%I:%M %p")        # e.g., "10:30 PM"

    # Build the metadata header in Markdown format
    metadata = f"""# {topic}: A Comprehensive Research Report

---
**Generated by:** Multi-Agent Research Assistant (LangGraph + Groq LLaMA)
**Date:** {date_str} at {time_str}
**Sub-topics Researched:** {num_sources}
**Model:** Groq LLaMA 3.3-70B
**Sources:** DuckDuckGo Web Search + Wikipedia

---

"""

    # Combine metadata with the report content
    return metadata + report


def generate_fallback_report(
    topic: str,
    formatted_summaries: str,
    report_sections: List[str]
) -> str:
    """
    Generate a structured fallback report when the LLM fails.

    WHY THIS FUNCTION EXISTS:
        If the Groq API fails during report writing, we need to provide
        something useful rather than an empty result. This fallback
        creates a basic structured report by directly using the summaries,
        ensuring the user always gets some output even in failure cases.

    HOW IT WORKS:
        Creates a simple Markdown report with section headers and
        includes the formatted summaries as the content. While not
        as polished as the LLM-written report, it's still readable
        and contains all the research information.

    Args:
        topic (str): The research topic.
        formatted_summaries (str): All summaries formatted as a document.
        report_sections (List[str]): List of section names.

    Returns:
        str: A basic structured report using the summaries directly.
    """
    sections_text = "\n\n".join(
        f"## {section}\n\n*Content for this section could not be generated. "
        f"Please refer to the research summaries below.*"
        for section in report_sections
    )

    return f"""## Introduction

This report covers the topic: **{topic}**. The following sections provide
an overview based on research gathered from multiple sources.

{sections_text}

---

## Research Data

The following summaries were gathered during research:

{formatted_summaries}
"""
