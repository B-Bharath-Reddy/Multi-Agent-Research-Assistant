# ============================================================
# agents/planner_agent.py
# ============================================================
# PURPOSE:
#   This module implements the Planner Agent -- the FIRST agent
#   in the multi-agent research pipeline. Its sole responsibility
#   is to take a broad research topic from the user and break it
#   down into a set of focused, specific sub-questions.
#
# WHY WE NEED A PLANNER AGENT:
#   When a user asks about a broad topic like "Quantum Computing",
#   a single search query won't cover all the important aspects.
#   The Planner Agent acts like a smart research assistant that
#   thinks: "What are the key angles I need to explore to give
#   a comprehensive answer?" It generates targeted sub-questions
#   that guide the Search Agent to fetch well-rounded information.
#
# HOW IT FITS IN THE PIPELINE:
#   User Topic -> [PLANNER AGENT] -> Sub-questions -> Search Agent
#
# TECHNOLOGY USED:
#   - Groq API: Ultra-fast, free LLM inference using LLaMA 3.3-70B
#   - LangChain's ChatGroq: Clean Python interface to Groq API
#   - config_loader: Reads settings from config.yaml
#
# EXAMPLE:
#   Input:  "Impact of Artificial Intelligence on Healthcare"
#   Output: [
#     "What are the current AI applications in medical diagnosis?",
#     "How is AI improving drug discovery and development?",
#     "What are the ethical concerns of AI in healthcare?",
#     "How does AI impact patient data privacy and security?",
#     "What is the future outlook for AI in personalized medicine?"
#   ]
# ============================================================

from langchain_groq import ChatGroq          # LangChain wrapper for Groq API
from langchain_core.messages import HumanMessage, SystemMessage  # Message types for LLM
from utils.config_loader import load_config  # Load settings from config.yaml
from typing import List                      # Type hint for list return


def create_llm(config: dict) -> ChatGroq:
    """
    Create and configure a ChatGroq LLM instance from config settings.

    WHY THIS FUNCTION EXISTS:
        Every agent needs an LLM instance to generate text. Instead of
        duplicating the LLM setup code in every agent, this helper
        function creates a properly configured ChatGroq instance using
        settings from config.yaml. This ensures all agents use the
        same model and settings consistently.

    HOW IT WORKS:
        1. Reads the Groq API key from config
        2. Reads the model name (e.g., llama-3.3-70b-versatile)
        3. Reads temperature and max_tokens settings
        4. Creates and returns a ChatGroq instance

    WHY GROQ:
        Groq provides FREE, ultra-fast inference for open-source LLMs
        like LLaMA. It's significantly faster than OpenAI and completely
        free for reasonable usage -- perfect for a portfolio project.

    Args:
        config (dict): The configuration dictionary loaded from config.yaml.
                       Must contain 'api.groq_api_key', 'llm.model',
                       'llm.temperature', and 'llm.max_tokens'.

    Returns:
        ChatGroq: A configured LangChain ChatGroq LLM instance ready
                  to accept messages and generate responses.

    Example:
        >>> config = load_config()
        >>> llm = create_llm(config)
        >>> response = llm.invoke([HumanMessage(content="Hello!")])
    """
    return ChatGroq(
        api_key=config["api"]["groq_api_key"],          # Your free Groq API key
        model=config["llm"]["model"],                    # e.g., "llama-3.3-70b-versatile"
        temperature=config["llm"]["temperature"],        # 0.3 = factual, not too creative
        max_tokens=config["llm"]["max_tokens"]           # Max response length (2048 tokens)
    )


def run_planner_agent(topic: str, config: dict = None) -> List[str]:
    """
    Run the Planner Agent to decompose a research topic into sub-questions.

    WHY THIS FUNCTION EXISTS:
        This is the core function of the Planner Agent. It takes a broad
        research topic and uses an LLM to intelligently break it down into
        specific, focused sub-questions. These sub-questions are then used
        by the Search Agent to fetch targeted information from the web.

        Without this decomposition step, we would either:
        a) Search for the entire topic at once (too broad, poor results)
        b) Miss important aspects of the topic (incomplete research)

        The Planner Agent ensures comprehensive, well-rounded research
        by identifying the key dimensions of any topic.

    HOW IT WORKS:
        1. Load configuration (if not provided)
        2. Create a Groq LLM instance
        3. Craft a system prompt that instructs the LLM to act as a
           research planner
        4. Send the user's topic as a human message
        5. Parse the LLM's response to extract the numbered sub-questions
        6. Return a clean list of sub-question strings

    PROMPT STRATEGY:
        We use a System + Human message pattern:
        - System message: Sets the LLM's role and output format
        - Human message: Provides the actual research topic
        This is more reliable than a single prompt because the system
        message establishes clear behavioral constraints.

    Args:
        topic (str): The broad research topic to decompose.
                     Example: "Impact of AI on Healthcare"
        config (dict, optional): Configuration dictionary. If None,
                                  loads from config.yaml automatically.
                                  Useful for testing with custom configs.

    Returns:
        List[str]: A list of focused sub-questions derived from the topic.
                   Length is controlled by config['agents']['planner']['num_subquestions'].
                   Example:
                   [
                     "What are the current AI applications in medical diagnosis?",
                     "How is AI improving drug discovery?",
                     ...
                   ]
                   Returns a fallback list with the original topic if LLM fails.

    Raises:
        Does NOT raise exceptions -- returns fallback questions on any error.
        This ensures the pipeline continues even if the planner fails.

    Example:
        >>> questions = run_planner_agent("Quantum Computing")
        >>> print(questions[0])
        "What is quantum computing and how does it differ from classical computing?"
    """

    # ----------------------------------------------------------------
    # Step 1: Load configuration if not provided
    # This allows the function to be called standalone (for testing)
    # or as part of the pipeline (config passed in for efficiency)
    # ----------------------------------------------------------------
    if config is None:
        config = load_config()

    # ----------------------------------------------------------------
    # Step 2: Get the number of sub-questions from config
    # Default to 5 if not specified -- 5 is a good balance between
    # comprehensiveness and speed
    # ----------------------------------------------------------------
    num_questions = config.get("agents", {}).get("planner", {}).get("num_subquestions", 5)

    # ----------------------------------------------------------------
    # Step 3: Create the LLM instance
    # ----------------------------------------------------------------
    llm = create_llm(config)

    # ----------------------------------------------------------------
    # Step 4: Craft the System Prompt
    # The system prompt defines the LLM's role and output format.
    # Being very specific about the format (numbered list) makes
    # parsing the response much easier and more reliable.
    # ----------------------------------------------------------------
    system_prompt = f"""You are an expert research planner and academic strategist.
Your job is to analyze a broad research topic and break it down into {num_questions}
specific, focused sub-questions that together provide comprehensive coverage of the topic.

RULES:
1. Generate EXACTLY {num_questions} sub-questions
2. Each sub-question must be specific and searchable (not vague)
3. Cover different aspects: definitions, applications, challenges, trends, future
4. Number each question (1. 2. 3. etc.)
5. Each question on its own line
6. Do NOT add any introduction or conclusion text -- ONLY the numbered questions

EXAMPLE FORMAT:
1. What is [topic] and how does it work fundamentally?
2. What are the main applications of [topic] in industry?
3. What are the key challenges and limitations of [topic]?
4. What are the latest trends and developments in [topic]?
5. What is the future outlook and potential impact of [topic]?"""

    # ----------------------------------------------------------------
    # Step 5: Create the Human Message with the user's topic
    # ----------------------------------------------------------------
    human_message = f"Research Topic: {topic}"

    # ----------------------------------------------------------------
    # Step 6: Call the LLM with both system and human messages
    # The LLM will respond with a numbered list of sub-questions
    # ----------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])

        # Extract the text content from the LLM response object
        response_text = response.content

        # ----------------------------------------------------------------
        # Step 7: Parse the response to extract individual sub-questions
        # The LLM returns a numbered list like:
        #   "1. What is quantum computing?\n2. How does it work?\n..."
        # We split by newlines and clean up each line
        # ----------------------------------------------------------------
        sub_questions = parse_subquestions(response_text, num_questions)

        # ----------------------------------------------------------------
        # Step 8: Validate we got enough questions
        # If parsing failed or returned too few questions, use fallback
        # ----------------------------------------------------------------
        if len(sub_questions) < 2:
            print(f"WARNING: Planner got too few questions ({len(sub_questions)}), using fallback")
            return generate_fallback_questions(topic, num_questions)

        print(f"OK: Planner Agent generated {len(sub_questions)} sub-questions for: '{topic}'")
        return sub_questions

    except Exception as e:
        # ----------------------------------------------------------------
        # If the LLM call fails for any reason (network, API limit, etc.)
        # we return fallback questions so the pipeline can continue.
        # This makes the system resilient to temporary failures.
        # ----------------------------------------------------------------
        print(f"WARNING: Planner Agent LLM call failed: {e}")
        print("   Using fallback sub-questions...")
        return generate_fallback_questions(topic, num_questions)


def parse_subquestions(response_text: str, expected_count: int) -> List[str]:
    """
    Parse the LLM's numbered list response into a clean Python list.

    WHY THIS FUNCTION EXISTS:
        The LLM returns a raw text string with numbered questions.
        We need to convert this into a Python list of clean strings
        that the Search Agent can iterate over. This function handles
        various formatting variations the LLM might produce.

    HOW IT WORKS:
        1. Split the response text by newlines
        2. For each line, check if it starts with a number (1., 2., etc.)
        3. Strip the number prefix and clean whitespace
        4. Collect valid questions into a list

    HANDLES THESE FORMATS:
        - "1. Question text here"
        - "1) Question text here"
        - "1: Question text here"
        - Lines with extra whitespace

    Args:
        response_text (str): The raw text response from the LLM.
        expected_count (int): How many questions we expected (for validation).

    Returns:
        List[str]: A list of clean sub-question strings without numbering.
                   May be shorter than expected_count if parsing fails.

    Example:
        >>> text = "1. What is AI?\\n2. How does AI work?\\n3. What are AI applications?"
        >>> parse_subquestions(text, 3)
        ["What is AI?", "How does AI work?", "What are AI applications?"]
    """
    questions = []

    # Split the response into individual lines
    lines = response_text.strip().split('\n')

    for line in lines:
        # Remove leading/trailing whitespace from each line
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Check if the line starts with a number followed by . ) or :
        # This identifies it as a numbered list item
        if line and line[0].isdigit():
            # Remove the number prefix (handles "1.", "1)", "1:", "10.", etc.)
            # Find where the actual question text starts
            for i, char in enumerate(line):
                if char in '.):' and i > 0:
                    # Extract everything after the separator and clean it
                    question = line[i+1:].strip()
                    if question:  # Only add non-empty questions
                        questions.append(question)
                    break
            else:
                # No separator found -- the whole line might be the question
                # (rare edge case)
                if line[0].isdigit():
                    questions.append(line)

    return questions


def generate_fallback_questions(topic: str, num_questions: int) -> List[str]:
    """
    Generate generic fallback sub-questions when the LLM fails.

    WHY THIS FUNCTION EXISTS:
        If the Groq API is unavailable or returns an unexpected response,
        we need a way to continue the pipeline. These generic questions
        work for virtually any research topic and ensure the system
        degrades gracefully rather than crashing completely.

    HOW IT WORKS:
        Uses a template of universal research questions and fills in
        the topic name. These cover the standard research dimensions:
        definition, applications, challenges, trends, and future.

    Args:
        topic (str): The research topic to insert into the templates.
        num_questions (int): How many fallback questions to generate.

    Returns:
        List[str]: A list of generic but useful sub-questions about the topic.

    Example:
        >>> generate_fallback_questions("Blockchain", 3)
        [
          "What is Blockchain and how does it work fundamentally?",
          "What are the main real-world applications of Blockchain?",
          "What are the key challenges and limitations of Blockchain?"
        ]
    """
    # Template questions that work for any research topic
    all_fallback_questions = [
        f"What is {topic} and how does it work fundamentally?",
        f"What are the main real-world applications of {topic}?",
        f"What are the key challenges and limitations of {topic}?",
        f"What are the latest trends and recent developments in {topic}?",
        f"What is the future outlook and long-term impact of {topic}?",
        f"Who are the key players and organizations involved in {topic}?",
        f"What are the ethical considerations and societal implications of {topic}?",
    ]

    # Return only as many as requested
    return all_fallback_questions[:num_questions]
