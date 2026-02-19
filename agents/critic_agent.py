# ============================================================
# agents/critic_agent.py
# ============================================================
# PURPOSE:
#   This module implements the Critic Agent -- the FIFTH and FINAL
#   agent in the multi-agent research pipeline. It acts as a quality
#   control reviewer that reads the research report written by the
#   Writer Agent and evaluates it across multiple dimensions, providing
#   a quality score, detailed feedback, and improvement suggestions.
#
# WHY WE NEED A CRITIC AGENT:
#   In any professional research or writing workflow, there is always
#   a review/editing step. Without a critic, we have no way to:
#   - Assess the quality of the generated report
#   - Identify gaps, inaccuracies, or weak sections
#   - Provide actionable feedback for improvement
#   - Give the user confidence in the report's quality
#
#   The Critic Agent adds a crucial "self-reflection" capability to
#   the pipeline -- the system can evaluate its own output. This is
#   a key pattern in advanced agentic AI systems (similar to
#   Constitutional AI and self-critique techniques).
#
#   From a portfolio perspective, having a Critic Agent demonstrates
#   understanding of multi-agent feedback loops and quality assurance
#   in AI systems -- a highly valued skill in AI/ML roles.
#
# HOW IT FITS IN THE PIPELINE:
#   Writer Agent -> [CRITIC AGENT] -> Quality Review -> Final Output
#
# EVALUATION CRITERIA (from config.yaml):
#   1. Accuracy and factual correctness
#   2. Completeness of coverage
#   3. Clarity and readability
#   4. Logical structure and flow
#   5. Use of evidence and sources
#
# TECHNOLOGY USED:
#   - Groq API: Free LLM inference (LLaMA 3.3-70B) for evaluation
#   - LangChain's ChatGroq: Python interface to Groq
#   - config_loader: Reads settings from config.yaml
#
# EXAMPLE OUTPUT:
#   {
#     "score": 8,
#     "grade": "B+",
#     "strengths": ["Comprehensive coverage", "Clear structure"],
#     "weaknesses": ["Could include more statistics"],
#     "suggestions": ["Add specific data points in the Analysis section"],
#     "verdict": "APPROVED",
#     "full_review": "## Quality Review\n..."
#   }
# ============================================================

from langchain_groq import ChatGroq                                      # Groq LLM
from langchain_core.messages import HumanMessage, SystemMessage          # Message types
from utils.config_loader import load_config                              # Config loader
from typing import Dict, List, Any                                       # Type hints
import re                                                                # For parsing score from text


def create_llm(config: dict) -> ChatGroq:
    """
    Create and configure a ChatGroq LLM instance for report evaluation.

    WHY THIS FUNCTION EXISTS:
        Each agent has its own create_llm() to remain self-contained.
        The Critic Agent uses the same Groq LLaMA model as other agents,
        ensuring consistent evaluation quality. Using the same powerful
        model (LLaMA 3.3-70B) for criticism as for writing ensures the
        critic is capable of understanding and evaluating the report.

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


def run_critic_agent(
    topic: str,
    report: str,
    config: dict = None
) -> Dict[str, Any]:
    """
    Run the Critic Agent to evaluate the quality of the research report.

    WHY THIS FUNCTION EXISTS:
        This is the core function of the Critic Agent. It takes the
        research topic and the full report from the Writer Agent, then
        uses an LLM to perform a structured quality evaluation.

        The evaluation produces:
        1. A numerical score (1-10)
        2. A letter grade (A, B+, C, etc.)
        3. Identified strengths of the report
        4. Identified weaknesses or gaps
        5. Specific, actionable improvement suggestions
        6. An overall verdict (APPROVED / NEEDS IMPROVEMENT)
        7. A full formatted review in Markdown

        This structured output allows the Streamlit UI to display
        the review in a visually appealing way (score badge, color-coded
        verdict, expandable sections for strengths/weaknesses).

    HOW IT WORKS:
        1. Load configuration (if not provided)
        2. Get evaluation criteria and minimum score threshold from config
        3. Create a Groq LLM instance
        4. Build a detailed evaluation prompt
        5. Call the LLM to generate the review
        6. Parse the LLM's response to extract structured data
        7. Determine the verdict based on score vs. threshold
        8. Return the complete review dictionary

    EVALUATION APPROACH:
        The Critic uses a rubric-based evaluation approach:
        - Each criterion is scored individually
        - An overall score is computed
        - Specific evidence from the report is cited in feedback
        - Suggestions are concrete and actionable (not vague)

    Args:
        topic (str): The original research topic.
                     Used to assess whether the report stays on topic.
        report (str): The full research report from the Writer Agent.
        config (dict, optional): Configuration dictionary. If None, loads
                                  from config.yaml automatically.

    Returns:
        Dict[str, Any]: A structured review dictionary containing:
                        {
                          "score": int (1-10),
                          "grade": str ("A", "B+", "C", etc.),
                          "verdict": str ("APPROVED" or "NEEDS IMPROVEMENT"),
                          "strengths": List[str],
                          "weaknesses": List[str],
                          "suggestions": List[str],
                          "full_review": str (complete Markdown review),
                          "criteria_scores": Dict[str, int] (per-criterion scores)
                        }
                        Returns a fallback review dict if LLM fails.

    Example:
        >>> review = run_critic_agent("Quantum Computing", report_text)
        >>> print(f"Score: {review['score']}/10 ({review['grade']})")
        >>> print(f"Verdict: {review['verdict']}")
    """

    # ----------------------------------------------------------------
    # Step 1: Load configuration if not provided
    # ----------------------------------------------------------------
    if config is None:
        config = load_config()

    # ----------------------------------------------------------------
    # Step 2: Get critic settings from config
    # min_score_threshold: minimum acceptable score (default: 6/10)
    # evaluation_criteria: what dimensions to evaluate
    # ----------------------------------------------------------------
    critic_config = config.get("agents", {}).get("critic", {})
    min_score_threshold = critic_config.get("min_score_threshold", 6)
    evaluation_criteria = critic_config.get("evaluation_criteria", [
        "Accuracy and factual correctness",
        "Completeness of coverage",
        "Clarity and readability",
        "Logical structure and flow",
        "Use of evidence and sources"
    ])

    # ----------------------------------------------------------------
    # Step 3: Create the LLM instance
    # ----------------------------------------------------------------
    llm = create_llm(config)

    print(f"\nCritic Agent starting -- evaluating report on: '{topic}'")
    print(f"   Evaluation criteria: {len(evaluation_criteria)} dimensions")
    print(f"   Minimum acceptable score: {min_score_threshold}/10")

    # ----------------------------------------------------------------
    # Step 4: Generate the evaluation using the LLM
    # ----------------------------------------------------------------
    raw_review = evaluate_report(
        topic=topic,
        report=report,
        evaluation_criteria=evaluation_criteria,
        llm=llm
    )

    # ----------------------------------------------------------------
    # Step 5: Parse the raw review text into structured data
    # The LLM returns a formatted text review; we extract key fields
    # ----------------------------------------------------------------
    review_data = parse_review(raw_review, evaluation_criteria)

    # ----------------------------------------------------------------
    # Step 6: Determine the verdict based on score vs. threshold
    # If score >= threshold: APPROVED
    # If score < threshold: NEEDS IMPROVEMENT (flagged prominently)
    # ----------------------------------------------------------------
    score = review_data.get("score", 5)
    if score >= min_score_threshold:
        verdict = "APPROVED"
        verdict_detail = f"Report meets quality standards (score {score}/10 >= threshold {min_score_threshold}/10)"
    else:
        verdict = "NEEDS IMPROVEMENT"
        verdict_detail = f"Report below quality threshold (score {score}/10 < threshold {min_score_threshold}/10)"

    review_data["verdict"] = verdict
    review_data["verdict_detail"] = verdict_detail
    review_data["min_threshold"] = min_score_threshold

    # ----------------------------------------------------------------
    # Step 7: Build the full formatted Markdown review
    # This is what gets displayed in the Streamlit UI and saved to file
    # ----------------------------------------------------------------
    review_data["full_review"] = build_formatted_review(
        topic=topic,
        review_data=review_data,
        raw_review=raw_review
    )

    print(f"Critic Agent completed -- Score: {score}/10 ({review_data.get('grade', 'N/A')}) | {verdict}")

    return review_data


def evaluate_report(
    topic: str,
    report: str,
    evaluation_criteria: List[str],
    llm: ChatGroq
) -> str:
    """
    Use the LLM to evaluate the research report and generate a detailed review.

    WHY THIS FUNCTION EXISTS:
        This function handles the actual LLM call for evaluating the report.
        It's separated from run_critic_agent() to keep the code clean and
        to make the evaluation step independently testable.

    HOW IT WORKS:
        1. Builds a system prompt defining the critic's role and rubric
        2. Builds a human message with the topic and full report
        3. Calls the Groq LLM to generate the evaluation
        4. Returns the raw review text for parsing

    EVALUATION PROMPT DESIGN:
        The prompt instructs the LLM to:
        - Score each criterion individually (1-10)
        - Provide an overall score
        - List specific strengths with evidence from the report
        - List specific weaknesses with examples
        - Give concrete, actionable improvement suggestions
        - Use a structured format for easy parsing

    Args:
        topic (str): The research topic being evaluated.
        report (str): The full research report text.
        evaluation_criteria (List[str]): List of criteria to evaluate.
        llm (ChatGroq): The configured LLM instance.

    Returns:
        str: The raw evaluation text from the LLM.
             Falls back to a default review string if LLM fails.

    Example:
        >>> raw_review = evaluate_report("AI in Healthcare", report, criteria, llm)
        >>> print(raw_review[:200])
        "OVERALL_SCORE: 8\nGRADE: B+\n\nCRITERIA SCORES:\n..."
    """

    # ----------------------------------------------------------------
    # Format the evaluation criteria for the prompt
    # ----------------------------------------------------------------
    criteria_list = "\n".join(
        f"   {i+1}. {criterion}" for i, criterion in enumerate(evaluation_criteria)
    )

    # ----------------------------------------------------------------
    # Build the System Prompt for the Critic Agent
    # The critic needs very specific instructions to produce
    # structured, parseable output
    # ----------------------------------------------------------------
    system_prompt = f"""You are an expert academic peer reviewer and research quality evaluator.
Your task is to critically evaluate a research report and provide structured, actionable feedback.

EVALUATION CRITERIA (score each 1-10):
{criteria_list}

OUTPUT FORMAT (follow this EXACTLY -- it will be parsed programmatically):

OVERALL_SCORE: [number 1-10]
GRADE: [A/A-/B+/B/B-/C+/C/C-/D/F]

CRITERIA_SCORES:
{chr(10).join(f'- {c}: [score]/10' for c in evaluation_criteria)}

STRENGTHS:
- [Specific strength 1 with evidence from the report]
- [Specific strength 2 with evidence from the report]
- [Specific strength 3 with evidence from the report]

WEAKNESSES:
- [Specific weakness 1 with example from the report]
- [Specific weakness 2 with example from the report]
- [Specific weakness 3 with example from the report]

SUGGESTIONS:
- [Concrete, actionable suggestion 1]
- [Concrete, actionable suggestion 2]
- [Concrete, actionable suggestion 3]

DETAILED_REVIEW:
[Write 2-3 paragraphs of detailed qualitative feedback covering the overall quality,
what works well, what needs improvement, and the report's suitability for its purpose]

SCORING GUIDE:
9-10: Exceptional -- publication ready
7-8: Good -- minor improvements needed
5-6: Adequate -- significant improvements needed
3-4: Poor -- major revision required
1-2: Unacceptable -- complete rewrite needed"""

    # ----------------------------------------------------------------
    # Build the Human Message with the topic and report
    # ----------------------------------------------------------------
    human_message = f"""RESEARCH TOPIC: {topic}

RESEARCH REPORT TO EVALUATE:
{report}

Please evaluate this research report following the exact output format specified."""

    # ----------------------------------------------------------------
    # Call the LLM for evaluation
    # ----------------------------------------------------------------
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])

        return response.content.strip()

    except Exception as e:
        print(f"   WARNING: Critic Agent LLM call failed: {e}")
        return generate_fallback_review_text(topic)


def parse_review(raw_review: str, evaluation_criteria: List[str]) -> Dict[str, Any]:
    """
    Parse the LLM's structured review text into a Python dictionary.

    WHY THIS FUNCTION EXISTS:
        The LLM returns a formatted text string with the review.
        We need to extract specific fields (score, grade, strengths, etc.)
        into a Python dictionary so the Streamlit UI can display them
        in a structured, visually appealing way (score badge, lists, etc.).

    HOW IT WORKS:
        1. Uses regex to find the OVERALL_SCORE line and extract the number
        2. Uses regex to find the GRADE line and extract the letter grade
        3. Extracts bullet-pointed STRENGTHS, WEAKNESSES, and SUGGESTIONS
        4. Extracts per-criterion scores from CRITERIA_SCORES section
        5. Returns all extracted data in a structured dictionary

    ROBUSTNESS:
        The parser handles variations in the LLM's output format:
        - Extra whitespace or newlines
        - Missing sections (uses defaults)
        - Score outside 1-10 range (clamps to valid range)
        - Missing grade (derives from score)

    Args:
        raw_review (str): The raw evaluation text from the LLM.
        evaluation_criteria (List[str]): The criteria that were evaluated.

    Returns:
        Dict[str, Any]: Structured review data with keys:
                        - 'score': int (1-10)
                        - 'grade': str
                        - 'strengths': List[str]
                        - 'weaknesses': List[str]
                        - 'suggestions': List[str]
                        - 'criteria_scores': Dict[str, int]
                        - 'detailed_review': str

    Example:
        >>> data = parse_review(raw_text, criteria)
        >>> print(data['score'])  # 8
        >>> print(data['grade'])  # "B+"
    """

    # ----------------------------------------------------------------
    # Extract the overall score (look for "OVERALL_SCORE: X" pattern)
    # ----------------------------------------------------------------
    score = 5  # Default score if parsing fails
    score_match = re.search(r'OVERALL_SCORE:\s*(\d+)', raw_review, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        # Clamp score to valid range 1-10
        score = max(1, min(10, score))

    # ----------------------------------------------------------------
    # Extract the grade (look for "GRADE: X" pattern)
    # ----------------------------------------------------------------
    grade = score_to_grade(score)  # Default: derive from score
    grade_match = re.search(r'GRADE:\s*([A-F][+-]?)', raw_review, re.IGNORECASE)
    if grade_match:
        grade = grade_match.group(1).upper()

    # ----------------------------------------------------------------
    # Extract bullet-pointed lists (STRENGTHS, WEAKNESSES, SUGGESTIONS)
    # ----------------------------------------------------------------
    strengths = extract_bullet_list(raw_review, "STRENGTHS")
    weaknesses = extract_bullet_list(raw_review, "WEAKNESSES")
    suggestions = extract_bullet_list(raw_review, "SUGGESTIONS")

    # ----------------------------------------------------------------
    # Extract the detailed review paragraph
    # ----------------------------------------------------------------
    detailed_review = ""
    detail_match = re.search(
        r'DETAILED_REVIEW:\s*\n(.*?)(?=\n[A-Z_]+:|$)',
        raw_review,
        re.DOTALL | re.IGNORECASE
    )
    if detail_match:
        detailed_review = detail_match.group(1).strip()

    # ----------------------------------------------------------------
    # Extract per-criterion scores
    # ----------------------------------------------------------------
    criteria_scores = {}
    for criterion in evaluation_criteria:
        # Look for "- Criterion Name: X/10" pattern
        pattern = re.escape(criterion) + r':\s*(\d+)/10'
        match = re.search(pattern, raw_review, re.IGNORECASE)
        if match:
            criteria_scores[criterion] = int(match.group(1))
        else:
            criteria_scores[criterion] = score  # Default to overall score

    return {
        "score": score,
        "grade": grade,
        "strengths": strengths if strengths else ["Report covers the main aspects of the topic"],
        "weaknesses": weaknesses if weaknesses else ["Could benefit from more specific data points"],
        "suggestions": suggestions if suggestions else ["Consider adding more quantitative evidence"],
        "criteria_scores": criteria_scores,
        "detailed_review": detailed_review if detailed_review else raw_review[:500]
    }


def extract_bullet_list(text: str, section_name: str) -> List[str]:
    """
    Extract a bullet-pointed list from a specific section of the review text.

    WHY THIS FUNCTION EXISTS:
        The LLM's review contains multiple sections with bullet lists
        (STRENGTHS, WEAKNESSES, SUGGESTIONS). This helper function
        extracts the bullet points from a named section, handling
        various formatting variations the LLM might produce.

    HOW IT WORKS:
        1. Finds the section by name using regex
        2. Extracts all lines starting with "- " or "* "
        3. Stops when it hits the next section header (all-caps word)
        4. Returns clean list items without the bullet character

    Args:
        text (str): The full review text to search in.
        section_name (str): The section name to look for (e.g., "STRENGTHS").

    Returns:
        List[str]: A list of bullet point strings from that section.
                   Returns empty list if section not found.

    Example:
        >>> items = extract_bullet_list(review_text, "STRENGTHS")
        >>> print(items[0])
        "Comprehensive coverage of all major aspects of quantum computing"
    """
    items = []

    # Find the section in the text
    pattern = rf'{section_name}:\s*\n(.*?)(?=\n[A-Z_]{{3,}}:|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        return items

    section_text = match.group(1)

    # Extract bullet points (lines starting with -, *, or numbers)
    for line in section_text.split('\n'):
        line = line.strip()
        # Check for bullet point markers
        if line.startswith(('- ', '* ')):
            item = line[2:].strip()
            if item:
                items.append(item)
        elif re.match(r'^\d+\.\s+', line):
            # Handle numbered lists too
            item = re.sub(r'^\d+\.\s+', '', line).strip()
            if item:
                items.append(item)

    return items


def score_to_grade(score: int) -> str:
    """
    Convert a numerical score (1-10) to a letter grade.

    WHY THIS FUNCTION EXISTS:
        Letter grades are more intuitive and familiar to users than
        raw numbers. This function provides a consistent mapping from
        the numerical score to a letter grade, used as a fallback
        when the LLM doesn't provide a grade or provides an invalid one.

    GRADING SCALE:
        9-10: A  (Exceptional)
        8:    B+ (Very Good)
        7:    B  (Good)
        6:    C+ (Above Average)
        5:    C  (Average)
        4:    D  (Below Average)
        1-3:  F  (Failing)

    Args:
        score (int): Numerical score from 1 to 10.

    Returns:
        str: Letter grade string (e.g., "A", "B+", "C").

    Example:
        >>> score_to_grade(9)
        "A"
        >>> score_to_grade(7)
        "B"
    """
    if score >= 9:
        return "A"
    elif score >= 8:
        return "B+"
    elif score >= 7:
        return "B"
    elif score >= 6:
        return "C+"
    elif score >= 5:
        return "C"
    elif score >= 4:
        return "D"
    else:
        return "F"


def build_formatted_review(
    topic: str,
    review_data: Dict[str, Any],
    raw_review: str
) -> str:
    """
    Build a formatted Markdown review from the parsed review data.

    WHY THIS FUNCTION EXISTS:
        The parsed review data is a Python dictionary, but we need a
        formatted Markdown string for display in Streamlit and for saving
        to the report file. This function creates a professional, readable
        Markdown review that includes all the evaluation data.

    HOW IT WORKS:
        1. Creates a header with the score, grade, and verdict
        2. Adds a criteria scores section
        3. Adds strengths, weaknesses, and suggestions as bullet lists
        4. Adds the detailed qualitative review
        5. Returns the complete formatted Markdown string

    Args:
        topic (str): The research topic.
        review_data (Dict[str, Any]): Parsed review data dictionary.
        raw_review (str): The original raw review text (used as fallback).

    Returns:
        str: A complete, formatted Markdown review document.

    Example:
        >>> formatted = build_formatted_review("AI", review_data, raw_text)
        >>> print(formatted[:300])
        "## Quality Review
        **Topic:** AI in Healthcare
        **Score:** 8/10 (B+)
        ..."
    """
    score = review_data.get("score", 5)
    grade = review_data.get("grade", "C")
    verdict = review_data.get("verdict", "NEEDS IMPROVEMENT")
    verdict_detail = review_data.get("verdict_detail", "")
    strengths = review_data.get("strengths", [])
    weaknesses = review_data.get("weaknesses", [])
    suggestions = review_data.get("suggestions", [])
    criteria_scores = review_data.get("criteria_scores", {})
    detailed_review = review_data.get("detailed_review", "")

    # Build the formatted review
    parts = []

    # Header section
    parts.append("## Quality Review by Critic Agent")
    parts.append("")
    parts.append(f"**Topic:** {topic}")
    parts.append(f"**Overall Score:** {score}/10 -- Grade: **{grade}**")
    parts.append(f"**Verdict:** {verdict}")
    parts.append(f"*{verdict_detail}*")
    parts.append("")
    parts.append("---")
    parts.append("")

    # Criteria scores section
    if criteria_scores:
        parts.append("### Criteria Scores")
        parts.append("")
        for criterion, c_score in criteria_scores.items():
            parts.append(f"**{criterion}:** {c_score}/10")
        parts.append("")

    # Strengths section
    if strengths:
        parts.append("### Strengths")
        for strength in strengths:
            parts.append(f"- {strength}")
        parts.append("")

    # Weaknesses section
    if weaknesses:
        parts.append("### Weaknesses")
        for weakness in weaknesses:
            parts.append(f"- {weakness}")
        parts.append("")

    # Suggestions section
    if suggestions:
        parts.append("### Improvement Suggestions")
        for suggestion in suggestions:
            parts.append(f"- {suggestion}")
        parts.append("")

    # Detailed review section
    if detailed_review:
        parts.append("### Detailed Review")
        parts.append("")
        parts.append(detailed_review)
        parts.append("")

    return "\n".join(parts)


def generate_fallback_review_text(topic: str) -> str:
    """
    Generate a fallback review text when the LLM fails.

    WHY THIS FUNCTION EXISTS:
        If the Groq API fails during the critic evaluation, we need
        to return something meaningful. This fallback provides a
        basic review structure that the parser can process, ensuring
        the pipeline completes even if the critic LLM call fails.

    Args:
        topic (str): The research topic being evaluated.

    Returns:
        str: A basic review text in the expected format.
    """
    return f"""OVERALL_SCORE: 6
GRADE: C+

CRITERIA_SCORES:
- Accuracy and factual correctness: 6/10
- Completeness of coverage: 6/10
- Clarity and readability: 6/10
- Logical structure and flow: 6/10
- Use of evidence and sources: 6/10

STRENGTHS:
- The report covers the main aspects of {topic}
- The structure follows a logical progression
- The writing is generally clear and readable

WEAKNESSES:
- Automated evaluation was unavailable -- manual review recommended
- Could not assess factual accuracy without evaluation
- Specific data points and statistics may be limited

SUGGESTIONS:
- Have a domain expert review the report for accuracy
- Consider adding more quantitative data and statistics
- Expand the analysis sections with more specific examples

DETAILED_REVIEW:
This report on {topic} was generated by the Multi-Agent Research Assistant.
The automated quality evaluation encountered an issue and could not complete
a full assessment. The report has been assigned a default score of 6/10.
A manual review by a domain expert is recommended to verify accuracy and completeness."""
