# ============================================================
# app.py
# ============================================================
# PURPOSE:
#   This is the main entry point for the Multi-Agent Research
#   Assistant web application. It provides an interactive
#   Streamlit UI that allows users to:
#   - Enter a research topic
#   - Watch the 5 agents work in real-time (with progress indicators)
#   - View the generated research report
#   - See the Critic Agent's quality review and score
#   - Download the report as a Markdown file
#   - View intermediate results (sub-questions, search results, summaries)
#
# WHY STREAMLIT:
#   Streamlit is the fastest way to build a professional web UI for
#   Python AI/ML projects. It requires zero HTML/CSS/JavaScript knowledge,
#   renders Markdown beautifully, and is widely used in the AI/ML industry.
#   It's completely free and can be deployed to Streamlit Cloud for free.
#
# HOW IT WORKS:
#   1. User enters a research topic in the text input
#   2. User clicks "Start Research"
#   3. The app calls run_research_pipeline() from graph/research_graph.py
#   4. The pipeline runs all 5 agents sequentially
#   5. Results are displayed in organized tabs and expandable sections
#   6. User can download the final report as a .md file
#
# HOW TO RUN:
#   streamlit run app.py
#
# STREAMLIT KEY CONCEPTS USED:
#   - st.title(), st.header(): Text display
#   - st.text_input(): User input field
#   - st.button(): Clickable button
#   - st.spinner(): Loading indicator
#   - st.tabs(): Tabbed content sections
#   - st.expander(): Collapsible sections
#   - st.markdown(): Render Markdown text
#   - st.metric(): Display key numbers (score, word count)
#   - st.success/warning/error(): Colored status messages
#   - st.download_button(): File download button
#   - st.sidebar: Side panel for settings/info
#   - st.session_state: Persist data between reruns
# ============================================================

import streamlit as st                                    # Streamlit web framework
from graph.research_graph import run_research_pipeline    # Main pipeline runner
from utils.config_loader import load_config               # Config loader
import datetime                                           # For timestamps


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# st.set_page_config() MUST be the first Streamlit call in the script.
# It sets the browser tab title, icon, and layout.
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",    # Browser tab title
    page_icon="R",                                   # Browser tab icon
    layout="wide",                                   # Use full browser width
    initial_sidebar_state="expanded"                 # Show sidebar by default
)


# ============================================================
# CUSTOM CSS STYLING
# ============================================================
# Inject custom CSS to make the UI look more polished.
# Streamlit allows injecting raw HTML/CSS via st.markdown().
# ============================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_app_config():
    """
    Load the application configuration from config.yaml.

    WHY THIS FUNCTION EXISTS:
        The app needs the configuration for two purposes:
        1. To display app settings (title, description, example topics)
        2. To pass to the research pipeline when running

        We try to load the config once and return None if the API key
        is not yet configured, so the UI can show a helpful error message.

    Returns:
        dict: The configuration dictionary, or None if loading fails.
              Returns None if the API key has not been set yet.
    """
    try:
        return load_config()
    except ValueError:
        # ValueError is raised when API key is not set
        return None
    except Exception as e:
        st.error(f"Failed to load config.yaml: {e}")
        return None


def display_agent_progress(agent_num: int, agent_name: str, status: str):
    """
    Display a visual progress indicator for an agent.

    WHY THIS FUNCTION EXISTS:
        Users want to see which agent is currently running and which
        have completed. This function creates a consistent, visually
        appealing status card for each agent in the pipeline.

    Args:
        agent_num (int): The agent's position in the pipeline (1-5).
        agent_name (str): The agent's display name.
        status (str): Current status message.
    """
    if "OK" in status or "complete" in status.lower() or "generated" in status.lower():
        color = "#2ca02c"
        label = "DONE"
    elif "failed" in status.lower() or "error" in status.lower():
        color = "#d62728"
        label = "FAIL"
    else:
        color = "#aaa"
        label = "WAIT"

    st.markdown(
        f'<div style="background:#f8f9fa; border-left:4px solid {color}; '
        f'padding:0.5rem 1rem; border-radius:5px; margin:0.3rem 0;">'
        f'<b>[{label}] Agent {agent_num}/5: {agent_name}</b><br>'
        f'<small style="color:#666">{status}</small>'
        f'</div>',
        unsafe_allow_html=True
    )


def get_score_color(score: int) -> str:
    """
    Get a color code based on the quality score.

    WHY THIS FUNCTION EXISTS:
        Color-coding the score makes it immediately clear whether the
        report is high quality (green), acceptable (orange), or poor (red).
        This is a common UX pattern for quality indicators.

    Args:
        score (int): Quality score from 1 to 10.

    Returns:
        str: A hex color code string.
    """
    if score >= 8:
        return "#2ca02c"    # Green -- excellent
    elif score >= 6:
        return "#ff7f0e"    # Orange -- acceptable
    elif score >= 4:
        return "#d62728"    # Red -- needs improvement
    else:
        return "#7f7f7f"    # Gray -- very poor


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """
    Render the sidebar with app information and settings.

    WHY THIS FUNCTION EXISTS:
        The sidebar provides a persistent panel for:
        - App description and how-to-use instructions
        - Example research topics (clickable)
        - Tech stack information
        - Links to get a free Groq API key

        Separating sidebar rendering into its own function keeps
        the main app code clean and organized.

    Returns:
        str or None: The example topic selected by the user (if any),
                     or None if no example was clicked.
    """
    with st.sidebar:
        st.markdown("## Multi-Agent Research Assistant")
        st.markdown("*Powered by LangGraph + Groq LLaMA*")
        st.divider()

        st.markdown("### How It Works")
        st.markdown("""
        1. **Enter** a research topic
        2. **Click** "Start Research"
        3. **5 AI agents** work in sequence:
           - **Planner** -- Breaks topic into sub-questions
           - **Search** -- Fetches web + Wikipedia data
           - **Summarizer** -- Distills key information
           - **Writer** -- Writes the full report
           - **Critic** -- Reviews and scores the report
        4. **Download** your research report
        """)

        st.divider()

        st.markdown("### Example Topics")
        st.markdown("*Click to use as your research topic:*")

        example_topics = [
            "Impact of Artificial Intelligence on Healthcare",
            "Quantum Computing: Current State and Future",
            "Climate Change Solutions and Technologies",
            "Blockchain Technology in Finance",
            "The Future of Electric Vehicles"
        ]

        selected_example = None
        for topic in example_topics:
            if st.button(topic, key=f"example_{topic}", use_container_width=True):
                selected_example = topic

        st.divider()

        st.markdown("### Tech Stack")
        st.markdown("""
        | Component | Technology |
        |-----------|-----------|
        | LLM | Groq LLaMA 3.3-70B |
        | Agents | LangGraph |
        | Search | DuckDuckGo |
        | Knowledge | Wikipedia |
        | UI | Streamlit |
        """)

        st.divider()

        st.markdown("### API Key")
        st.markdown("""
        Get your **free** Groq API key at:
        [console.groq.com](https://console.groq.com)

        Then add it to `config.yaml`:
        ```yaml
        api:
          groq_api_key: "gsk_..."
        ```
        """)

        return selected_example


# ============================================================
# RESULTS DISPLAY FUNCTIONS
# ============================================================

def display_results(result: dict):
    """
    Display the complete pipeline results in a tabbed interface.

    WHY THIS FUNCTION EXISTS:
        The pipeline produces a lot of data (sub-questions, search results,
        summaries, report, review). Displaying all of this in a flat layout
        would be overwhelming. This function organizes everything into
        logical tabs so users can navigate to what they care about.

    TABS:
        1. Research Report -- The main output (full Markdown report)
        2. Quality Review -- Critic's score, grade, and feedback
        3. Research Process -- Sub-questions, search results, summaries
        4. Download -- Download the report as a file

    Args:
        result (dict): The final pipeline state from run_research_pipeline().
    """
    topic = result.get("topic", "Unknown Topic")
    report = result.get("report", "")
    review = result.get("review", {})
    sub_questions = result.get("sub_questions", [])
    summaries = result.get("summaries", {})
    search_results = result.get("search_results", {})

    # ----------------------------------------------------------------
    # Summary metrics row at the top
    # ----------------------------------------------------------------
    st.markdown("### Research Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Report Words",
            value=f"{len(report.split()):,}",
            help="Total word count of the generated research report"
        )
    with col2:
        st.metric(
            label="Sub-Questions",
            value=len(sub_questions),
            help="Number of focused sub-questions researched"
        )
    with col3:
        score = review.get("score", 0)
        st.metric(
            label="Quality Score",
            value=f"{score}/10",
            help="Quality score assigned by the Critic Agent"
        )
    with col4:
        grade = review.get("grade", "N/A")
        st.metric(
            label="Grade",
            value=grade,
            help="Letter grade based on quality score"
        )

    st.divider()

    # ----------------------------------------------------------------
    # Create tabs for organized content display
    # ----------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Research Report",
        "Quality Review",
        "Research Process",
        "Download"
    ])

    # ----------------------------------------------------------------
    # TAB 1: Research Report
    # ----------------------------------------------------------------
    with tab1:
        st.markdown("### Generated Research Report")
        st.info(
            f"**Topic:** {topic} | "
            f"**Words:** {len(report.split()):,} | "
            f"**Generated:** {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
        )
        st.markdown(report)

    # ----------------------------------------------------------------
    # TAB 2: Quality Review
    # ----------------------------------------------------------------
    with tab2:
        st.markdown("### Critic Agent Quality Review")

        if review:
            score = review.get("score", 0)
            grade = review.get("grade", "N/A")
            verdict = review.get("verdict", "Unknown")
            color = get_score_color(score)

            col_score, col_verdict = st.columns([1, 2])
            with col_score:
                st.markdown(
                    f'<div style="text-align:center; background:{color}20; '
                    f'border:3px solid {color}; border-radius:15px; padding:1.5rem;">'
                    f'<div style="font-size:3rem; font-weight:bold; color:{color}">{score}/10</div>'
                    f'<div style="font-size:1.5rem; color:{color}">{grade}</div>'
                    f'<div style="font-size:0.9rem; color:#666">Quality Score</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col_verdict:
                if "APPROVED" in verdict:
                    st.success(f"**Verdict:** {verdict}")
                else:
                    st.warning(f"**Verdict:** {verdict}")

                criteria_scores = review.get("criteria_scores", {})
                if criteria_scores:
                    st.markdown("**Criteria Breakdown:**")
                    for criterion, c_score in criteria_scores.items():
                        progress_val = c_score / 10
                        st.markdown(f"*{criterion}:* **{c_score}/10**")
                        st.progress(progress_val)

            st.divider()

            col_s, col_w = st.columns(2)

            with col_s:
                strengths = review.get("strengths", [])
                if strengths:
                    st.markdown("### Strengths")
                    for s in strengths:
                        st.markdown(f"- {s}")

            with col_w:
                weaknesses = review.get("weaknesses", [])
                if weaknesses:
                    st.markdown("### Weaknesses")
                    for w in weaknesses:
                        st.markdown(f"- {w}")

            suggestions = review.get("suggestions", [])
            if suggestions:
                st.markdown("### Improvement Suggestions")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")

            full_review = review.get("full_review", "")
            if full_review:
                with st.expander("View Full Detailed Review"):
                    st.markdown(full_review)
        else:
            st.warning("No review data available.")

    # ----------------------------------------------------------------
    # TAB 3: Research Process
    # ----------------------------------------------------------------
    with tab3:
        st.markdown("### Research Process Details")
        st.markdown("*See how the agents worked step by step*")

        if sub_questions:
            with st.expander(f"Planner Agent -- {len(sub_questions)} Sub-Questions Generated", expanded=True):
                for i, q in enumerate(sub_questions, 1):
                    st.markdown(f"**{i}.** {q}")

        if summaries:
            with st.expander(f"Summarizer Agent -- {len(summaries)} Topic Summaries"):
                for i, (question, summary) in enumerate(summaries.items(), 1):
                    st.markdown(f"**Sub-topic {i}:** *{question}*")
                    st.markdown(summary)
                    if i < len(summaries):
                        st.divider()

        if search_results:
            with st.expander("Search Agent -- Raw Search Results"):
                for question, result_data in search_results.items():
                    st.markdown(f"**Question:** *{question}*")
                    web_results = result_data.get("web_results", [])
                    wiki = result_data.get("wikipedia", {})

                    if web_results:
                        st.markdown(f"*Web results: {len(web_results)}*")
                        for r in web_results:
                            st.markdown(f"  - [{r.get('title', 'N/A')}]({r.get('url', '#')})")

                    if wiki and wiki.get("title") != "Not Found":
                        st.markdown(f"*Wikipedia: [{wiki.get('title', 'N/A')}]({wiki.get('url', '#')})*")

                    st.divider()

    # ----------------------------------------------------------------
    # TAB 4: Download
    # ----------------------------------------------------------------
    with tab4:
        st.markdown("### Download Your Research Report")
        st.markdown("Download the complete research report including the quality review.")

        full_review_text = review.get("full_review", "")
        download_content = report
        if full_review_text:
            download_content += "\n\n---\n\n" + full_review_text

        clean_topic = "".join(
            c if c.isalnum() or c == ' ' else '_'
            for c in topic
        ).replace(' ', '_').lower()[:40]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{clean_topic}_{timestamp}.md"

        st.download_button(
            label="Download Report (.md)",
            data=download_content,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
            help="Download the research report as a Markdown file"
        )

        st.markdown("---")
        st.markdown("**File format:** Markdown (.md)")
        st.markdown("**Contents:** Full research report + Critic Agent review")
        st.markdown("**Tip:** Open with any Markdown viewer, VS Code, or GitHub for best rendering")


# ============================================================
# MAIN APP
# ============================================================

def main():
    """
    Main function that renders the complete Streamlit application.

    WHY THIS FUNCTION EXISTS:
        Organizing the entire app in a main() function is a Python
        best practice. It makes the code cleaner, easier to test,
        and prevents code from running when the module is imported.

    HOW IT WORKS:
        1. Render the sidebar (returns selected example topic if clicked)
        2. Display the main title and description
        3. Show the API key warning if not configured
        4. Render the topic input and research button
        5. When research is triggered, run the pipeline and display results
        6. Use st.session_state to persist results between reruns

    STREAMLIT SESSION STATE:
        Streamlit reruns the entire script on every user interaction.
        st.session_state is a dictionary that persists between reruns.
        We use it to store the research results so they don't disappear
        when the user clicks a tab or interacts with the UI.
    """

    # ----------------------------------------------------------------
    # Render sidebar and get any selected example topic
    # ----------------------------------------------------------------
    selected_example = render_sidebar()

    # ----------------------------------------------------------------
    # Main title and description
    # ----------------------------------------------------------------
    st.markdown('<div class="main-title">Multi-Agent Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Autonomous AI research powered by <b>LangGraph</b> + <b>Groq LLaMA 3.3-70B</b> | '
        'Free - Fast - Comprehensive</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ----------------------------------------------------------------
    # Load configuration and check API key
    # ----------------------------------------------------------------
    config = load_app_config()

    if config is None:
        st.error("""
        **Groq API Key Not Configured!**

        Please add your free Groq API key to `config.yaml`:
        ```yaml
        api:
          groq_api_key: "gsk_your_actual_key_here"
        ```

        Get your free API key at: [console.groq.com](https://console.groq.com)
        """)
        st.stop()

    # ----------------------------------------------------------------
    # Topic input section
    # ----------------------------------------------------------------
    st.markdown("### Enter Your Research Topic")

    default_topic = selected_example if selected_example else ""

    if "last_topic" in st.session_state and not selected_example:
        default_topic = st.session_state.get("last_topic", "")

    topic = st.text_input(
        label="Research Topic",
        value=default_topic,
        placeholder="e.g., Impact of Artificial Intelligence on Healthcare",
        help="Enter any research topic. The AI will generate a comprehensive report.",
        label_visibility="collapsed"
    )

    # ----------------------------------------------------------------
    # Research button and pipeline execution
    # ----------------------------------------------------------------
    col_btn, col_info = st.columns([1, 3])

    with col_btn:
        start_button = st.button(
            "Start Research",
            type="primary",
            use_container_width=True,
            disabled=(not topic.strip()),
            help="Click to start the multi-agent research pipeline"
        )

    with col_info:
        if topic.strip():
            st.info(f"Ready to research: **{topic}**")
        else:
            st.info("Enter a research topic above to get started")

    # ----------------------------------------------------------------
    # Run the pipeline when button is clicked
    # ----------------------------------------------------------------
    if start_button and topic.strip():
        st.session_state["last_topic"] = topic

        if "research_result" in st.session_state:
            del st.session_state["research_result"]

        st.divider()
        st.markdown("### Research Pipeline Running...")

        with st.spinner("Agents are researching your topic... This may take 1-3 minutes."):
            try:
                result = run_research_pipeline(topic=topic, config=config)
                st.session_state["research_result"] = result
                st.success("Research complete! Scroll down to view your report.")

            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                st.exception(e)
                result = None

    # ----------------------------------------------------------------
    # Display results if available (from current run or session state)
    # ----------------------------------------------------------------
    if "research_result" in st.session_state:
        result = st.session_state["research_result"]

        if result:
            st.divider()
            display_results(result)

            if result.get("error"):
                with st.expander("Pipeline Warnings/Errors"):
                    st.warning(f"Some agents encountered issues: {result['error']}")
                    st.info("The pipeline continued with fallback behavior. Results may be incomplete.")

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    st.divider()
    st.markdown(
        '<div style="text-align:center; color:#888; font-size:0.85rem;">'
        'Multi-Agent Research Assistant | Built with LangGraph + Groq + Streamlit | '
        'All free resources -- no paid APIs required'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
else:
    main()
