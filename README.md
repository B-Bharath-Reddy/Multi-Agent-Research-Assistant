# Multi-Agent Research Assistant

> Autonomous AI research powered by LangGraph + Groq LLaMA 3.3-70B
> 5 specialized AI agents that collaborate to research any topic and generate a comprehensive, reviewed report -- completely free.


## Project Overview

The **Multi-Agent Research Assistant** is an autonomous AI system that takes any research topic as input and produces a comprehensive, professionally written research report -- completely automatically, using only free tools and APIs.

Instead of a single AI model trying to do everything, this system uses **5 specialized AI agents**, each with a distinct role, working together in a coordinated pipeline orchestrated by **LangGraph**. This mirrors how real research teams work: a planner, researchers, a writer, and an editor -- each doing what they do best.

**What it does in 3 steps:**
1. You type a research topic (e.g., "Impact of AI on Healthcare")
2. 5 AI agents automatically research, summarize, write, and review
3. You get a full Markdown research report with a quality score

---

## Objective

The primary objectives of this project are:

1. **Demonstrate multi-agent AI architecture** -- Show how multiple specialized agents can collaborate on a complex task that no single agent could do as well alone.

2. **Implement real-world tool use** -- Agents use actual web search (DuckDuckGo) and Wikipedia to ground their outputs in real, current information -- not just LLM training data.

3. **Apply LangGraph for agent orchestration** -- Use the industry-standard framework for building production-grade multi-agent systems.

4. **Build a complete, deployable application** -- Not just a script, but a full web application with a professional UI, configuration management, error handling, and file output.

5. **Use only free resources** -- Groq API (free tier), DuckDuckGo (no API key), Wikipedia (no API key), Streamlit (free hosting) -- zero cost to run.

---




## Tech Stack

| Component | Technology | Why Chosen |
|---|---|---|
| LLM | Groq LLaMA 3.3-70B | Free, ultra-fast inference (500+ tokens/sec) |
| Agent Framework | LangGraph | Industry standard for multi-agent systems |
| LLM Interface | LangChain + langchain-groq | Clean Python API for LLM interactions |
| Web Search | DuckDuckGo (duckduckgo-search) | Free, no API key, privacy-focused |
| Knowledge Base | Wikipedia (wikipedia library) | Free, authoritative, structured content |
| Web UI | Streamlit | Fast to build, free to deploy, renders Markdown |
| Configuration | PyYAML (config.yaml) | Single config file for all settings |
| Language | Python 3.9+ | Industry standard for AI/ML |

**Total cost to run: $0** -- All tools are free.

---

## Project Structure

```
multi-agent-research-assistant/
|
|-- config.yaml                    <- Master configuration (API keys, model, settings)
|-- app.py                         <- Streamlit web UI (main entry point)
|-- requirements.txt               <- Python dependencies
|-- README.md                      <- This file
|
|-- agents/                        <- The 5 AI agents
|   |-- __init__.py                <- Package marker
|   |-- planner_agent.py           <- Agent 1: Breaks topic into sub-questions
|   |-- search_agent.py            <- Agent 2: Fetches web + Wikipedia data
|   |-- summarizer_agent.py        <- Agent 3: Summarizes raw search results
|   |-- writer_agent.py            <- Agent 4: Writes the full research report
|   `-- critic_agent.py            <- Agent 5: Reviews and scores the report
|
|-- graph/                         <- LangGraph pipeline orchestration
|   |-- __init__.py                <- Package marker
|   `-- research_graph.py          <- StateGraph connecting all 5 agents
|
|-- tools/                         <- External tool integrations
|   |-- __init__.py                <- Package marker
|   |-- web_search.py              <- DuckDuckGo search wrapper
|   `-- wikipedia_tool.py          <- Wikipedia search wrapper
|
|-- utils/                         <- Utility modules
|   |-- __init__.py                <- Package marker
|   `-- config_loader.py           <- Loads and validates config.yaml
|
`-- reports/                       <- Auto-generated research reports (created at runtime)
    `-- (your_topic_timestamp.md)  <- Saved reports appear here
```

---

## Architecture and Agent Flow

### High-Level Pipeline

```
User Input (Topic)
        |
        v
+-----------------------------------------------------------+
|          LangGraph StateGraph (research_graph.py)         |
|                                                           |
|  +-------------+    +-------------+    +--------------+  |
|  |   PLANNER   |--->|   SEARCH    |--->|  SUMMARIZER  |  |
|  |   Agent 1   |    |   Agent 2   |    |   Agent 3    |  |
|  +-------------+    +-------------+    +--------------+  |
|         |                  |                   |          |
|   Sub-questions      Web + Wiki           Clean           |
|   (5 questions)      Results             Summaries        |
|                                               |           |
|                    +--------------+    +--------------+  |
|                    |    CRITIC    |<---|    WRITER    |  |
|                    |   Agent 5   |    |   Agent 4    |  |
|                    +--------------+    +--------------+  |
|                           |                   |          |
|                    Quality Review        Full Report      |
|                    (Score + Feedback)    (Markdown)       |
+-----------------------------------------------------------+
        |
        v
Final Output:
  [OK] Research Report (Markdown)
  [OK] Quality Score (1-10)
  [OK] Saved to reports/ folder
  [OK] Downloadable from UI
```

### LangGraph State Flow

```
START
  |
  v
[planner_node]
  |  Reads:  state.topic, state.config
  |  Writes: state.sub_questions
  v
[search_node]
  |  Reads:  state.sub_questions, state.config
  |  Writes: state.search_results
  v
[summarizer_node]
  |  Reads:  state.search_results, state.config
  |  Writes: state.summaries
  v
[writer_node]
  |  Reads:  state.topic, state.summaries, state.config
  |  Writes: state.report
  v
[critic_node]
  |  Reads:  state.topic, state.report, state.config
  |  Writes: state.review
  v
[save_node]
  |  Reads:  state.report, state.review, state.config
  |  Writes: state.status (file path)
  v
END
```

---

## Agent Descriptions

### Agent 1: Planner Agent (agents/planner_agent.py)

**Role:** Research strategist and question decomposer

**What it does:**
Takes a broad research topic and breaks it down into 5 specific, focused sub-questions that together provide comprehensive coverage of the topic.

**Why it's needed:**
A single broad search query returns poor, unfocused results. By decomposing the topic into targeted sub-questions, the Search Agent can fetch precise, relevant information for each dimension of the topic.

**How it works:**
1. Receives the user's topic (e.g., "Quantum Computing")
2. Sends it to Groq LLaMA with a carefully crafted system prompt
3. The LLM generates 5 focused sub-questions covering: definition, applications, challenges, trends, and future
4. Parses the numbered list response into a Python list
5. Returns the list to the pipeline state

**Example:**
```
Input:  "Quantum Computing"
Output: [
  "What is quantum computing and how does it differ from classical computing?",
  "What are the main real-world applications of quantum computing?",
  "What are the key technical challenges in building quantum computers?",
  "What are the latest breakthroughs and developments in quantum computing?",
  "What is the future outlook and timeline for practical quantum computing?"
]
```

---

### Agent 2: Search Agent (agents/search_agent.py)

**Role:** Information retrieval specialist

**What it does:**
For each sub-question from the Planner Agent, searches two sources:
1. DuckDuckGo -- Current web pages, news, recent articles
2. Wikipedia -- Authoritative, encyclopedic background knowledge

**Why it's needed:**
LLMs have a knowledge cutoff date and can hallucinate facts. The Search Agent grounds the pipeline in real, current, verifiable information -- this is the "Retrieval" in RAG (Retrieval-Augmented Generation).

**How it works:**
1. Receives the list of 5 sub-questions
2. For each question:
   - Calls search_web() -> fetches 3 DuckDuckGo results
   - Extracts the key topic from the question for Wikipedia
   - Calls search_wikipedia() -> fetches a 5-sentence summary
   - Combines both into a single formatted context string
3. Returns a dictionary mapping each question to its search results

**Dual-source strategy:**
- DuckDuckGo -> recency and breadth (current events, recent research)
- Wikipedia -> depth and accuracy (definitions, history, established facts)

---

### Agent 3: Summarizer Agent (agents/summarizer_agent.py)

**Role:** Information distiller and cleaner

**What it does:**
Takes the raw, messy search results (web snippets + Wikipedia text) for each sub-topic and uses the LLM to produce a clean, concise, factual ~150-word summary.

**Why it's needed:**
Raw search results contain ads, navigation text, repeated information, and irrelevant content. The Summarizer acts as an intelligent filter -- extracting only the key facts and presenting them in a clean format that the Writer Agent can easily use.

**How it works:**
1. Receives the search results dictionary from the Search Agent
2. For each sub-question:
   - Reads the formatted_context (combined web + Wikipedia text)
   - Sends it to Groq LLaMA with a summarization prompt
   - The LLM extracts key facts and writes a clean ~150-word summary
3. Returns a dictionary mapping each question to its clean summary

**Output quality:**
- Factual, neutral academic tone
- Specific data points and statistics preserved
- No opinions or speculation
- Directly addresses the sub-question

---

### Agent 4: Writer Agent (agents/writer_agent.py)

**Role:** Professional research report author

**What it does:**
Takes all 5 clean summaries from the Summarizer Agent and synthesizes them into a single, comprehensive, professionally written research report in Markdown format with 7 structured sections.

**Why it's needed:**
Having 5 separate summaries is not a research report. The Writer Agent weaves them together into a coherent narrative with proper flow, transitions, and structure -- transforming research notes into a publication-ready document.

**How it works:**
1. Receives the topic and all 5 summaries
2. Formats summaries into a structured input document
3. Sends to Groq LLaMA with a detailed writing prompt specifying:
   - Required sections (from config.yaml)
   - Markdown formatting rules
   - Academic tone requirements
   - Synthesis instructions (not just copying summaries)
4. Adds metadata header (title, date, model info)
5. Returns the complete Markdown report

**Report sections (configurable in config.yaml):**
1. Introduction
2. Key Findings
3. Detailed Analysis
4. Current Trends
5. Challenges and Limitations
6. Future Outlook
7. Conclusion

---

### Agent 5: Critic Agent (agents/critic_agent.py)

**Role:** Quality control reviewer and evaluator

**What it does:**
Reads the complete research report and evaluates it across 5 dimensions, producing a numerical score (1-10), letter grade, verdict (APPROVED/NEEDS IMPROVEMENT), strengths, weaknesses, and actionable improvement suggestions.

**Why it's needed:**
Without a quality check, there's no way to assess whether the generated report is actually good. The Critic Agent adds a "self-reflection" capability -- the system evaluates its own output. This is a key pattern in advanced AI systems (similar to Constitutional AI).

**How it works:**
1. Receives the topic and full report
2. Sends to Groq LLaMA with a structured evaluation rubric
3. The LLM scores each criterion and provides detailed feedback
4. Parses the structured response to extract:
   - Overall score (1-10)
   - Letter grade (A through F)
   - Per-criterion scores
   - Strengths (bullet list)
   - Weaknesses (bullet list)
   - Improvement suggestions (bullet list)
   - Detailed qualitative review
5. Determines verdict: APPROVED (>=6/10) or NEEDS IMPROVEMENT (<6/10)

**Evaluation criteria (configurable in config.yaml):**
1. Accuracy and factual correctness
2. Completeness of coverage
3. Clarity and readability
4. Logical structure and flow
5. Use of evidence and sources

---

## File-by-File Explanation

### config.yaml -- Master Configuration
The single source of truth for all settings. Edit this file to customize:
- Your Groq API key
- LLM model and parameters (temperature, max tokens)
- Search settings (number of results, Wikipedia sentences)
- Agent-specific settings (number of sub-questions, summary length)
- Report sections and structure
- Output settings (save location, format)
- Streamlit UI settings (title, example topics)

### app.py -- Streamlit Web Application
The main entry point. Run with `python -m streamlit run app.py`.
- Renders the web UI with sidebar, topic input, and results display
- Calls run_research_pipeline() when user clicks "Start Research"
- Displays results in 4 tabs: Report, Quality Review, Research Process, Download
- Uses st.session_state to persist results between UI interactions

### graph/research_graph.py -- LangGraph Pipeline
The central orchestrator. Defines:
- ResearchState TypedDict -- the shared state schema
- 5 node functions (one per agent) + save node
- The StateGraph with edges connecting all nodes
- run_research_pipeline() -- the main function called by app.py

### agents/planner_agent.py -- Planner Agent
Decomposes a broad topic into 5 focused sub-questions using Groq LLaMA.

### agents/search_agent.py -- Search Agent
Fetches DuckDuckGo web results and Wikipedia content for each sub-question.

### agents/summarizer_agent.py -- Summarizer Agent
Summarizes raw search results into clean, concise per-topic summaries.

### agents/writer_agent.py -- Writer Agent
Synthesizes all summaries into a full structured research report in Markdown.

### agents/critic_agent.py -- Critic Agent
Evaluates the report quality with a score, grade, and detailed feedback.

### tools/web_search.py -- DuckDuckGo Search Tool
Wraps the duckduckgo-search library into a clean function returning structured results.

### tools/wikipedia_tool.py -- Wikipedia Search Tool
Wraps the wikipedia library to fetch article summaries with disambiguation handling.

### utils/config_loader.py -- Configuration Loader
Loads and validates config.yaml, providing safe access to all settings.

---

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- A free Groq API key (get one at https://console.groq.com)

### Step 1: Navigate to the Project Folder
```bash
cd path/to/agent
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Your Free Groq API Key
1. Go to https://console.groq.com
2. Sign up for a free account
3. Navigate to API Keys -> Create API Key
4. Copy your key (it starts with gsk_...)

### Step 5: Configure Your API Key
Open `config.yaml` and replace the placeholder with your actual key:
```yaml
api:
  groq_api_key: "gsk_your_actual_key_here"
```

---

## How to Run

### IMPORTANT: Use the Correct Python

This project requires packages installed in your Python environment.
Always run the app using the same Python that has the packages installed.

---

### Option 1: Standard Run (Recommended)

If you created a virtual environment and activated it:

```bash
# Make sure your virtual environment is activated first:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Then run:
python -m streamlit run app.py
```

The app will open in your browser at: **http://localhost:8501**

---

### Option 2: Run with a Specific Python Path

If you have multiple Python installations (e.g., Anaconda + system Python),
specify the exact Python executable that has the packages installed:

**Windows (system Python):**
```bash
"C:\Program Files\Python312\python.exe" -m streamlit run app.py
```

**Windows (Anaconda environment):**
```bash
"C:\Users\YourName\anaconda3\envs\your_env\python.exe" -m streamlit run app.py
```

**Mac/Linux:**
```bash
/usr/local/bin/python3 -m streamlit run app.py
```

---

### Option 3: Find Which Python Has the Packages

If you're unsure which Python to use, run this to check:
```bash
python -c "import langgraph; print('OK - use this Python')"
```

If it prints "OK", use `python -m streamlit run app.py`.
If it raises an error, try the specific path options above.

---

### What Happens After You Run

1. The terminal will show:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   ```

2. Your browser opens automatically (or navigate to http://localhost:8501)

3. You will see the Multi-Agent Research Assistant web interface

4. Enter a research topic in the text box (e.g., "Quantum Computing")

5. Click the **"Start Research"** button

6. Wait 1-3 minutes while the 5 agents work:
   - Agent 1 (Planner): Generates 5 sub-questions
   - Agent 2 (Search): Fetches web + Wikipedia data
   - Agent 3 (Summarizer): Creates clean summaries
   - Agent 4 (Writer): Writes the full report
   - Agent 5 (Critic): Reviews and scores the report

7. View your results in 4 tabs:
   - **Research Report** -- The full Markdown report
   - **Quality Review** -- Score, grade, strengths, weaknesses
   - **Research Process** -- Sub-questions, summaries, search results
   - **Download** -- Download as .md file

8. Reports are also auto-saved to the `reports/` folder as:
   ```
   reports/your_topic_20260218_221500.md
   ```

---

### Stopping the App

Press `Ctrl + C` in the terminal to stop the Streamlit server.

---

### Troubleshooting

**Error: "No module named 'langgraph'"**
- You are using the wrong Python. Use `python -m streamlit run app.py` with the Python that has packages installed.
- Or install packages: `pip install -r requirements.txt`

**Error: "Groq API key not set"**
- Open `config.yaml` and replace `"your_groq_api_key_here"` with your actual key from https://console.groq.com

**Error: "No module named 'streamlit'"**
- Install streamlit: `pip install streamlit`

**App opens but shows API key error**
- Make sure you saved config.yaml after adding your API key
- The key should start with `gsk_`

**Search returns no results**
- Check your internet connection
- DuckDuckGo may be temporarily rate-limiting -- wait a minute and try again

---

## Example Output

### Input Topic:
```
"Impact of Artificial Intelligence on Healthcare"
```

### Generated Sub-Questions (Planner Agent):
```
1. What are the current AI applications in medical diagnosis and imaging?
2. How is AI transforming drug discovery and pharmaceutical research?
3. What are the ethical concerns and regulatory challenges of AI in healthcare?
4. How does AI impact patient data privacy and electronic health records?
5. What is the future outlook for AI-powered personalized medicine?
```

### Report Structure (Writer Agent):
```markdown
# Impact of Artificial Intelligence on Healthcare: A Comprehensive Research Report

---
**Generated by:** Multi-Agent Research Assistant
**Date:** February 18, 2026
**Sub-topics Researched:** 5
**Model:** Groq LLaMA 3.3-70B

## Introduction
Artificial intelligence is fundamentally transforming healthcare...

## Key Findings
The integration of AI into healthcare has yielded remarkable results...

## Detailed Analysis
[2-4 paragraphs of in-depth analysis]

## Current Trends
[Latest developments and emerging patterns]

## Challenges and Limitations
[Technical, ethical, and regulatory challenges]

## Future Outlook
[Predictions and timeline for AI in healthcare]

## Conclusion
[Synthesis of key insights and implications]
```

### Quality Review (Critic Agent):
```
Score: 8/10 | Grade: B+ | Verdict: APPROVED

Strengths:
- Comprehensive coverage of all major AI healthcare applications
- Clear structure with logical flow between sections
- Specific examples and data points throughout

Suggestions:
- Could include more specific statistics on adoption rates
- Consider adding case studies from specific hospitals
```

---

## Configuration Guide

All settings are in `config.yaml`. Key settings to customize:

```yaml
# Change the LLM model
llm:
  model: "llama-3.3-70b-versatile"   # Best quality
  # model: "llama-3.1-8b-instant"    # Faster, lighter

# Control research depth
agents:
  planner:
    num_subquestions: 5    # More = more comprehensive but slower

  search:
    use_wikipedia: true    # Set false to skip Wikipedia

  summarizer:
    max_summary_length: 150   # Words per summary

  writer:
    report_sections:
      - "Introduction"
      - "Key Findings"
      - "Detailed Analysis"
      - "Current Trends"
      - "Challenges and Limitations"
      - "Future Outlook"
      - "Conclusion"

  critic:
    min_score_threshold: 6   # Reports below this are flagged

output:
  save_reports: true
  output_dir: "reports/"
  include_review: true
```


