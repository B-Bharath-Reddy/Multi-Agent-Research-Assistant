# ============================================================
# graph/__init__.py
# ============================================================
# PURPOSE:
#   This file marks the 'graph' directory as a Python package.
#   It allows other modules to import from the graph package like:
#     from graph.research_graph import run_research_pipeline
#
# WHY THIS DIRECTORY EXISTS:
#   The 'graph' directory contains the LangGraph state machine that
#   orchestrates all 5 agents in the research pipeline. Separating
#   the orchestration logic into its own package keeps the codebase
#   clean and modular — agents are in 'agents/', tools are in 'tools/',
#   and the pipeline wiring is in 'graph/'.
# ============================================================
