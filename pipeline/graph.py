"""
MARS — LangGraph Pipeline
─────────────────────────────────────────────────────────────────────────────
Wires all four agent nodes into a LangGraph StateGraph.
Each node reads from and writes to the shared ResearchState dict.

Flow:
  orchestrator → search → summarize → fact_check → compile_report → END
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from pipeline.state import ResearchState

from agents.orchestrator import orchestrator_node, compile_report_node
from agents.search       import search_node
from agents.summarizer   import summarization_node
from agents.fact_checker import fact_check_node


def build_graph():
    """Build and compile the MARS multi-agent research pipeline."""
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("orchestrator",    orchestrator_node)
    graph.add_node("search",          search_node)
    graph.add_node("summarize",       summarization_node)
    graph.add_node("fact_check",      fact_check_node)
    graph.add_node("compile_report",  compile_report_node)

    # Define edges (linear pipeline for Phase 2; conditional routing in Phase 3)
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator",   "search")
    graph.add_edge("search",         "summarize")
    graph.add_edge("summarize",      "fact_check")
    graph.add_edge("fact_check",     "compile_report")
    graph.add_edge("compile_report", END)

    return graph.compile()


# Singleton app instance
_app = None

def get_pipeline():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
