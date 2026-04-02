"""
MARS — Orchestrator Agent
─────────────────────────────────────────────────────────────────────────────
Receives the raw user query, decomposes it into sub-questions, manages the
task queue, and compiles the final report from all specialist agents.

Owner: Akash Chandru
Status: Phase 2 (in development)
"""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from config import get_groq_client, GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from pipeline.state import ResearchState


DECOMPOSE_SYSTEM = """You are a research orchestrator. Given a user query,
decompose it into 3–5 focused sub-questions that together fully answer the
main question. Each sub-question should be independently searchable.

Return ONLY valid JSON in this format:
{"sub_questions": ["question 1", "question 2", "question 3"]}"""

COMPILE_SYSTEM = """You are a research editor. Given a fact-checked research
report, compile it into a final polished markdown document with:
- A clear title
- Executive summary (2–3 sentences)
- Key Findings (bulleted, with inline citations)
- Fact-Check Summary (verified / flagged counts)
- Sources list

Be concise, accurate, and professional."""


class OrchestratorAgent:
    def __init__(self):
        self.client = get_groq_client()

    def decompose_query(self, query: str) -> list[str]:
        """Break a research query into 3–5 focused sub-questions."""
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user",   "content": f"Query: {query}"},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return data["sub_questions"]

    def compile_report(self, state: ResearchState) -> str:
        """Merge fact-checked results into a final polished report."""
        fact_checked = state.get("fact_checked")
        query        = state.get("query", "")

        if not fact_checked:
            return "Error: no fact-checked content available."

        # Build a structured input for the compiler
        results_text = ""
        for r in fact_checked.results:
            results_text += (
                f"- [{r.verdict}] {r.claim.text}\n"
                f"  Source: {r.claim.source_url}\n"
                f"  Evidence: {r.evidence_quote[:200]}\n\n"
            )

        hallucination_rate = fact_checked.hallucination_rate * 100

        prompt = (
            f"Original Query: {query}\n\n"
            f"Fact-Checked Claims:\n{results_text}\n"
            f"Overall Hallucination Rate: {hallucination_rate:.1f}%\n\n"
            f"Please compile the final research report."
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": COMPILE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content


# ── LangGraph node wrapper ────────────────────────────────────────────────────

_agent = None

def orchestrator_node(state: ResearchState) -> ResearchState:
    """LangGraph node: decompose query into sub-questions."""
    global _agent
    if _agent is None:
        _agent = OrchestratorAgent()
    try:
        sub_questions = _agent.decompose_query(state["query"])
        return {**state, "sub_questions": sub_questions}
    except Exception as e:
        return {**state, "error": f"Orchestrator failed: {e}", "sub_questions": []}


def compile_report_node(state: ResearchState) -> ResearchState:
    """LangGraph node: compile final report from fact-checked data."""
    global _agent
    if _agent is None:
        _agent = OrchestratorAgent()
    try:
        report = _agent.compile_report(state)
        return {**state, "final_report": report}
    except Exception as e:
        return {**state, "error": f"Compile failed: {e}", "final_report": ""}
