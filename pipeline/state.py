"""
MARS — Shared State Schema
Defines all data structures that flow between agents through the LangGraph StateGraph.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


# ── Document (output of Search Agent) ─────────────────────────────────────────

@dataclass
class Document:
    title:   str
    url:     str
    snippet: str
    content: str                    # full extracted text
    score:   float = 1.0            # relevance score from reranker
    source:  str   = "duckduckgo"   # "duckduckgo" | "wikipedia" | "arxiv"


# ── Claim (extracted by Summarization Agent) ──────────────────────────────────

@dataclass
class Claim:
    text:       str
    source_url: str
    confidence: float = 1.0


# ── Summary (output of Summarization Agent) ───────────────────────────────────

@dataclass
class Summary:
    text:            str                    # full structured summary text
    claims:          list[Claim] = field(default_factory=list)
    executive_summary: str = ""


# ── Fact-check result (output of Fact-Checking Agent) ─────────────────────────

Verdict = Literal["VERIFIED", "UNVERIFIED", "CONTRADICTED"]

@dataclass
class FactCheckResult:
    claim:          Claim
    verdict:        Verdict
    confidence:     float
    evidence_quote: str = ""


@dataclass
class FactCheckedReport:
    results:            list[FactCheckResult] = field(default_factory=list)
    hallucination_rate: float = 0.0          # % of UNVERIFIED claims

    def compute_hallucination_rate(self) -> float:
        if not self.results:
            return 0.0
        unverified = sum(1 for r in self.results if r.verdict == "UNVERIFIED")
        self.hallucination_rate = unverified / len(self.results)
        return self.hallucination_rate


# ── Master state dict (shared across all LangGraph nodes) ─────────────────────

from typing import TypedDict, Optional

class ResearchState(TypedDict, total=False):
    query:          str
    sub_questions:  list[str]
    raw_docs:       list[Document]
    summary:        Summary
    fact_checked:   FactCheckedReport
    final_report:   str
    error:          Optional[str]
