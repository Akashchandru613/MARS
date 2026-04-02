"""
MARS — Fact-Checking Agent
─────────────────────────────────────────────────────────────────────────────
Verifies each claim in the Summary against the original retrieved documents.
Labels every claim as VERIFIED, UNVERIFIED, or CONTRADICTED and computes the
overall hallucination rate — the primary quality metric for the benchmark.

Owner: Abhishek Tuteja
Status: Phase 2 (in development)
"""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pipeline.state import (
    Claim, Document, FactCheckResult, FactCheckedReport, ResearchState
)
from config import get_groq_client, GROQ_MODEL


VERIFY_SYSTEM = """You are a fact-checker. Given a claim and a set of source
passages, determine whether the claim is:

  VERIFIED      — clearly and directly supported by at least one source
  CONTRADICTED  — directly contradicted or refuted by a source
  UNVERIFIED    — not addressed by any source (possible hallucination)

Return ONLY valid JSON:
{
  "verdict": "VERIFIED" | "CONTRADICTED" | "UNVERIFIED",
  "confidence": 0.0–1.0,
  "evidence_quote": "the exact quote from the source that supports your verdict"
}"""


class FactCheckAgent:
    def __init__(self):
        self.client = get_groq_client()

    def _verify_claim(self, claim: Claim, docs: list[Document]) -> FactCheckResult:
        """Verify a single claim against the retrieved documents."""
        # Build evidence block (top-3 most relevant docs by snippet overlap)
        evidence = "\n\n".join(
            f"[Source {i+1}: {d.url}]\n{d.content[:800]}"
            for i, d in enumerate(docs[:5])
        )
        prompt = (
            f"Claim to verify: {claim.text}\n\n"
            f"Available Sources:\n{evidence}"
        )
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=256,
                temperature=0.0,   # deterministic — critical for fact-checking
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return FactCheckResult(
                claim=claim,
                verdict=data.get("verdict", "UNVERIFIED"),
                confidence=data.get("confidence", 0.5),
                evidence_quote=data.get("evidence_quote", ""),
            )
        except Exception as e:
            print(f"[fact_checker] Error verifying claim: {e}")
            return FactCheckResult(
                claim=claim,
                verdict="UNVERIFIED",
                confidence=0.0,
                evidence_quote="",
            )

    def run(self, summary, docs: list[Document]) -> FactCheckedReport:
        """Verify all claims in the summary and return a FactCheckedReport."""
        results = []
        for claim in summary.claims:
            result = self._verify_claim(claim, docs)
            results.append(result)
            verdict_icon = {"VERIFIED": "✓", "UNVERIFIED": "?", "CONTRADICTED": "✗"}
            print(
                f"[fact_checker] {verdict_icon.get(result.verdict, '?')} "
                f"{result.verdict} ({result.confidence:.2f}) — {claim.text[:60]}..."
            )

        report = FactCheckedReport(results=results)
        rate   = report.compute_hallucination_rate()
        print(f"[fact_checker] Hallucination rate: {rate*100:.1f}%")
        return report


# ── LangGraph node wrapper ────────────────────────────────────────────────────

_agent = None

def fact_check_node(state: ResearchState) -> ResearchState:
    """LangGraph node: verify all claims against retrieved documents."""
    global _agent
    if state.get("error"):
        return state
    if _agent is None:
        _agent = FactCheckAgent()
    try:
        fact_checked = _agent.run(
            summary=state.get("summary"),
            docs=state.get("raw_docs", []),
        )
        return {**state, "fact_checked": fact_checked}
    except Exception as e:
        return {**state, "error": f"Fact-check failed: {e}"}
