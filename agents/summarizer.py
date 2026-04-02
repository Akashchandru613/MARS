"""
MARS — Summarization Agent
─────────────────────────────────────────────────────────────────────────────
Receives raw Document objects from the Search Agent and produces a structured
Summary using a Map-Reduce strategy:
  MAP:    Extract key claims from each document independently.
  REDUCE: Synthesize all claims into a coherent, cited research summary.

Owner: Abhishek Tuteja
Status: Phase 2 (in development)
"""

import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pipeline.state import Document, Claim, Summary, ResearchState
from config import get_groq_client, GROQ_MODEL, MAX_TOKENS, TEMPERATURE


MAP_SYSTEM = """You are a precise research analyst. Extract key factual claims
from the document provided. For each claim, record the source URL and your
confidence (0.0–1.0) that it is factually stated in the document.

Return ONLY valid JSON:
{
  "claims": [
    {"text": "...", "source_url": "...", "confidence": 0.9},
    ...
  ]
}"""

REDUCE_SYSTEM = """You are a research writer. Given a collection of factual
claims extracted from multiple sources, write a structured research summary.

Your output must be in this markdown format:

## Executive Summary
(2–3 sentence overview)

## Key Findings
- Finding 1 [Source: URL]
- Finding 2 [Source: URL]
- ...

## Sources
- [Title or URL]
- ...

Be concise and accurate. Do not introduce facts not present in the claims."""


class SummarizationAgent:
    def __init__(self):
        self.client = get_groq_client()

    def _map_document(self, doc: Document) -> list[Claim]:
        """Extract claims from a single document (MAP step)."""
        prompt = (
            f"Document Title: {doc.title}\n"
            f"Source URL: {doc.url}\n"
            f"Content:\n{doc.content[:2000]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": MAP_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data   = json.loads(raw)
            claims = [
                Claim(
                    text=c["text"],
                    source_url=c.get("source_url", doc.url),
                    confidence=c.get("confidence", 1.0),
                )
                for c in data.get("claims", [])
            ]
            return claims
        except Exception as e:
            print(f"[summarizer] MAP error for '{doc.title}': {e}")
            return []

    def _reduce_claims(self, all_claims: list[Claim], query: str) -> str:
        """Synthesize all claims into a structured summary (REDUCE step)."""
        claims_text = "\n".join(
            f"- {c.text} [Source: {c.source_url}]"
            for c in all_claims
        )
        prompt = (
            f"Original Research Query: {query}\n\n"
            f"Extracted Claims:\n{claims_text}\n\n"
            f"Please write the structured research summary."
        )
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": REDUCE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content

    def run(self, docs: list[Document], query: str) -> Summary:
        """Full map-reduce summarization pipeline."""
        all_claims: list[Claim] = []
        for doc in docs:
            claims = self._map_document(doc)
            all_claims.extend(claims)

        if not all_claims:
            return Summary(text="No content could be summarized.", claims=[])

        summary_text = self._reduce_claims(all_claims, query)

        # Extract executive summary (first paragraph after heading)
        exec_summary = ""
        lines = summary_text.split("\n")
        capture = False
        for line in lines:
            if "Executive Summary" in line:
                capture = True
                continue
            if capture and line.strip().startswith("##"):
                break
            if capture and line.strip():
                exec_summary += line.strip() + " "

        return Summary(
            text=summary_text,
            claims=all_claims,
            executive_summary=exec_summary.strip(),
        )


# ── LangGraph node wrapper ────────────────────────────────────────────────────

_agent = None

def summarization_node(state: ResearchState) -> ResearchState:
    """LangGraph node: map-reduce summarization over retrieved documents."""
    global _agent
    if state.get("error"):
        return state
    if _agent is None:
        _agent = SummarizationAgent()
    try:
        summary = _agent.run(
            docs=state.get("raw_docs", []),
            query=state.get("query", ""),
        )
        return {**state, "summary": summary}
    except Exception as e:
        return {**state, "error": f"Summarization failed: {e}"}
