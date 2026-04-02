# System Architecture — MARS

**Project:** MARS — Multi-Agent Research System  
**Authors:** Akash Chandru, Abhishek Tuteja  
**Version:** Phase 1  

---

## 1. System Overview

MARS is a multi-agent pipeline where four specialized agents collaborate under
a LangGraph StateGraph to answer research queries with higher accuracy and lower
hallucination rates than a single-agent baseline.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│            ORCHESTRATOR AGENT               │
│   Decomposes query into 3–5 sub-questions   │
└──────────────────┬──────────────────────────┘
                   │ sub_questions[]
                   ▼
┌─────────────────────────────────────────────┐
│              SEARCH AGENT                   │
│   DuckDuckGo + Wikipedia + arXiv            │
│   Reranks by cosine similarity (MiniLM)     │
└──────────────────┬──────────────────────────┘
                   │ raw_docs[]
                   ▼
┌─────────────────────────────────────────────┐
│           SUMMARIZATION AGENT               │
│   Map: extract claims per document          │
│   Reduce: synthesize into structured report │
└──────────────────┬──────────────────────────┘
                   │ summary (claims + text)
                   ▼
┌─────────────────────────────────────────────┐
│           FACT-CHECKING AGENT               │
│   Verifies each claim → VERIFIED /          │
│   UNVERIFIED / CONTRADICTED                 │
│   Computes hallucination rate               │
└──────────────────┬──────────────────────────┘
                   │ fact_checked report
                   ▼
┌─────────────────────────────────────────────┐
│         COMPILE REPORT (Orchestrator)       │
│   Merges all outputs into final markdown    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
             Final Report
```

---

## 2. Shared State Schema

All agents communicate exclusively through a single `ResearchState` TypedDict.
No agent calls another agent directly.

```python
class ResearchState(TypedDict):
    query:          str                    # raw user input
    sub_questions:  list[str]             # from orchestrator
    raw_docs:       list[Document]        # from search agent
    summary:        Summary               # from summarization agent
    fact_checked:   FactCheckedReport     # from fact-checking agent
    final_report:   str                   # compiled markdown output
    error:          str | None            # propagated on failure
```

---

## 3. Agent Roles

| Agent | Input | Output | LLM Calls |
|---|---|---|---|
| Orchestrator | `query` | `sub_questions` | 1 (decompose) |
| Search | `sub_questions` | `raw_docs` | 0 (tool calls only) |
| Summarizer | `raw_docs` | `summary` | N+1 (N=doc count for map + 1 reduce) |
| Fact-checker | `summary`, `raw_docs` | `fact_checked` | 1 per claim |
| Compile | `fact_checked` | `final_report` | 1 |

---

## 4. Data Flow Details

### 4.1 Orchestrator → Search
The orchestrator produces a `sub_questions` list (typically 3–5 items).
Example for query *"What caused the 2008 financial crisis?"*:
```json
{
  "sub_questions": [
    "What role did subprime mortgages play in the 2008 crisis?",
    "How did mortgage-backed securities contribute to the 2008 collapse?",
    "What was the regulatory environment before the 2008 financial crisis?"
  ]
}
```

### 4.2 Search → Summarizer
Each sub-question triggers parallel searches across three sources.
Documents are reranked by cosine similarity and top-3 per sub-question are kept.
Result: a deduplicated list of `Document` objects (typically 6–12 docs).

### 4.3 Summarizer → Fact-checker
The summarizer uses map-reduce:
- **MAP:** one LLM call per document → extracts a list of `Claim` objects
- **REDUCE:** one LLM call over all claims → produces structured markdown summary

The `Summary` object carries both the full text and the individual `Claim` list,
which the fact-checker uses for verification.

### 4.4 Fact-checker → Compile
Each claim gets a `FactCheckResult` with:
- `verdict`: VERIFIED | UNVERIFIED | CONTRADICTED
- `confidence`: 0.0–1.0
- `evidence_quote`: the exact text from the source that supports the verdict

The `FactCheckedReport` aggregates these and computes:
```
hallucination_rate = count(UNVERIFIED) / total_claims
```

---

## 5. Error Handling & Fallback Strategy

```
Search Agent:
  Primary:  DuckDuckGo (no API key, always available)
  Fallback: Wikipedia + arXiv (always available)
  On error: propagate state["error"] and continue with empty docs

LLM Calls (all agents):
  - temperature=0.0 for fact-checking (deterministic)
  - temperature=0.2 for summarization (slight diversity)
  - temperature=0.3 for decomposition (creative sub-question generation)
  - Retry: tenacity with 3 attempts, exponential backoff
  - On persistent failure: mark state["error"] and skip downstream nodes
```

---

## 6. Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| LLM | Groq + Llama 3.1 70B | Free tier, fast inference, strong performance |
| Agent framework | LangGraph | Typed state, explicit control flow, inspectable |
| Web search | DuckDuckGo (`duckduckgo-search`) | Free, no API key |
| Academic search | arXiv Python client | Free, no API key |
| Encyclopedia | Wikipedia API | Free, no API key |
| Reranking | `sentence-transformers` MiniLM | Lightweight, runs locally |
| Evaluation | `rouge-score`, custom metrics | Standard NLP benchmarking |
| Language | Python 3.11+ | Team familiarity |

**Total infrastructure cost: $0**

---

## 7. Evaluation Design

Both systems receive the same 10-query benchmark set.

| Metric | Formula | Better if |
|---|---|---|
| Hallucination Rate | UNVERIFIED / total claims | Lower |
| Source Coverage | claims with valid URL / total | Higher |
| ROUGE-L | F1 vs reference answer | Higher |
| Claim Density | # distinct claims per report | Higher |
| Latency | wall-clock seconds per query | Lower |

The hypothesis: **multi-agent specialization reduces hallucination rate and
improves source coverage at the cost of higher latency.**
