"""
MARS — Single-Agent Baseline
─────────────────────────────────────────────────────────────────────────────
Control condition for the benchmark. One Llama 3.1 70B call handles search,
summarization, and fact-checking in a single prompt — same model, same tools,
no specialization. Every improvement the multi-agent system shows is measured
against this baseline.

Usage:
    python baseline/single_agent.py "What caused the 2008 financial crisis?"
"""

import sys
import json
import time
from dataclasses import dataclass, asdict
from ddgs import DDGS
import wikipedia as wikipedia_lib
import arxiv as arxiv_lib

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from config import get_groq_client, GROQ_MODEL, MAX_TOKENS, TEMPERATURE


# ── Search helpers (free, no API keys) ───────────────────────────────────────

def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via DuckDuckGo."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source":  "duckduckgo",
                })
    except Exception as e:
        print(f"[search] DuckDuckGo error: {e}")
    return results


def search_wikipedia(query: str) -> list[dict]:
    """Fetch a Wikipedia summary for the query."""
    try:
        page    = wikipedia_lib.page(query, auto_suggest=True)
        snippet = wikipedia_lib.summary(query, sentences=5, auto_suggest=True)
        return [{
            "title":   page.title,
            "url":     page.url,
            "snippet": snippet[:800],
            "source":  "wikipedia",
        }]
    except Exception:
        return []


def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    """Search arXiv for academic papers."""
    results = []
    try:
        search = arxiv_lib.Search(query=query, max_results=max_results)
        client = arxiv_lib.Client()
        for paper in client.results(search):
            results.append({
                "title":   paper.title,
                "url":     paper.entry_id,
                "snippet": paper.summary[:600],
                "source":  "arxiv",
            })
    except Exception as e:
        print(f"[search] arXiv error: {e}")
    return results


def gather_context(query: str) -> list[dict]:
    """Aggregate search results from all three free sources."""
    docs = []
    docs.extend(search_duckduckgo(query, max_results=4))
    docs.extend(search_wikipedia(query))
    docs.extend(search_arxiv(query, max_results=2))
    # deduplicate by URL
    seen, unique = set(), []
    for d in docs:
        if d["url"] not in seen:
            seen.add(d["url"])
            unique.append(d)
    return unique


# ── Baseline Output Schema ────────────────────────────────────────────────────

@dataclass
class BaselineResult:
    query:          str
    report:         str
    sources:        list[dict]
    latency_sec:    float
    model:          str
    token_count:    int = 0


# ── Single-Agent Pipeline ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant. Given a user query and a set of
retrieved source documents, you must:

1. SEARCH & SELECT: Identify the most relevant passages from the provided documents.
2. SUMMARIZE: Write a structured research summary with:
   - An executive summary (2–3 sentences)
   - Key findings (bulleted list, each with an inline citation [Source: URL])
3. FACT-CHECK: For each key finding, assess whether it is:
   - VERIFIED (clearly supported by the provided sources)
   - UNVERIFIED (not addressed by any source — flag it)
   - CONTRADICTED (conflicts with a source — correct it)
4. FINAL REPORT: Compile everything into a clean markdown report.

Be precise. Never invent facts. If a claim has no source support, say so explicitly."""


def run_baseline(query: str, verbose: bool = True) -> BaselineResult:
    """
    Run the single-agent baseline on a research query.
    Returns a BaselineResult with the report, sources, and timing.
    """
    client = get_groq_client()
    start  = time.time()

    if verbose:
        print(f"\n[baseline] Query: {query}")
        print("[baseline] Gathering context from DuckDuckGo, Wikipedia, arXiv...")

    docs = gather_context(query)

    if verbose:
        print(f"[baseline] Retrieved {len(docs)} documents")

    # Format retrieved docs into the prompt
    context_str = ""
    for i, doc in enumerate(docs, 1):
        context_str += (
            f"\n--- Source {i}: {doc['title']} ({doc['source']}) ---\n"
            f"URL: {doc['url']}\n"
            f"{doc['snippet']}\n"
        )

    user_message = (
        f"Research Query: {query}\n\n"
        f"Retrieved Documents:\n{context_str}\n\n"
        f"Please produce a complete research report following the instructions."
    )

    if verbose:
        print(f"[baseline] Calling {GROQ_MODEL}...")

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    latency    = time.time() - start
    report     = response.choices[0].message.content
    token_count = response.usage.total_tokens if response.usage else 0

    result = BaselineResult(
        query=query,
        report=report,
        sources=docs,
        latency_sec=round(latency, 2),
        model=GROQ_MODEL,
        token_count=token_count,
    )

    if verbose:
        print(f"[baseline] Done in {latency:.2f}s | Tokens: {token_count}")
        print("\n" + "="*70)
        print(report)
        print("="*70)

    return result


def save_result(result: BaselineResult, path=None) -> str:
    """Save baseline result to JSON for later evaluation."""
    import pathlib, datetime
    if path is None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = result.query[:40].replace(" ", "_").replace("/", "_")
        path = str(
            pathlib.Path(__file__).parent.parent
            / "data" / "results" / f"baseline_{safe}_{ts}.json"
        )
    data = asdict(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[baseline] Result saved → {path}")
    return path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python baseline/single_agent.py \"your research query\"")
        sys.exit(1)

    query  = " ".join(sys.argv[1:])
    result = run_baseline(query, verbose=True)
    save_result(result)
