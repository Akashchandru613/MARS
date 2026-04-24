"""
MARS — Search Agent
─────────────────────────────────────────────────────────────────────────────
Takes sub-questions from the orchestrator, retrieves documents from three
free sources (DuckDuckGo, Wikipedia, arXiv) and optionally Tavily, reranks by relevance using
sentence-transformers, and returns deduplicated Document objects.

Owner: Akash Chandru
Status: Phase 2 (in development)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pipeline.state import Document, ResearchState
from config import MAX_SEARCH_RESULTS, MAX_DOCS_PER_QUERY, TAVILY_API_KEY, USE_TAVILY

from ddgs import DDGS
import wikipedia as wikipedia_lib
import arxiv as arxiv_lib


# ── Source-specific fetchers ──────────────────────────────────────────────────

def _duckduckgo(query: str, n: int = 5) -> list[Document]:
    docs = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                docs.append(Document(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    content=r.get("body", ""),
                    source="duckduckgo",
                ))
    except Exception as e:
        print(f"[search] DuckDuckGo error: {e}")
    return docs


def _wikipedia(query: str) -> list[Document]:
    try:
        page    = wikipedia_lib.page(query, auto_suggest=True)
        snippet = wikipedia_lib.summary(query, sentences=5, auto_suggest=True)
        return [Document(
            title=page.title,
            url=page.url,
            snippet=snippet[:500],
            content=page.content[:3000],
            source="wikipedia",
        )]
    except Exception:
        return []


def _tavily(query: str, n: int = 5) -> list[Document]:
    """Fetch results from Tavily web search API."""
    docs = []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=n, search_depth="basic")
        for r in response.get("results", []):
            docs.append(Document(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:500],
                content=r.get("content", ""),
                source="tavily",
            ))
    except Exception as e:
        print(f"[search] Tavily error: {e}")
    return docs


def _arxiv(query: str, n: int = 3) -> list[Document]:
    docs = []
    try:
        client = arxiv_lib.Client()
        for paper in client.results(arxiv_lib.Search(query=query, max_results=n)):
            docs.append(Document(
                title=paper.title,
                url=paper.entry_id,
                snippet=paper.summary[:500],
                content=paper.summary,
                source="arxiv",
            ))
    except Exception as e:
        print(f"[search] arXiv error: {e}")
    return docs


# ── Reranker ──────────────────────────────────────────────────────────────────

def _rerank(docs: list[Document], query: str) -> list[Document]:
    """Score docs by cosine similarity to the query using MiniLM embeddings."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model  = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb  = model.encode(query, convert_to_tensor=True)
        d_embs = model.encode([d.snippet for d in docs], convert_to_tensor=True)
        scores = util.cos_sim(q_emb, d_embs)[0].tolist()
        for doc, score in zip(docs, scores):
            doc.score = round(float(score), 4)
        docs.sort(key=lambda d: d.score, reverse=True)
    except ImportError:
        pass   # sentence-transformers not installed; skip reranking
    return docs


# ── Search Agent ──────────────────────────────────────────────────────────────

class SearchAgent:
    def run(self, sub_questions: list[str]) -> list[Document]:
        """Retrieve and rerank documents for each sub-question."""
        all_docs: list[Document] = []

        for q in sub_questions:
            raw = []
            raw.extend(_duckduckgo(q, n=MAX_SEARCH_RESULTS))
            if USE_TAVILY and TAVILY_API_KEY:
                raw.extend(_tavily(q, n=MAX_SEARCH_RESULTS))
            raw.extend(_wikipedia(q))
            raw.extend(_arxiv(q, n=2))
            ranked = _rerank(raw, q)
            all_docs.extend(ranked[:MAX_DOCS_PER_QUERY])

        # global deduplication by URL
        seen, unique = set(), []
        for d in all_docs:
            if d.url not in seen:
                seen.add(d.url)
                unique.append(d)

        return unique


# ── LangGraph node wrapper ────────────────────────────────────────────────────

_agent = None

def search_node(state: ResearchState) -> ResearchState:
    """LangGraph node: retrieve documents for all sub-questions."""
    global _agent
    if state.get("error"):
        return state
    if _agent is None:
        _agent = SearchAgent()
    try:
        docs = _agent.run(state.get("sub_questions", []))
        return {**state, "raw_docs": docs}
    except Exception as e:
        return {**state, "error": f"Search failed: {e}", "raw_docs": []}
