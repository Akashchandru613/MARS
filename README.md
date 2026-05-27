# MARS — Multi-Agent Research System

**Authors:** Akash Chandru 
**Phase:** 1 of 3 (Planning & Prototyping)  
**Status:** Single-agent baseline complete · Multi-agent pipeline in development

---

## What Is MARS?

MARS is a multi-agent research assistant where specialized LLM agents for
**Search**, **Summarization**, and **Fact-Checking** collaborate under an
**Orchestrator**, benchmarked against a single-agent baseline to evaluate the
impact of agent specialization on research quality and accuracy.

```
User Query → Orchestrator → Search → Summarize → Fact-Check → Report
```

**Hypothesis:** Specialized agents produce more accurate, better-cited, and
less hallucinatory research outputs than a single generalist agent performing
all tasks in one prompt.

---

## Project Structure

```
MARS/
├── README.md
├── requirements.txt
├── .env.example              ← copy to .env and add your Groq API key
├── config.py                 ← central configuration
│
├── baseline/
│   └── single_agent.py       ← WORKING: single-agent control condition
│
├── agents/
│   ├── orchestrator.py       ← query decomposition + report compilation
│   ├── search.py             ← DuckDuckGo + Wikipedia + arXiv retrieval
│   ├── summarizer.py         ← map-reduce summarization
│   └── fact_checker.py       ← per-claim verification
│
├── pipeline/
│   ├── state.py              ← shared ResearchState schema
│   ├── graph.py              ← LangGraph StateGraph definition
│   └── runner.py             ← pipeline entry point
│
├── evaluation/
│   ├── metrics.py            ← ROUGE, hallucination rate, source coverage
│   └── benchmark.py          ← benchmark runner
│
├── data/
│   ├── benchmark_queries.json ← 10-query evaluation set
│   └── results/              ← output JSONs (gitignored)
│
├── docs/
│   ├── literature_review.md  ← AutoGen vs CrewAI vs LangGraph analysis
│   └── architecture.md       ← full system design document
│
└── notebooks/
    └── evaluation.ipynb      ← results dashboard (Phase 3)
```

---

## Quick Start

### 1. Clone and install
```bash
git clone <your-repo-url>
cd MARS
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your free Groq API key (https://console.groq.com)
```

### 3. Run the single-agent baseline (works right now)
```bash
python baseline/single_agent.py "What caused the 2008 financial crisis?"
```

### 4. Run the full multi-agent pipeline (Phase 2)
```bash
python pipeline/runner.py "What caused the 2008 financial crisis?"
```

### 5. Run the benchmark
```bash
python evaluation/benchmark.py --dry-run   # 3 queries (fast test)
python evaluation/benchmark.py             # full 10-query benchmark
```

---

## Cost

**$0.** The entire project uses:
- **Groq** (free tier) — Llama 3.1 70B inference
- **DuckDuckGo** — no API key required
- **Wikipedia API** — free
- **arXiv API** — free
- All Python libraries — open source

---

## Team

| Member | Role |
|---|---|
| Akash Chandru | Orchestrator Agent · Search Agent · LangGraph pipeline · Integration |
| Abhishek Tuteja | Summarization Agent · Fact-Checking Agent · Benchmark · Evaluation |

---

## Milestones

| Phase | Goal | Status |
|---|---|---|
| Phase 1 | Architecture, literature review, baseline | April 1 — Done |
| Phase 2 | All agents built and tested independently | ~April 15 |
| Phase 3 | Integration, benchmark, evaluation, demo | ~April 28 |
