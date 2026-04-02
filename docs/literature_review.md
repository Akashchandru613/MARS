# Literature Review: Multi-Agent LLM Frameworks

**Project:** MARS — Multi-Agent Research System  
**Authors:** Akash Chandru, Abhishek Tuteja  
**Date:** April 2026  

---

## 1. Overview

This review examines three dominant frameworks for building multi-agent LLM systems:
AutoGen, CrewAI, and LangGraph. The goal is to justify our choice of LangGraph for MARS
and document the trade-offs considered.

---

## 2. AutoGen (Microsoft Research, 2023)

**Paper:** "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"  
**Source:** arXiv:2308.08155

### How It Works
AutoGen models agents as conversational participants. Agents communicate by sending
messages to each other in a chat-like loop. The framework supports:
- `AssistantAgent`: an LLM-backed agent that responds to messages
- `UserProxyAgent`: executes code and relays user input
- Group chat with a `GroupChatManager` that decides who speaks next

### Strengths
- Very easy to set up for code generation and debugging tasks
- Natural fit for back-and-forth reasoning patterns
- Built-in code execution sandbox

### Weaknesses for MARS
- **No explicit shared state:** data passes through message strings, making structured
  data flow fragile
- **Non-deterministic routing:** the group chat manager decides agent order via LLM,
  which is hard to control and debug
- **Poor fit for sequential pipelines:** AutoGen shines in conversational loops, not
  directed data processing graphs

### Verdict: Not chosen
AutoGen's conversational paradigm is a poor fit for MARS, which needs a strict
linear pipeline with typed data passing between stages.

---

## 3. CrewAI (2024)

**Source:** https://github.com/joaomdmoura/crewAI

### How It Works
CrewAI organizes agents into a "crew" — each agent has a `role`, `goal`, and
`backstory` defined in natural language. Tasks are assigned to agents, and the
crew runs them sequentially or in parallel under a `Process.sequential` or
`Process.hierarchical` mode.

### Strengths
- Extremely developer-friendly API — minimal boilerplate
- Role-based abstraction maps well to human team metaphors
- Good documentation and fast-growing community

### Weaknesses for MARS
- **Black-box state management:** data flow between agents is implicit; no typed
  shared state schema
- **Limited graph control:** sequential and hierarchical modes only — no conditional
  branching or loop support without significant hacking
- **Prompt injection risk:** agent goals defined in free-form text can bleed into
  each other in unexpected ways
- **Less transparent:** harder to inspect intermediate outputs for debugging and evaluation

### Verdict: Not chosen
CrewAI's abstraction level is too high for a research system where we need full
visibility into every intermediate state for evaluation purposes.

---

## 4. LangGraph (LangChain Inc., 2024)

**Source:** https://github.com/langchain-ai/langgraph  
**Docs:** https://langchain-ai.github.io/langgraph/

### How It Works
LangGraph models the agent pipeline as a directed graph where:
- **Nodes** are Python functions (each implementing one agent's logic)
- **Edges** define the flow of execution between nodes
- **State** is a typed Python `TypedDict` that persists across all nodes

The `StateGraph` abstraction gives developers full programmatic control over:
- Which agent runs in what order
- What data each agent can read and write
- Conditional branching (e.g., retry on error)
- Checkpointing for long-running workflows

### Strengths for MARS
| Feature | Benefit |
|---|---|
| Typed `ResearchState` dict | All agents share a strongly-typed schema — no data loss between stages |
| Explicit edges | The pipeline is deterministic and auditable |
| Node isolation | Each agent is a pure Python function — easy to test independently |
| Conditional edges | Enables retry logic and error routing without framework hacks |
| Built-in checkpointing | Resume interrupted runs without re-running expensive API calls |
| Composability | Easy to add or remove agents without rewriting the entire pipeline |

### Weaknesses
- Steeper learning curve than CrewAI
- Requires more boilerplate than AutoGen for simple use cases

### Verdict: **CHOSEN**
LangGraph provides the fine-grained control over state management and agent
communication that MARS requires. The typed `ResearchState` is central to our
evaluation methodology — every intermediate output is inspectable, serializable,
and reproducible.

---

## 5. Decision Summary

| Criterion | AutoGen | CrewAI | LangGraph |
|---|---|---|---|
| Typed shared state | No | No | **Yes** |
| Deterministic agent ordering | No | Partial | **Yes** |
| Inspectable intermediate outputs | Hard | Hard | **Yes** |
| Conditional routing / error handling | Limited | Limited | **Yes** |
| Ease of independent agent testing | Low | Medium | **High** |
| Fit for evaluation / benchmarking | Poor | Medium | **High** |
| Learning curve | Low | Low | Medium |

**LangGraph was chosen unanimously** because MARS is fundamentally an experimental
system — we need full observability and reproducibility at every step, and LangGraph's
StateGraph abstraction is the only framework that gives us that.

---

## 6. Related Work

- **ReAct (Yao et al., 2022):** Interleaving reasoning and acting in LLMs — foundational
  for agent tool use.
- **Toolformer (Schick et al., 2023):** LLMs learning to use tools self-supervised.
- **Chain-of-Thought Prompting (Wei et al., 2022):** Multi-step reasoning as the
  basis for agent decomposition strategies.
- **FActScorer (Min et al., 2023):** Fine-grained atomic fact evaluation — directly
  informs our fact-checking agent design.
- **RAGAS (Es et al., 2023):** Evaluation framework for RAG systems — we adapt
  their faithfulness metric for our hallucination rate computation.
