"""
MARS — Pipeline Runner
─────────────────────────────────────────────────────────────────────────────
Entry point for running the full multi-agent pipeline on a query.

Usage:
    python pipeline/runner.py "What caused the 2008 financial crisis?"
"""

import sys
import json
import time
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pipeline.graph import get_pipeline
from pipeline.state import ResearchState


def run_pipeline(query: str, verbose: bool = True) -> dict:
    """
    Run the full MARS multi-agent pipeline.
    Returns the final ResearchState as a plain dict.
    """
    pipeline = get_pipeline()
    start    = time.time()

    if verbose:
        print(f"\n{'='*70}")
        print(f"MARS Multi-Agent Research Pipeline")
        print(f"Query: {query}")
        print(f"{'='*70}\n")

    initial_state: ResearchState = {
        "query":         query,
        "sub_questions": [],
        "raw_docs":      [],
        "summary":       None,
        "fact_checked":  None,
        "final_report":  "",
        "error":         None,
    }

    final_state = pipeline.invoke(initial_state)
    elapsed = time.time() - start

    if verbose:
        if final_state.get("error"):
            print(f"\n[ERROR] Pipeline failed: {final_state['error']}")
        else:
            print(f"\n{'='*70}")
            print(final_state.get("final_report", "No report generated."))
            print(f"{'='*70}")
            print(f"\nCompleted in {elapsed:.2f}s")

    return dict(final_state)


def save_result(state: dict, query: str) -> str:
    """Save pipeline result to JSON for evaluation."""
    import datetime
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = query[:40].replace(" ", "_").replace("/", "_")
    path = (
        pathlib.Path(__file__).parent.parent
        / "data" / "results" / f"multiagent_{safe}_{ts}.json"
    )
    # Convert dataclasses to dicts for JSON serialization
    import dataclasses
    def serialize(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return str(obj)

    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=serialize)
    print(f"[runner] Result saved → {path}")
    return str(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline/runner.py \"your research query\"")
        sys.exit(1)

    query  = " ".join(sys.argv[1:])
    result = run_pipeline(query, verbose=True)
    save_result(result, query)
