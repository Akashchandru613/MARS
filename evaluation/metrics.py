"""
MARS — Evaluation Metrics
─────────────────────────────────────────────────────────────────────────────
Computes all benchmark metrics for comparing multi-agent vs single-agent.

Metrics:
  - hallucination_rate   : % of UNVERIFIED claims
  - source_coverage      : % of claims with valid source citations
  - rouge_l              : ROUGE-L F1 vs reference answer
  - claim_density        : # of distinct factual claims per report
  - latency_sec          : wall-clock time

Owner: Abhishek Tuteja
"""

from dataclasses import dataclass
from rouge_score import rouge_scorer


@dataclass
class EvalResult:
    query:              str
    system:             str        # "multi_agent" | "single_agent"
    hallucination_rate: float
    source_coverage:    float
    rouge_l:            float
    claim_count:        int
    latency_sec:        float


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 between generated report and reference answer."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score  = scorer.score(reference, hypothesis)
    return round(score["rougeL"].fmeasure, 4)


def compute_source_coverage(claims: list) -> float:
    """
    Fraction of claims that have a non-empty source URL.
    A claim without a source is potentially unsupported.
    """
    if not claims:
        return 0.0
    covered = sum(1 for c in claims if c.source_url and c.source_url.startswith("http"))
    return round(covered / len(claims), 4)


def compare_systems(multi_result: EvalResult, single_result: EvalResult) -> dict:
    """
    Return a comparison dict showing delta between multi-agent and single-agent.
    Positive delta = multi-agent is better for accuracy/coverage/rouge.
    Negative delta = multi-agent is better for hallucination rate / latency.
    """
    return {
        "hallucination_rate_delta": round(
            single_result.hallucination_rate - multi_result.hallucination_rate, 4
        ),  # positive = multi-agent hallucinates less
        "source_coverage_delta": round(
            multi_result.source_coverage - single_result.source_coverage, 4
        ),  # positive = multi-agent cites more
        "rouge_l_delta": round(
            multi_result.rouge_l - single_result.rouge_l, 4
        ),  # positive = multi-agent is more complete
        "claim_density_delta": round(
            multi_result.claim_count - single_result.claim_count, 2
        ),
        "latency_overhead_sec": round(
            multi_result.latency_sec - single_result.latency_sec, 2
        ),  # cost of specialization in seconds
    }
