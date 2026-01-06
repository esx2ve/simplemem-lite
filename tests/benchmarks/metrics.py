"""Retrieval evaluation metrics.

Implements standard IR metrics for evaluating temporal decay
scoring effectiveness.
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""

    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float  # Mean Reciprocal Rank
    win_rate: float  # % where temporal beats vector
    counterfactual_win_rate: float  # Of changes, % that were improvements
    top_1_change_rate: float  # % where top result changed
    n_queries: int


def compute_dcg(relevance_scores: list[float], k: int | None = None) -> float:
    """Compute Discounted Cumulative Gain.

    DCG = sum(rel_i / log2(i + 1)) for i in 1..k

    Args:
        relevance_scores: Relevance scores in ranked order (1-5 scale)
        k: Cutoff position (None = use all)

    Returns:
        DCG value
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        # Position is 1-indexed, so i+1
        # log2(i+2) because we want log2(position+1) = log2(i+1+1)
        dcg += rel / math.log2(i + 2)

    return dcg


def compute_ndcg(
    relevance_scores: list[float],
    k: int | None = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    NDCG = DCG / IDCG where IDCG is DCG with perfect ranking.

    Args:
        relevance_scores: Relevance scores in ranked order (1-5 scale)
        k: Cutoff position (None = use all)

    Returns:
        NDCG value in [0, 1]
    """
    if not relevance_scores:
        return 0.0

    dcg = compute_dcg(relevance_scores, k)

    # Ideal DCG: scores sorted in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = compute_dcg(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_mrr(
    results: list[dict[str, Any]],
    relevant_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank.

    MRR = 1 / rank of first relevant result

    Args:
        results: Ranked results with 'uuid' field
        relevant_ids: Set of relevant memory UUIDs

    Returns:
        MRR value in (0, 1] or 0 if no relevant results
    """
    for i, r in enumerate(results):
        if r.get("uuid") in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_win_rate(
    judgments: list[dict[str, Any]],
    system_name: str = "temporal",
) -> float:
    """Compute win rate for a system.

    Args:
        judgments: List of judgment dicts with 'actual_winner' field
        system_name: Name of system to compute win rate for

    Returns:
        Win rate as fraction in [0, 1]
    """
    if not judgments:
        return 0.0

    wins = sum(1 for j in judgments if j.get("actual_winner") == system_name)
    return wins / len(judgments)


def compute_counterfactual_wins(
    judgments: list[dict[str, Any]],
    rankings_changed: list[bool],
) -> float:
    """Compute counterfactual win rate.

    Of the queries where rankings changed, what percentage
    were judged as improvements?

    Args:
        judgments: List of judgment dicts with 'actual_winner' field
        rankings_changed: Boolean for each query if rankings changed

    Returns:
        Counterfactual win rate in [0, 1]
    """
    if len(judgments) != len(rankings_changed):
        raise ValueError("judgments and rankings_changed must have same length")

    changed_indices = [i for i, changed in enumerate(rankings_changed) if changed]
    if not changed_indices:
        return 0.0

    improvements = sum(
        1 for i in changed_indices
        if judgments[i].get("actual_winner") == "temporal"
    )
    return improvements / len(changed_indices)


def compute_top_1_change_rate(
    results_vector: list[list[dict[str, Any]]],
    results_temporal: list[list[dict[str, Any]]],
) -> float:
    """Compute how often the top-1 result changed.

    Args:
        results_vector: List of vector result lists (one per query)
        results_temporal: List of temporal result lists (one per query)

    Returns:
        Change rate in [0, 1]
    """
    if len(results_vector) != len(results_temporal):
        raise ValueError("Must have same number of query results")

    if not results_vector:
        return 0.0

    changes = 0
    for vec_results, temp_results in zip(results_vector, results_temporal):
        vec_top1 = vec_results[0].get("uuid") if vec_results else None
        temp_top1 = temp_results[0].get("uuid") if temp_results else None
        if vec_top1 != temp_top1:
            changes += 1

    return changes / len(results_vector)


def rankings_differ(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
    k: int = 5,
) -> bool:
    """Check if two rankings differ in top-k.

    Args:
        results_a: First result list
        results_b: Second result list
        k: Number of top results to compare

    Returns:
        True if rankings differ
    """
    uuids_a = [r.get("uuid") for r in results_a[:k]]
    uuids_b = [r.get("uuid") for r in results_b[:k]]
    return uuids_a != uuids_b


class MetricsAggregator:
    """Aggregates metrics across multiple queries."""

    def __init__(self):
        self.ndcg_5_values: list[float] = []
        self.ndcg_10_values: list[float] = []
        self.mrr_values: list[float] = []
        self.judgments: list[dict[str, Any]] = []
        self.rankings_changed: list[bool] = []

    def add_query_result(
        self,
        vector_relevance_scores: list[float],
        temporal_relevance_scores: list[float],
        judgment: dict[str, Any],
        ranking_changed: bool,
    ) -> None:
        """Add results for a single query.

        Args:
            vector_relevance_scores: Relevance scores for vector results (1-5)
            temporal_relevance_scores: Relevance scores for temporal results (1-5)
            judgment: Judge's comparison result
            ranking_changed: Whether rankings differed
        """
        # NDCG for temporal system (the one we're evaluating)
        self.ndcg_5_values.append(compute_ndcg(temporal_relevance_scores, k=5))
        self.ndcg_10_values.append(compute_ndcg(temporal_relevance_scores, k=10))

        # Store judgment
        self.judgments.append(judgment)
        self.rankings_changed.append(ranking_changed)

    def compute_aggregate(self) -> EvaluationMetrics:
        """Compute aggregate metrics across all queries.

        Returns:
            EvaluationMetrics with aggregated values
        """
        n = len(self.judgments)
        if n == 0:
            return EvaluationMetrics(
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                mrr=0.0,
                win_rate=0.0,
                counterfactual_win_rate=0.0,
                top_1_change_rate=0.0,
                n_queries=0,
            )

        return EvaluationMetrics(
            ndcg_at_5=sum(self.ndcg_5_values) / n if self.ndcg_5_values else 0.0,
            ndcg_at_10=sum(self.ndcg_10_values) / n if self.ndcg_10_values else 0.0,
            mrr=sum(self.mrr_values) / n if self.mrr_values else 0.0,
            win_rate=compute_win_rate(self.judgments, "temporal"),
            counterfactual_win_rate=compute_counterfactual_wins(
                self.judgments, self.rankings_changed
            ),
            top_1_change_rate=sum(self.rankings_changed) / n,
            n_queries=n,
        )

    def print_summary(self) -> None:
        """Print formatted summary of metrics."""
        metrics = self.compute_aggregate()
        print("\n" + "=" * 50)
        print("TEMPORAL DECAY EVALUATION RESULTS")
        print("=" * 50)
        print(f"Queries evaluated: {metrics.n_queries}")
        print("-" * 50)
        print(f"NDCG@5:                    {metrics.ndcg_at_5:.3f}")
        print(f"NDCG@10:                   {metrics.ndcg_at_10:.3f}")
        print(f"Win rate (temporal):       {metrics.win_rate:.1%}")
        print(f"Top-1 change rate:         {metrics.top_1_change_rate:.1%}")
        print(f"Counterfactual win rate:   {metrics.counterfactual_win_rate:.1%}")
        print("=" * 50)

        # Pass/fail based on success criteria
        print("\nSUCCESS CRITERIA:")
        passed = 0
        total = 3

        if metrics.win_rate > 0.5:
            print("  [PASS] Win rate > 50%")
            passed += 1
        else:
            print(f"  [FAIL] Win rate > 50% (got {metrics.win_rate:.1%})")

        if metrics.ndcg_at_5 >= 0.7:  # Assuming baseline is similar
            print("  [PASS] NDCG@5 >= 0.7 (no regression)")
            passed += 1
        else:
            print(f"  [WARN] NDCG@5 < 0.7 (got {metrics.ndcg_at_5:.3f})")
            passed += 0.5

        if metrics.counterfactual_win_rate > 0.6:
            print("  [PASS] Counterfactual wins > 60%")
            passed += 1
        else:
            print(f"  [FAIL] Counterfactual wins > 60% (got {metrics.counterfactual_win_rate:.1%})")

        print(f"\nOverall: {passed}/{total} criteria met")
