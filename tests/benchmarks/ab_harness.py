"""A/B evaluation harness for temporal decay scoring.

Runs side-by-side comparison of vector-only vs temporal-weighted
retrieval and evaluates using LLM-as-judge.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .generate_temporal_dataset import TemporalDataset, TemporalTestCase
from .llm_judge import LLMJudge, MultiJudgePanel
from .metrics import (
    EvaluationMetrics,
    MetricsAggregator,
    compute_ndcg,
    rankings_differ,
)


@dataclass
class ScorerResult:
    """Result from a scoring function."""

    results: list[dict[str, Any]]
    elapsed_ms: float


@dataclass
class QueryComparison:
    """Side-by-side comparison for a single query."""

    query: str
    scenario: str
    vector_results: list[dict[str, Any]]
    temporal_results: list[dict[str, Any]]
    judgment: dict[str, Any]
    vector_relevance_scores: list[float]
    temporal_relevance_scores: list[float]
    ranking_changed: bool


@dataclass
class ABResults:
    """Aggregated results from A/B comparison."""

    metrics: EvaluationMetrics
    comparisons: list[QueryComparison]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics": {
                "ndcg_at_5": self.metrics.ndcg_at_5,
                "ndcg_at_10": self.metrics.ndcg_at_10,
                "mrr": self.metrics.mrr,
                "win_rate": self.metrics.win_rate,
                "counterfactual_win_rate": self.metrics.counterfactual_win_rate,
                "top_1_change_rate": self.metrics.top_1_change_rate,
                "n_queries": self.metrics.n_queries,
            },
            "comparisons": [
                {
                    "query": c.query,
                    "scenario": c.scenario,
                    "judgment": c.judgment,
                    "ranking_changed": c.ranking_changed,
                }
                for c in self.comparisons
            ],
            "metadata": self.metadata,
        }


class ABHarness:
    """A/B testing harness for retrieval scoring functions.

    Compares two scoring approaches:
    - Baseline (vector-only): Pure cosine similarity
    - Treatment (temporal): Vector + temporal decay + importance

    Uses LLM-as-judge for pairwise comparison and computes
    standard IR metrics (NDCG, win-rate, etc.)
    """

    def __init__(
        self,
        judge: LLMJudge | None = None,
        use_multi_judge: bool = False,
        verbose: bool = True,
    ):
        """Initialize harness.

        Args:
            judge: LLM judge instance (creates default if None)
            use_multi_judge: Use multi-model consensus (slower but more robust)
            verbose: Print progress during evaluation
        """
        if use_multi_judge:
            self.judge_panel = MultiJudgePanel()
            self.judge = None
        else:
            self.judge = judge or LLMJudge()
            self.judge_panel = None

        self.verbose = verbose

    def _score_with_vector_only(
        self,
        memories: list[dict[str, Any]],
        query: str,  # noqa: ARG002 - reserved for future use
    ) -> list[dict[str, Any]]:
        """Score memories using only vector similarity.

        This simulates the baseline system that uses pure cosine similarity.

        Args:
            memories: List of memory dicts with 'score' field
            query: The search query (unused, for interface consistency)

        Returns:
            Memories sorted by vector score descending
        """
        # Memories should already have vector similarity scores
        # Higher score = more similar
        scored = []
        for m in memories:
            scored.append({
                **m,
                "final_score": m.get("score", 0.0),
                "scoring_method": "vector_only",
            })

        return sorted(scored, key=lambda x: -x["final_score"])

    def _score_with_temporal(
        self,
        memories: list[dict[str, Any]],
        query: str,  # noqa: ARG002 - reserved for future use
        vector_weight: float = 0.6,
        temporal_weight: float = 0.25,
        importance_weight: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Score memories using temporal decay + importance.

        This is the treatment system we're evaluating.

        Args:
            memories: List of memory dicts with 'score' and 'created_at'
            query: The search query (unused, for interface consistency)
            vector_weight: Weight for normalized vector similarity
            temporal_weight: Weight for temporal decay factor
            importance_weight: Weight for type-based importance

        Returns:
            Memories sorted by combined score descending
        """
        from math import exp, log

        # Half-life per memory type (days)
        HALF_LIVES = {
            "decision": 180,
            "lesson_learned": 90,
            "pattern": 120,
            "fact": 60,
            "session_summary": 45,
            "chunk_summary": 30,
        }

        # Importance prior per type
        IMPORTANCE_PRIOR = {
            "decision": 0.9,
            "lesson_learned": 0.7,
            "pattern": 0.6,
            "fact": 0.5,
            "session_summary": 0.4,
            "chunk_summary": 0.3,
        }

        # Decay floors - minimum temporal factor
        DECAY_FLOORS = {
            "decision": 0.4,
            "lesson_learned": 0.3,
            "pattern": 0.35,
            "fact": 0.4,
            "session_summary": 0.2,
            "chunk_summary": 0.1,
        }

        now = datetime.now(timezone.utc)

        # CRITICAL: Normalize vector scores within this result set
        # Use rank-preserving normalization (divide by max) to preserve relative ratios
        # This prevents distortion with small result sets where min-max would set
        # lowest score to 0.0 regardless of actual value
        vector_scores = [m.get("score", 0.0) for m in memories]
        max_score = max(vector_scores) if vector_scores else 1.0

        scored = []
        for m in memories:
            # Normalize vector score to [0, 1] using rank-preserving method
            raw_score = m.get("score", 0.0)
            if max_score > 1e-9:
                vector_normalized = raw_score / max_score
            else:
                vector_normalized = 1.0  # All same score

            # Compute temporal decay
            mem_type = m.get("type", "fact")
            created_at = m.get("created_at")
            if isinstance(created_at, (int, float)):
                created_dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
            elif isinstance(created_at, datetime):
                created_dt = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
            else:
                # Default to now if no timestamp
                created_dt = now

            age_days = (now - created_dt).total_seconds() / 86400
            half_life = HALF_LIVES.get(mem_type, 90)
            floor = DECAY_FLOORS.get(mem_type, 0.1)
            temporal_decay = max(floor, exp(-log(2) * age_days / half_life))

            # Get importance prior
            importance = IMPORTANCE_PRIOR.get(mem_type, 0.5)

            # Compute combined score
            final_score = (
                vector_weight * vector_normalized +
                temporal_weight * temporal_decay +
                importance_weight * importance
            )

            scored.append({
                **m,
                "score_normalized": vector_normalized,
                "temporal_decay": temporal_decay,
                "importance_prior": importance,
                "final_score": final_score,
                "scoring_method": "temporal_weighted",
            })

        return sorted(scored, key=lambda x: -x["final_score"])

    async def _judge_comparison(
        self,
        query: str,
        vector_results: list[dict[str, Any]],
        temporal_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get LLM judgment on which results are better.

        Args:
            query: The search query
            vector_results: Results from vector-only scoring
            temporal_results: Results from temporal scoring

        Returns:
            Judgment dict with winner, confidence, reasoning
        """
        if self.judge_panel:
            return await self.judge_panel.judge_with_consensus(
                query, vector_results, temporal_results
            )
        else:
            result = await self.judge.judge_pairwise(
                query, vector_results, temporal_results
            )
            return {
                "winner": result.actual_winner,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "presentation_order": result.presentation_order,
            }

    async def _score_relevance(
        self,
        query: str,
        results: list[dict[str, Any]],
        k: int = 5,
    ) -> list[float]:
        """Score relevance of top-k results using LLM.

        Args:
            query: The search query
            results: Ranked results
            k: Number of results to score

        Returns:
            List of relevance scores (1-5 scale)
        """
        if not self.judge:
            # Skip if using multi-judge panel (too slow)
            return [3.0] * min(k, len(results))

        scores = await self.judge.score_result_set(query, results, k)
        return [s.score for s in scores]

    async def evaluate_test_case(
        self,
        test_case: TemporalTestCase,
    ) -> QueryComparison:
        """Evaluate a single test case.

        Args:
            test_case: Test case with query and memories

        Returns:
            QueryComparison with results and judgment
        """
        query = test_case.query
        memories = test_case.memories

        # Simulate vector scores based on how well content matches query
        # In real usage, these would come from the vector DB
        for m in memories:
            if "score" not in m:
                # Simple heuristic: count query term overlap
                query_terms = set(query.lower().split())
                content_terms = set(m.get("content", "").lower().split())
                overlap = len(query_terms & content_terms)
                m["score"] = 0.5 + (overlap * 0.1)  # Base 0.5 + term bonus

        # Run both scorers
        vector_results = self._score_with_vector_only(memories, query)
        temporal_results = self._score_with_temporal(memories, query)

        # Check if rankings changed
        ranking_changed = rankings_differ(vector_results, temporal_results, k=5)

        # Get LLM judgment
        judgment = await self._judge_comparison(query, vector_results, temporal_results)

        # Score relevance for NDCG
        vector_relevance = await self._score_relevance(query, vector_results, k=10)
        temporal_relevance = await self._score_relevance(query, temporal_results, k=10)

        return QueryComparison(
            query=query,
            scenario=test_case.scenario,
            vector_results=vector_results,
            temporal_results=temporal_results,
            judgment=judgment,
            vector_relevance_scores=vector_relevance,
            temporal_relevance_scores=temporal_relevance,
            ranking_changed=ranking_changed,
        )

    async def run_evaluation(
        self,
        dataset: TemporalDataset,
        max_queries: int | None = None,
    ) -> ABResults:
        """Run full A/B evaluation on dataset.

        Args:
            dataset: Dataset of test cases
            max_queries: Maximum queries to evaluate (None = all)

        Returns:
            ABResults with metrics and comparisons
        """
        test_cases = dataset.test_cases
        if max_queries:
            test_cases = test_cases[:max_queries]

        if self.verbose:
            print(f"\nRunning A/B evaluation on {len(test_cases)} test cases...")
            print("-" * 50)

        comparisons = []
        aggregator = MetricsAggregator()

        for i, tc in enumerate(test_cases):
            if self.verbose:
                print(f"[{i+1}/{len(test_cases)}] {tc.scenario}: {tc.query[:50]}...")

            comparison = await self.evaluate_test_case(tc)
            comparisons.append(comparison)

            # Add to aggregator
            aggregator.add_query_result(
                vector_relevance_scores=comparison.vector_relevance_scores,
                temporal_relevance_scores=comparison.temporal_relevance_scores,
                judgment={"actual_winner": comparison.judgment.get("winner", "tie")},
                ranking_changed=comparison.ranking_changed,
            )

            if self.verbose:
                winner = comparison.judgment.get("winner", "tie")
                changed = "CHANGED" if comparison.ranking_changed else "same"
                print(f"         -> Winner: {winner}, Ranking: {changed}")

        metrics = aggregator.compute_aggregate()

        if self.verbose:
            aggregator.print_summary()

        return ABResults(
            metrics=metrics,
            comparisons=comparisons,
            metadata={
                "evaluated_at": datetime.now().isoformat(),
                "n_test_cases": len(test_cases),
                "scenarios": list(set(tc.scenario for tc in test_cases)),
            },
        )

    async def run_with_custom_scorers(
        self,
        dataset: TemporalDataset,
        scorer_a: Callable[[list[dict], str], list[dict]],
        scorer_b: Callable[[list[dict], str], list[dict]],
        scorer_a_name: str = "baseline",
        scorer_b_name: str = "treatment",
        max_queries: int | None = None,
    ) -> ABResults:
        """Run A/B evaluation with custom scoring functions.

        Allows comparing any two scoring approaches, not just
        vector-only vs temporal.

        Args:
            dataset: Dataset of test cases
            scorer_a: First scoring function (memories, query) -> ranked results
            scorer_b: Second scoring function (memories, query) -> ranked results
            scorer_a_name: Name for scorer A
            scorer_b_name: Name for scorer B
            max_queries: Maximum queries to evaluate

        Returns:
            ABResults with metrics and comparisons
        """
        test_cases = dataset.test_cases
        if max_queries:
            test_cases = test_cases[:max_queries]

        if self.verbose:
            print(f"\nComparing {scorer_a_name} vs {scorer_b_name}")
            print(f"Running on {len(test_cases)} test cases...")
            print("-" * 50)

        comparisons = []
        aggregator = MetricsAggregator()

        for i, tc in enumerate(test_cases):
            if self.verbose:
                print(f"[{i+1}/{len(test_cases)}] {tc.scenario}: {tc.query[:50]}...")

            query = tc.query
            memories = tc.memories

            # Run both scorers
            results_a = scorer_a(memories, query)
            results_b = scorer_b(memories, query)

            # Check if rankings changed
            ranking_changed = rankings_differ(results_a, results_b, k=5)

            # Get LLM judgment (using results_a as "vector" and results_b as "temporal"
            # for internal tracking, but labels are hidden from judge)
            judgment = await self._judge_comparison(query, results_a, results_b)

            # Remap winner to scorer names
            actual_winner = judgment.get("winner", "tie")
            if actual_winner == "vector":
                actual_winner = scorer_a_name
            elif actual_winner == "temporal":
                actual_winner = scorer_b_name
            judgment["winner"] = actual_winner

            # Score relevance
            relevance_a = await self._score_relevance(query, results_a, k=10)
            relevance_b = await self._score_relevance(query, results_b, k=10)

            comparison = QueryComparison(
                query=query,
                scenario=tc.scenario,
                vector_results=results_a,
                temporal_results=results_b,
                judgment=judgment,
                vector_relevance_scores=relevance_a,
                temporal_relevance_scores=relevance_b,
                ranking_changed=ranking_changed,
            )
            comparisons.append(comparison)

            # Add to aggregator (treatment is scorer_b)
            aggregator.add_query_result(
                vector_relevance_scores=relevance_a,
                temporal_relevance_scores=relevance_b,
                judgment={"actual_winner": "temporal" if actual_winner == scorer_b_name else actual_winner},
                ranking_changed=ranking_changed,
            )

            if self.verbose:
                changed = "CHANGED" if ranking_changed else "same"
                print(f"         -> Winner: {actual_winner}, Ranking: {changed}")

        metrics = aggregator.compute_aggregate()

        if self.verbose:
            aggregator.print_summary()

        return ABResults(
            metrics=metrics,
            comparisons=comparisons,
            metadata={
                "evaluated_at": datetime.now().isoformat(),
                "n_test_cases": len(test_cases),
                "scorer_a": scorer_a_name,
                "scorer_b": scorer_b_name,
            },
        )


# CLI entry point for running evaluation
if __name__ == "__main__":
    from .generate_temporal_dataset import TemporalDatasetGenerator

    async def main():
        # Generate handcrafted test cases
        generator = TemporalDatasetGenerator()
        dataset = generator.generate_handcrafted_cases()

        print(f"Generated {len(dataset.test_cases)} test cases")

        # Run A/B evaluation
        harness = ABHarness(verbose=True)
        results = await harness.run_evaluation(dataset)

        # Print success criteria
        print("\n" + "=" * 50)
        print("SUCCESS CRITERIA CHECK")
        print("=" * 50)

        criteria_met = 0
        total_criteria = 3

        if results.metrics.win_rate > 0.5:
            print("[PASS] Win rate > 50%")
            criteria_met += 1
        else:
            print(f"[FAIL] Win rate > 50% (got {results.metrics.win_rate:.1%})")

        if results.metrics.ndcg_at_5 >= 0.7:
            print("[PASS] NDCG@5 >= 0.7")
            criteria_met += 1
        else:
            print(f"[WARN] NDCG@5 >= 0.7 (got {results.metrics.ndcg_at_5:.3f})")

        if results.metrics.counterfactual_win_rate > 0.6:
            print("[PASS] Counterfactual wins > 60%")
            criteria_met += 1
        else:
            print(f"[FAIL] Counterfactual wins > 60% (got {results.metrics.counterfactual_win_rate:.1%})")

        print(f"\nOverall: {criteria_met}/{total_criteria} criteria met")

        if criteria_met == total_criteria:
            print("\n*** TEMPORAL DECAY VALIDATED - Ready to ship! ***")
        else:
            print("\n*** Needs tuning before shipping ***")

    asyncio.run(main())
