"""LLM-as-Judge evaluator for retrieval quality.

Uses an LLM to compare two retrieval result sets and determine
which better answers the query.

CRITICAL: Implements position bias control by randomizing
presentation order and hiding system labels.
"""

import random
from dataclasses import dataclass
from typing import Any, Literal

from litellm import acompletion


@dataclass
class JudgmentResult:
    """Result of LLM judge comparison."""

    winner: Literal["a", "b", "tie"]
    confidence: float  # 0-1 scale
    reasoning: str
    # Internal tracking (not shown to judge)
    actual_winner: str | None = None  # "vector" or "temporal" after de-randomizing
    presentation_order: str | None = None  # "vector_first" or "temporal_first"


@dataclass
class RelevanceScore:
    """Single-output relevance score."""

    score: float  # 1-5 scale
    reasoning: str


class LLMJudge:
    """LLM-based judge for retrieval quality evaluation.

    Implements best practices from research:
    - Position bias control via randomization
    - No system labels shown to judge
    - Structured output format
    - Confidence calibration
    """

    PAIRWISE_PROMPT = '''You are evaluating search results for relevance to a query.
You will see two sets of results (Set A and Set B) and must determine which better answers the query.

## Query
{query}

## Set A Results
{results_a}

## Set B Results
{results_b}

## Evaluation Criteria
1. **Relevance**: Do the results directly answer the query?
2. **Accuracy**: Is the information current and correct? (consider if results mention updates or changes)
3. **Completeness**: Do the results provide sufficient context?
4. **Ranking Quality**: Are the most relevant results ranked first?

## Instructions
Compare the two sets and respond with EXACTLY this JSON format:
{{
    "winner": "a" or "b" or "tie",
    "confidence": 0.5 to 1.0,
    "reasoning": "Brief explanation of why this set is better"
}}

Only respond with the JSON, nothing else.'''

    RELEVANCE_PROMPT = '''Rate how well this search result answers the query.

## Query
{query}

## Result
{result}

## Rating Scale
1 = Completely irrelevant
2 = Tangentially related but doesn't answer query
3 = Partially relevant, missing key information
4 = Mostly relevant, answers the query
5 = Highly relevant, directly and completely answers the query

## Instructions
Respond with EXACTLY this JSON format:
{{
    "score": 1-5,
    "reasoning": "Brief explanation of the score"
}}

Only respond with the JSON, nothing else.'''

    def __init__(self, model: str = "gemini/gemini-2.0-flash-lite"):
        """Initialize judge.

        Args:
            model: LLM model for judging
        """
        self.model = model

    def _format_results(self, results: list[dict[str, Any]], max_results: int = 5) -> str:
        """Format results for presentation to judge.

        Args:
            results: List of memory results
            max_results: Maximum results to show

        Returns:
            Formatted string
        """
        formatted = []
        for i, r in enumerate(results[:max_results], 1):
            content = r.get("content", "")[:400]
            if len(r.get("content", "")) > 400:
                content += "..."

            mem_type = r.get("type", "unknown")
            formatted.append(f"[{i}] (type: {mem_type})\n{content}")

        return "\n\n".join(formatted) if formatted else "(no results)"

    async def judge_pairwise(
        self,
        query: str,
        results_vector: list[dict[str, Any]],
        results_temporal: list[dict[str, Any]],
    ) -> JudgmentResult:
        """Compare two result sets using pairwise comparison.

        CRITICAL: Randomizes presentation order to avoid position bias.
        Does NOT show system labels (vector vs temporal) to the judge.

        Args:
            query: The search query
            results_vector: Results from pure vector scoring
            results_temporal: Results from temporal-weighted scoring

        Returns:
            JudgmentResult with winner and reasoning
        """
        # CRITICAL: Randomize order to avoid position bias
        if random.random() < 0.5:
            results_a = results_vector
            results_b = results_temporal
            presentation_order = "vector_first"
        else:
            results_a = results_temporal
            results_b = results_vector
            presentation_order = "temporal_first"

        prompt = self.PAIRWISE_PROMPT.format(
            query=query,
            results_a=self._format_results(results_a),
            results_b=self._format_results(results_b),
        )

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,  # Low temperature for consistent judgments
            )

            # Parse JSON response
            import json
            result_text = response.choices[0].message.content.strip()
            # Handle markdown code blocks
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_data = json.loads(result_text)

            winner_raw = result_data.get("winner", "tie").lower()
            confidence = float(result_data.get("confidence", 0.5))
            reasoning = result_data.get("reasoning", "")

            # De-randomize to get actual winner
            if winner_raw == "tie":
                actual_winner = "tie"
            elif presentation_order == "vector_first":
                actual_winner = "vector" if winner_raw == "a" else "temporal"
            else:
                actual_winner = "temporal" if winner_raw == "a" else "vector"

            return JudgmentResult(
                winner=winner_raw,
                confidence=confidence,
                reasoning=reasoning,
                actual_winner=actual_winner,
                presentation_order=presentation_order,
            )

        except Exception as e:
            return JudgmentResult(
                winner="tie",
                confidence=0.0,
                reasoning=f"Judge error: {e}",
                actual_winner="error",
                presentation_order=presentation_order,
            )

    async def score_relevance(
        self,
        query: str,
        result: dict[str, Any],
    ) -> RelevanceScore:
        """Score a single result's relevance to the query.

        Args:
            query: The search query
            result: A single memory result

        Returns:
            RelevanceScore with 1-5 rating
        """
        content = result.get("content", "")[:600]
        if len(result.get("content", "")) > 600:
            content += "..."

        result_str = f"Type: {result.get('type', 'unknown')}\nContent: {content}"

        prompt = self.RELEVANCE_PROMPT.format(
            query=query,
            result=result_str,
        )

        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            import json
            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_data = json.loads(result_text)

            return RelevanceScore(
                score=float(result_data.get("score", 3)),
                reasoning=result_data.get("reasoning", ""),
            )

        except Exception as e:
            return RelevanceScore(
                score=0.0,
                reasoning=f"Scoring error: {e}",
            )

    async def score_result_set(
        self,
        query: str,
        results: list[dict[str, Any]],
        k: int = 5,
    ) -> list[RelevanceScore]:
        """Score relevance for top-k results.

        Args:
            query: The search query
            results: List of memory results
            k: Number of results to score

        Returns:
            List of RelevanceScore for each result
        """
        scores = []
        for result in results[:k]:
            score = await self.score_relevance(query, result)
            scores.append(score)
        return scores


# Multi-judge aggregation for robustness
class MultiJudgePanel:
    """Panel of multiple judges for more robust evaluation.

    Uses multiple judge models and aggregates their decisions
    through voting.
    """

    def __init__(
        self,
        models: list[str] | None = None,
    ):
        """Initialize panel.

        Args:
            models: List of model names to use as judges
        """
        self.models = models or [
            "gemini/gemini-2.0-flash-lite",
            "gpt-4o-mini",
        ]
        self.judges = [LLMJudge(model=m) for m in self.models]

    async def judge_with_consensus(
        self,
        query: str,
        results_vector: list[dict[str, Any]],
        results_temporal: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Get consensus judgment from multiple judges.

        Args:
            query: The search query
            results_vector: Results from pure vector scoring
            results_temporal: Results from temporal-weighted scoring

        Returns:
            Aggregated judgment with individual judge results
        """
        import asyncio

        # Run all judges in parallel
        tasks = [
            judge.judge_pairwise(query, results_vector, results_temporal)
            for judge in self.judges
        ]
        results = await asyncio.gather(*tasks)

        # Aggregate votes
        votes = {"vector": 0, "temporal": 0, "tie": 0, "error": 0}
        for r in results:
            if r.actual_winner in votes:
                votes[r.actual_winner] += 1

        # Determine consensus
        if votes["temporal"] > votes["vector"]:
            consensus = "temporal"
        elif votes["vector"] > votes["temporal"]:
            consensus = "vector"
        else:
            consensus = "tie"

        # Calculate confidence (agreement ratio)
        total_valid = votes["vector"] + votes["temporal"] + votes["tie"]
        if total_valid > 0:
            confidence = max(votes.values()) / total_valid
        else:
            confidence = 0.0

        return {
            "consensus": consensus,
            "confidence": confidence,
            "votes": votes,
            "individual_results": [
                {
                    "model": self.models[i],
                    "winner": r.actual_winner,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for i, r in enumerate(results)
            ],
        }
