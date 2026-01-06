"""Synthetic dataset generator for temporal decay benchmarking.

Generates query-memory pairs with temporal scenarios to test whether
temporal decay improves retrieval quality.
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from litellm import acompletion


@dataclass
class TemporalTestCase:
    """A single test case for temporal decay evaluation.

    Attributes:
        query: The search query
        scenario: Type of temporal scenario being tested
        memories: List of memories with temporal metadata
        expected_ranking: Expected order of memory UUIDs after temporal scoring
        notes: Human-readable explanation of what this tests
    """

    query: str
    scenario: str  # "old_but_relevant", "fresh_update", "superseded", "no_temporal_effect"
    memories: list[dict[str, Any]]
    expected_ranking: list[str]  # UUIDs in expected order
    notes: str = ""


@dataclass
class TemporalDataset:
    """Collection of test cases for temporal decay benchmarking."""

    test_cases: list[TemporalTestCase] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        data = {
            "metadata": self.metadata,
            "test_cases": [
                {
                    "query": tc.query,
                    "scenario": tc.scenario,
                    "memories": tc.memories,
                    "expected_ranking": tc.expected_ranking,
                    "notes": tc.notes,
                }
                for tc in self.test_cases
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TemporalDataset":
        """Load dataset from JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            metadata=data.get("metadata", {}),
            test_cases=[
                TemporalTestCase(
                    query=tc["query"],
                    scenario=tc["scenario"],
                    memories=tc["memories"],
                    expected_ranking=tc["expected_ranking"],
                    notes=tc.get("notes", ""),
                )
                for tc in data["test_cases"]
            ],
        )


class TemporalDatasetGenerator:
    """Generates synthetic test cases for temporal decay evaluation.

    Creates adversarial test cases that force a tradeoff between
    vector similarity and temporal relevance.
    """

    SCENARIOS = {
        "fresh_update": {
            "description": "New memory updates old fact - temporal should promote new",
            "expected_behavior": "newer should rank higher despite lower vector similarity",
        },
        "old_but_relevant": {
            "description": "Old memory is still correct - should not be over-penalized",
            "expected_behavior": "old memory should still rank well due to decay floor",
        },
        "superseded": {
            "description": "Old memory has been explicitly replaced",
            "expected_behavior": "superseding memory should rank much higher",
        },
        "new_but_irrelevant": {
            "description": "Recent memory doesn't match query - recency shouldn't help",
            "expected_behavior": "older relevant memory should beat newer irrelevant one",
        },
        "no_temporal_effect": {
            "description": "All memories are recent - temporal shouldn't change ranking",
            "expected_behavior": "ranking should match pure vector similarity",
        },
    }

    def __init__(self, model: str = "gemini/gemini-2.0-flash-lite"):
        """Initialize generator.

        Args:
            model: LLM model for generating queries and variations
        """
        self.model = model

    async def generate_query_from_memory(self, memory_content: str) -> str:
        """Generate a natural query that this memory would answer.

        Args:
            memory_content: The memory text content

        Returns:
            A natural language query
        """
        prompt = f"""Generate a natural search query that someone would use to find this information.
The query should be 5-15 words, as if typed into a search box.

Memory content:
{memory_content[:500]}

Return ONLY the query, nothing else."""

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip().strip('"')

    async def generate_paraphrase(self, content: str) -> str:
        """Generate a paraphrased version of content.

        Args:
            content: Original content

        Returns:
            Paraphrased version
        """
        prompt = f"""Paraphrase this text to say the same thing differently.
Keep the meaning but change the wording.

Original:
{content[:500]}

Return ONLY the paraphrase, nothing else."""

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    async def generate_update(self, content: str) -> str:
        """Generate an updated/corrected version of content.

        Args:
            content: Original content

        Returns:
            Updated version that supersedes the original
        """
        prompt = f"""Generate an update to this technical fact/decision that changes or corrects it.
The update should be plausible and reference the change.

Original:
{content[:500]}

Return ONLY the update, starting with something like "Update:" or "Changed to:" or "Switched to:"."""

        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.8,
        )
        return response.choices[0].message.content.strip()

    def create_fresh_update_case(
        self,
        base_memory: dict[str, Any],
        updated_memory: dict[str, Any],
        query: str,
    ) -> TemporalTestCase:
        """Create a test case where new memory should rank higher.

        Args:
            base_memory: Original memory (older, exact match)
            updated_memory: Updated memory (newer, paraphrase)
            query: Search query

        Returns:
            Test case expecting updated memory to rank first
        """
        # Ensure temporal ordering
        now = int(time.time())
        base_memory["created_at"] = now - 30 * 86400  # 30 days ago
        updated_memory["created_at"] = now - 2 * 86400  # 2 days ago

        return TemporalTestCase(
            query=query,
            scenario="fresh_update",
            memories=[base_memory, updated_memory],
            expected_ranking=[updated_memory["uuid"], base_memory["uuid"]],
            notes="Newer update should rank higher despite base having higher vector similarity",
        )

    def create_old_but_relevant_case(
        self,
        memory: dict[str, Any],
        irrelevant_recent: dict[str, Any],
        query: str,
    ) -> TemporalTestCase:
        """Create a test case where old memory should still rank well.

        Args:
            memory: Relevant but old memory
            irrelevant_recent: Recent but irrelevant memory
            query: Search query

        Returns:
            Test case expecting old memory to rank first
        """
        now = int(time.time())
        memory["created_at"] = now - 120 * 86400  # 120 days ago
        irrelevant_recent["created_at"] = now - 1 * 86400  # 1 day ago

        return TemporalTestCase(
            query=query,
            scenario="old_but_relevant",
            memories=[memory, irrelevant_recent],
            expected_ranking=[memory["uuid"], irrelevant_recent["uuid"]],
            notes="Old but relevant memory should beat recent irrelevant one due to decay floor",
        )

    def create_no_temporal_effect_case(
        self,
        memories: list[dict[str, Any]],
        query: str,
        vector_ranking: list[str],
    ) -> TemporalTestCase:
        """Create a test case where all memories are recent.

        Args:
            memories: List of memories, all recent
            query: Search query
            vector_ranking: Expected ranking by pure vector similarity

        Returns:
            Test case expecting vector ranking to be preserved
        """
        now = int(time.time())
        for m in memories:
            # All within last 2 days - minimal temporal effect
            m["created_at"] = now - random.randint(0, 2) * 86400

        return TemporalTestCase(
            query=query,
            scenario="no_temporal_effect",
            memories=memories,
            expected_ranking=vector_ranking,
            notes="All recent memories - temporal decay should not significantly change vector ranking",
        )

    async def generate_from_real_memories(
        self,
        memories: list[dict[str, Any]],
        n_cases: int = 50,
    ) -> TemporalDataset:
        """Generate test cases from real memory data.

        Args:
            memories: Real memories from the database
            n_cases: Number of test cases to generate

        Returns:
            TemporalDataset with generated test cases
        """
        test_cases = []
        cases_per_scenario = n_cases // len(self.SCENARIOS)

        for scenario in self.SCENARIOS:
            for _ in range(cases_per_scenario):
                if len(memories) < 2:
                    break

                # Sample memories for this test case
                sampled = random.sample(memories, min(3, len(memories)))

                if scenario == "fresh_update":
                    base = sampled[0].copy()
                    query = await self.generate_query_from_memory(base["content"])
                    updated_content = await self.generate_update(base["content"])
                    updated = {
                        "uuid": f"synthetic-{int(time.time())}-{random.randint(0, 9999)}",
                        "content": updated_content,
                        "type": base.get("type", "fact"),
                    }
                    test_cases.append(
                        self.create_fresh_update_case(base, updated, query)
                    )

                elif scenario == "old_but_relevant":
                    relevant = sampled[0].copy()
                    irrelevant = sampled[1].copy() if len(sampled) > 1 else sampled[0].copy()
                    query = await self.generate_query_from_memory(relevant["content"])
                    test_cases.append(
                        self.create_old_but_relevant_case(relevant, irrelevant, query)
                    )

                elif scenario == "no_temporal_effect":
                    mems = [m.copy() for m in sampled]
                    query = await self.generate_query_from_memory(mems[0]["content"])
                    # Assume vector ranking matches order in sampled (first is most relevant)
                    vector_ranking = [m["uuid"] for m in mems]
                    test_cases.append(
                        self.create_no_temporal_effect_case(mems, query, vector_ranking)
                    )

        return TemporalDataset(
            test_cases=test_cases,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "n_source_memories": len(memories),
                "n_test_cases": len(test_cases),
                "scenarios": list(self.SCENARIOS.keys()),
            },
        )

    def generate_handcrafted_cases(self) -> TemporalDataset:
        """Generate handcrafted adversarial test cases.

        These are carefully designed to force tradeoffs between
        vector similarity and temporal relevance.

        Returns:
            TemporalDataset with handcrafted test cases
        """
        test_cases = []
        now = int(time.time())

        # Case 1: Caching backend change
        # OLD memory has HIGHER vector similarity (0.88) but is 120 days old
        # NEW memory has LOWER vector similarity (0.82) but is only 1 day old
        # With decision half-life of 180 days, 120-day decay ~ 0.63
        # Expected: Temporal decay should promote the newer memory
        test_cases.append(
            TemporalTestCase(
                query="What caching backend do we use?",
                scenario="fresh_update",
                memories=[
                    {
                        "uuid": "cache-old",
                        "content": "Decision: Use Redis for caching. It provides fast in-memory storage with persistence options.",
                        "type": "decision",
                        "created_at": now - 120 * 86400,  # 120 days old - significant decay
                        "score": 0.88,  # High similarity - exact term match
                    },
                    {
                        "uuid": "cache-new",
                        "content": "Update: Switched to Memcached for the caching layer due to simpler scaling model and lower memory overhead.",
                        "type": "decision",
                        "created_at": now - 1 * 86400,  # 1 day old - fresh
                        "score": 0.82,  # Lower similarity - paraphrased
                    },
                ],
                expected_ranking=["cache-new", "cache-old"],
                notes="120-day old decision vs 1-day old - temporal should overcome 6% vector difference",
            )
        )

        # Case 2: Database connection issue
        # OLD memory has HIGHER vector similarity (0.86) - exact match on "database timeout"
        # NEW memory has LOWER vector similarity (0.80) - uses "DB connection" paraphrase
        # lesson_learned has 90-day half-life, so 90-day old = 0.5 decay
        test_cases.append(
            TemporalTestCase(
                query="database connection timeout error",
                scenario="fresh_update",
                memories=[
                    {
                        "uuid": "db-old-fix",
                        "content": "Fixed database timeout by increasing connection pool size from 5 to 20.",
                        "type": "lesson_learned",
                        "created_at": now - 90 * 86400,  # 90 days = half-life
                        "score": 0.86,  # High similarity - exact terms
                    },
                    {
                        "uuid": "db-new-fix",
                        "content": "Resolved DB connection issues by switching to connection pooling with PgBouncer instead of direct connections.",
                        "type": "lesson_learned",
                        "created_at": now - 2 * 86400,  # 2 days old
                        "score": 0.80,  # Lower similarity - paraphrased
                    },
                ],
                expected_ranking=["db-new-fix", "db-old-fix"],
                notes="90-day old lesson vs 2-day old - temporal should overcome 6% vector difference",
            )
        )

        # Case 3: Old architectural decision still valid
        # OLD memory has HIGH vector similarity (0.95) - exact match on "PostgreSQL"
        # NEW memory has LOW vector similarity (0.45) - tangentially related
        # Expected: Decay floor should keep old decision ranked higher
        test_cases.append(
            TemporalTestCase(
                query="why do we use PostgreSQL",
                scenario="old_but_relevant",
                memories=[
                    {
                        "uuid": "pg-decision",
                        "content": "Decision: Use PostgreSQL for the primary database. Reason: ACID compliance, JSON support, excellent query optimizer, and strong community.",
                        "type": "decision",
                        "created_at": now - 180 * 86400,  # 6 months ago
                        "score": 0.95,  # Very high - direct answer
                    },
                    {
                        "uuid": "pg-unrelated",
                        "content": "Added new index on users table for faster login queries.",
                        "type": "fact",
                        "created_at": now - 2 * 86400,
                        "score": 0.45,  # Low - tangentially related (database operations)
                    },
                ],
                expected_ranking=["pg-decision", "pg-unrelated"],
                notes="Old decision is still the canonical answer, shouldn't be penalized below unrelated recent fact",
            )
        )

        # Case 4: API version deprecation
        # OLD memory has HIGHER vector similarity (0.88) - exact match on "API authentication"
        # NEW memory has LOWER vector similarity (0.82) - mentions "OAuth 2.0"
        # fact has 60-day half-life, so 60-day old = 0.5 decay
        test_cases.append(
            TemporalTestCase(
                query="how to authenticate API requests",
                scenario="fresh_update",
                memories=[
                    {
                        "uuid": "auth-v1",
                        "content": "API authentication uses API key in X-API-Key header. Keys are generated in the developer portal.",
                        "type": "fact",
                        "created_at": now - 60 * 86400,  # 60 days = half-life
                        "score": 0.88,  # High - exact "API authentication" match
                    },
                    {
                        "uuid": "auth-v2",
                        "content": "Update: Migrated to OAuth 2.0 for API authentication. API keys deprecated. Use Bearer token in Authorization header.",
                        "type": "fact",
                        "created_at": now - 2 * 86400,  # 2 days old
                        "score": 0.82,  # Lower - uses different terminology
                    },
                ],
                expected_ranking=["auth-v2", "auth-v1"],
                notes="60-day old fact vs 2-day old - temporal should overcome 6% vector difference",
            )
        )

        # Case 5: Recent irrelevant noise
        # OLD memory has HIGH vector similarity (0.94) - directly about rate limiter
        # NEW memory has LOW vector similarity (0.55) - only mentions "rate limiter" in passing
        # Expected: Relevance (high vector score) should beat recency
        test_cases.append(
            TemporalTestCase(
                query="how does the rate limiter work",
                scenario="new_but_irrelevant",
                memories=[
                    {
                        "uuid": "rate-limiter",
                        "content": "Rate limiting implemented using token bucket algorithm with Redis backend. 100 requests/minute per IP.",
                        "type": "pattern",
                        "created_at": now - 45 * 86400,
                        "score": 0.94,  # Very high - exact topic match
                    },
                    {
                        "uuid": "recent-deploy",
                        "content": "Deployed hotfix for login page CSS issue. No rate limiter changes.",
                        "type": "session_summary",
                        "created_at": now - 1 * 86400,
                        "score": 0.55,  # Low - only tangentially mentions topic
                    },
                ],
                expected_ranking=["rate-limiter", "recent-deploy"],
                notes="Recent memory mentions 'rate limiter' but is irrelevant - relevance beats recency",
            )
        )

        # Case 6: All recent - no temporal effect
        # Both memories are recent (1-2 days) so temporal should have minimal impact
        # Vector similarity should dominate ranking
        test_cases.append(
            TemporalTestCase(
                query="error handling patterns",
                scenario="no_temporal_effect",
                memories=[
                    {
                        "uuid": "err-pattern-1",
                        "content": "Pattern: Use Result<T, E> for fallible operations. Never throw exceptions from library code.",
                        "type": "pattern",
                        "created_at": now - 1 * 86400,
                        "score": 0.88,  # Higher similarity
                    },
                    {
                        "uuid": "err-pattern-2",
                        "content": "Always log errors with context before re-raising.",
                        "type": "lesson_learned",
                        "created_at": now - 2 * 86400,
                        "score": 0.82,  # Lower similarity
                    },
                ],
                expected_ranking=["err-pattern-1", "err-pattern-2"],
                notes="Both recent - vector similarity should dominate",
            )
        )

        return TemporalDataset(
            test_cases=test_cases,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "n_test_cases": len(test_cases),
                "type": "handcrafted_adversarial",
            },
        )


# CLI entry point for generating datasets
if __name__ == "__main__":
    import asyncio

    async def main():
        generator = TemporalDatasetGenerator()

        # Generate handcrafted cases
        dataset = generator.generate_handcrafted_cases()

        # Save to file
        output_path = Path(__file__).parent / "temporal_test_dataset.json"
        dataset.save(output_path)
        print(f"Generated {len(dataset.test_cases)} test cases to {output_path}")

    asyncio.run(main())
