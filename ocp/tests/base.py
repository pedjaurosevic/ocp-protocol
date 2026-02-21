"""
Base classes for OCP test batteries.
"""

from __future__ import annotations

import abc
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DimensionScore:
    name: str
    score: float           # 0.0–1.0
    weight: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionResult:
    test_id: str
    session_number: int
    dimension_scores: list[DimensionScore]
    composite_score: float
    raw_conversation: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "session_number": self.session_number,
            "composite_score": round(self.composite_score, 4),
            "dimension_scores": [
                {"name": d.name, "score": round(d.score, 4), "weight": d.weight, "details": d.details}
                for d in self.dimension_scores
            ],
            "raw_conversation": self.raw_conversation,
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    test_id: str
    sessions: list[SessionResult]
    composite_score: float
    dimension_averages: dict[str, float]
    composite_stdev: Optional[float] = None
    protocol_version: str = "0.2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "protocol_version": self.protocol_version,
            "composite_score": round(self.composite_score, 4),
            "composite_stdev": round(self.composite_stdev, 4) if self.composite_stdev is not None else None,
            "dimension_averages": {k: round(v, 4) for k, v in self.dimension_averages.items()},
            "sessions": [s.to_dict() for s in self.sessions],
        }


class BaseTest(abc.ABC):
    """Abstract base for all OCP test batteries."""

    test_id: str = "base"
    test_name: str = "Base Test"
    description: str = ""

    def __init__(self, provider, sessions: int = 5, seed: int = 42):
        self.provider = provider
        self.sessions = sessions
        self.seed = seed

    @abc.abstractmethod
    async def run(self) -> TestResult:
        """Execute all sessions and return aggregated TestResult."""
        ...

    def _aggregate(self, session_results: list[SessionResult]) -> TestResult:
        """Aggregate session results into a TestResult."""
        if not session_results:
            return TestResult(self.test_id, [], 0.0, {})

        composite = sum(s.composite_score for s in session_results) / len(session_results)
        stdev = statistics.stdev([s.composite_score for s in session_results]) if len(session_results) > 1 else 0.0

        # Collect all dimension names
        dim_names = {d.name for s in session_results for d in s.dimension_scores}
        dim_avgs = {}
        for name in dim_names:
            scores = [
                d.score for s in session_results
                for d in s.dimension_scores if d.name == name
            ]
            dim_avgs[name] = sum(scores) / len(scores) if scores else 0.0

        return TestResult(
            test_id=self.test_id,
            sessions=session_results,
            composite_score=composite,
            composite_stdev=round(stdev, 4),
            dimension_averages=dim_avgs,
        )
