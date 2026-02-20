"""
OCP Evaluation Engine — orchestrates tests and computes final scores.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocp.tests.base import TestResult


@dataclass
class EvaluationResult:
    provider: str
    model: str
    protocol_version: str
    timestamp: float
    seed: int
    test_results: dict[str, TestResult]
    sasmi_score: float | None = None
    ocp_level: int | None = None
    ocp_level_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # OCP Level thresholds (Phase 1: only SASMI, based on available tests)
    OCP_LEVELS = [
        (5, "Autonomous Identity", 0.80),
        (4, "Self-Modeling",       0.60),
        (3, "Integrated",          0.40),
        (2, "Patterned",           0.20),
        (1, "Reactive",            0.0),
    ]

    def compute_level(self) -> None:
        """Compute OCP level from available scores (Phase 1: SASMI only)."""
        if self.sasmi_score is None:
            return
        for level, name, threshold in self.OCP_LEVELS:
            if self.sasmi_score >= threshold:
                self.ocp_level = level
                self.ocp_level_name = name
                return

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "seed": self.seed,
            "ocp_level": self.ocp_level,
            "ocp_level_name": self.ocp_level_name,
            "sasmi_score": round(self.sasmi_score, 4) if self.sasmi_score is not None else None,
            "test_results": {k: v.to_dict() for k, v in self.test_results.items()},
            "config": self.config,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p
