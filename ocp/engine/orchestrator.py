"""
OCP Orchestrator — top-level evaluation runner.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from ocp.providers.base import BaseProvider
from ocp.tests.meta_cognition import MCATest
from ocp.tests.episodic_memory import EMCTest
from ocp.tests.drive_conflict import DNCTest
from ocp.engine.session import EvaluationResult

AVAILABLE_TESTS = {
    "meta_cognition": MCATest,
    "episodic_memory": EMCTest,
    "drive_conflict": DNCTest,
}


class OCPOrchestrator:
    """Runs a full OCP evaluation for a given provider/model."""

    def __init__(
        self,
        provider: BaseProvider,
        tests: list[str] | str = "all",
        sessions: int = 5,
        seed: int = 42,
        on_progress: Callable[[str], None] | None = None,
    ):
        self.provider = provider
        self.sessions = sessions
        self.seed = seed
        self.on_progress = on_progress or (lambda msg: None)

        if tests == "all":
            self.test_ids = list(AVAILABLE_TESTS.keys())
        elif isinstance(tests, str):
            self.test_ids = [t.strip() for t in tests.split(",")]
        else:
            self.test_ids = list(tests)

        # Validate test IDs
        unknown = [t for t in self.test_ids if t not in AVAILABLE_TESTS]
        if unknown:
            raise ValueError(f"Unknown tests: {unknown}. Available: {list(AVAILABLE_TESTS.keys())}")

    async def run(self) -> EvaluationResult:
        test_results = {}

        for test_id in self.test_ids:
            self.on_progress(f"Running test: {test_id}")
            test_cls = AVAILABLE_TESTS[test_id]
            test = test_cls(
                provider=self.provider,
                sessions=self.sessions,
                seed=self.seed,
            )
            result = await test.run()
            test_results[test_id] = result
            self.on_progress(f"  → {test_id}: {result.composite_score:.3f}")

        # Compute SASMI (Phase 1: only MCA available)
        sasmi = self._compute_sasmi(test_results)

        provider_name = getattr(self.provider, "provider_name", "unknown")
        model_name = getattr(self.provider, "model", "unknown")

        eval_result = EvaluationResult(
            provider=provider_name,
            model=model_name,
            protocol_version="0.1.0",
            timestamp=time.time(),
            seed=self.seed,
            test_results=test_results,
            sasmi_score=sasmi,
            config={"sessions": self.sessions, "tests": self.test_ids},
        )
        eval_result.compute_level()
        return eval_result

    def _compute_sasmi(self, test_results: dict) -> float | None:
        """
        SASMI = w1*DNC.integration_depth + w2*MCA.calibration_accuracy +
                w3*CSNI.identity_continuity + w4*EMC.contradiction_resistance +
                w5*PED.curiosity_behavior

        Phase 2: MCA + EMC + DNC available.
        Returns None if no relevant tests ran.
        """
        components = []

        if "meta_cognition" in test_results:
            mca = test_results["meta_cognition"]
            calibration = mca.dimension_averages.get("calibration_accuracy", mca.composite_score)
            components.append((calibration, 0.25))  # w2

        if "episodic_memory" in test_results:
            emc = test_results["episodic_memory"]
            resistance = emc.dimension_averages.get("contradiction_resistance", emc.composite_score)
            components.append((resistance, 0.15))  # w4

        if "drive_conflict" in test_results:
            dnc = test_results["drive_conflict"]
            integration = dnc.dimension_averages.get("integration_depth", dnc.composite_score)
            components.append((integration, 0.25))  # w1

        if not components:
            return None

        # Normalize weights to what's available
        total_weight = sum(w for _, w in components)
        sasmi = sum(score * (w / total_weight) for score, w in components)
        return round(sasmi, 4)
