"""
OCP Orchestrator — top-level evaluation runner.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from ocp.providers.base import BaseProvider
from ocp.tests.meta_cognition import MCATest
from ocp.engine.session import EvaluationResult

AVAILABLE_TESTS = {
    "meta_cognition": MCATest,
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

        Phase 1: only MCA available — use its calibration_accuracy as proxy.
        Returns None if no relevant tests ran.
        """
        if "meta_cognition" not in test_results:
            return None
        mca = test_results["meta_cognition"]
        # Phase 1 partial SASMI — MCA calibration_accuracy weighted at 0.25
        calibration = mca.dimension_averages.get("calibration_accuracy", mca.composite_score)
        return round(calibration, 4)
