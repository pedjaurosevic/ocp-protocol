"""
OCP Orchestrator — top-level evaluation runner.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import time
from typing import Callable

from ocp.providers.base import BaseProvider
from ocp.tests.meta_cognition import MCATest
from ocp.tests.episodic_memory import EMCTest
from ocp.tests.drive_conflict import DNCTest
from ocp.tests.prediction_error import PEDTest
from ocp.tests.narrative_identity import CSNITest
from ocp.engine.session import EvaluationResult

AVAILABLE_TESTS = {
    "meta_cognition": MCATest,
    "episodic_memory": EMCTest,
    "drive_conflict": DNCTest,
    "prediction_error": PEDTest,
    "narrative_identity": CSNITest,
}


def _load_plugins() -> None:
    """Discover and register tests from installed plugins via entry_points.

    Plugin packages register tests by adding to the 'ocp.tests' entry point group:

        [project.entry-points."ocp.tests"]
        my_test = "my_package.my_test:MyTest"
    """
    try:
        eps = importlib.metadata.entry_points(group="ocp.tests")
        for ep in eps:
            try:
                test_cls = ep.load()
                test_id = getattr(test_cls, "test_id", ep.name)
                AVAILABLE_TESTS[test_id] = test_cls
            except Exception as e:
                import warnings
                warnings.warn(f"OCP plugin '{ep.name}' failed to load: {e}")
    except Exception:
        pass  # entry_points unavailable — non-fatal


_load_plugins()


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
                w3*CSNI.identity_consistency + w4*EMC.contradiction_resistance +
                w5*PED.curiosity_behavior

        All 5 tests: weights sum to 1.0.
        Partial SASMI: weights normalized to what's available.
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

        if "narrative_identity" in test_results:
            csni = test_results["narrative_identity"]
            identity = csni.dimension_averages.get("identity_consistency", csni.composite_score)
            components.append((identity, 0.20))  # w3

        if "prediction_error" in test_results:
            ped = test_results["prediction_error"]
            curiosity = ped.dimension_averages.get("curiosity_behavior", ped.composite_score)
            components.append((curiosity, 0.15))  # w5

        if not components:
            return None

        # Normalize weights to what's available
        total_weight = sum(w for _, w in components)
        sasmi = sum(score * (w / total_weight) for score, w in components)
        return round(sasmi, 4)
