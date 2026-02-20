"""
OCP Evaluation Engine — orchestrates tests and computes final scores.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ocp.tests.base import TestResult


@dataclass
class EvaluationResult:
    provider: str
    model: str
    protocol_version: str
    timestamp: float
    seed: int
    test_results: dict[str, TestResult]
    sasmi_score: Optional[float] = None
    phi_star: Optional[float] = None
    gwt_score: Optional[float] = None
    nii: Optional[float] = None
    nii_label: Optional[str] = None
    ocp_level: Optional[int] = None
    ocp_level_name: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    scale_details: dict[str, Any] = field(default_factory=dict)

    def compute_level(self) -> None:
        """Compute OCP level using all available Layer 2 scales.

        Per spec:
          OCP-5: SASMI > 0.80 AND Φ* > 0.80 AND GWT > 0.70 AND NII > 0.70
          OCP-4: SASMI 0.60–0.80 AND Φ* 0.60–0.80 AND GWT > 0.50 AND NII > 0.50
          OCP-3: SASMI 0.40–0.60 AND Φ* 0.40–0.60 AND GWT > 0.30
          OCP-2: SASMI 0.20–0.40 OR Φ* 0.20–0.40
          OCP-1: fallback
        """
        s = self.sasmi_score
        p = self.phi_star
        g = self.gwt_score
        n = self.nii

        if s is None:
            self.ocp_level = 1
            self.ocp_level_name = "Reactive"
            return

        # Full multi-scale certification (when all scales available)
        if p is not None and g is not None and n is not None:
            if s >= 0.80 and p >= 0.80 and g >= 0.70 and n >= 0.70:
                self.ocp_level, self.ocp_level_name = 5, "Autonomous Identity"
            elif s >= 0.60 and p >= 0.60 and g >= 0.50 and n >= 0.50:
                self.ocp_level, self.ocp_level_name = 4, "Self-Modeling"
            elif s >= 0.40 and p >= 0.40 and g >= 0.30:
                self.ocp_level, self.ocp_level_name = 3, "Integrated"
            elif s >= 0.20 or p >= 0.20:
                self.ocp_level, self.ocp_level_name = 2, "Patterned"
            else:
                self.ocp_level, self.ocp_level_name = 1, "Reactive"
            return

        # Partial certification (SASMI only — backward compatible)
        for level, name, threshold in [(5, "Autonomous Identity", 0.80),
                                        (4, "Self-Modeling", 0.60),
                                        (3, "Integrated", 0.40),
                                        (2, "Patterned", 0.20),
                                        (1, "Reactive", 0.0)]:
            if s >= threshold:
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
            "phi_star": round(self.phi_star, 4) if self.phi_star is not None else None,
            "gwt_score": round(self.gwt_score, 4) if self.gwt_score is not None else None,
            "nii": round(self.nii, 4) if self.nii is not None else None,
            "nii_label": self.nii_label,
            "test_results": {k: v.to_dict() for k, v in self.test_results.items()},
            "scale_details": self.scale_details,
            "config": self.config,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

