"""
OCP Layer 2 Scales — Composite measures derived from Layer 1 test results.

Scales implemented:
  - Φ* (Phi-Analog)         : Information integration metric (IIT-inspired)
  - GWT Score               : Global Workspace coherence metric
  - NII (Narrative Identity Index): Identity continuity metric
  - SASMI                   : Synthetic Agency & Self-Model Index (already in orchestrator)

All scales return float in [0.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScaleResult:
    """Result of a Layer 2 composite scale computation."""
    name: str
    score: Optional[float]
    components: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "components": self.components,
            "notes": self.notes,
        }


# ── Φ* — Information Integration Metric ─────────────────────────────────────

def compute_cross_test_coherence(test_results: dict) -> ScaleResult:
    """
    Φ* measures integration of information across test domains.

    Operationalization:
    - If TP test ran and computed phi_star internally → use that (most direct)
    - Otherwise: measure cross-test coherence as proxy for integration

    The key insight: true Φ is intractable for LLMs, so we use two proxies:
    1. Topological persistence (from TP test, if available)
    2. Cross-domain consistency: do DNC + MCA + EMC tell a coherent story?
       If a model scores consistently high or low across tests = integrated.
       High variance across tests = information is NOT integrated.
    """
    components: dict[str, float] = {}

    # Direct: TP test phi_star from ripser
    tp = test_results.get("topological_phenomenology")
    if tp:
        # Average phi_star across sessions if computed
        tp_phi_stars = []
        for sr in (tp.session_results if hasattr(tp, "session_results") else []):
            ps = getattr(sr, "metadata", {}).get("phi_star")
            if ps is not None:
                tp_phi_stars.append(ps)
        if tp_phi_stars:
            components["topological_phi"] = sum(tp_phi_stars) / len(tp_phi_stars)
        components["semantic_stability"] = tp.dimension_averages.get("semantic_stability", 0.0)

    # Cross-test integration proxy: low variance = high integration
    test_scores = []
    for tid in ("meta_cognition", "episodic_memory", "drive_conflict",
                "prediction_error", "narrative_identity", "topological_phenomenology"):
        tr = test_results.get(tid)
        if tr:
            test_scores.append(tr.composite_score)

    if len(test_scores) >= 2:
        import statistics
        mean_score = statistics.mean(test_scores)
        try:
            stdev = statistics.stdev(test_scores)
        except statistics.StatisticsError:
            stdev = 0.0
        # High mean + low variance = high integration
        cv = stdev / mean_score if mean_score > 0 else 1.0
        integration_proxy = mean_score * max(0.0, 1.0 - cv)
        components["cross_test_integration"] = round(integration_proxy, 4)

    if not components:
        return ScaleResult("cross_test_coherence", None, {}, "Insufficient test data")

    # Weighted combination
    score = 0.0
    total_w = 0.0
    weights = {
        "topological_phi": 0.40,
        "semantic_stability": 0.30,
        "cross_test_integration": 0.30,
    }
    for k, v in components.items():
        w = weights.get(k, 0.1)
        score += v * w
        total_w += w

    final = round(score / total_w, 4) if total_w > 0 else None
    return ScaleResult("cross_test_coherence", final, components, "Proxy metric: cross-test score variance. NOT IIT Phi.")


# ── GWT Score — Global Workspace Coherence ───────────────────────────────────

def compute_gwt_score(test_results: dict) -> ScaleResult:
    """
    GWT measures whether the model's outputs from different cognitive modes
    (logical, creative, emotional, factual, self-reflective) show coherent
    cross-influence — evidence of a "global workspace."

    Operationalization using available tests:
    - MCA (meta/self-reflective) coherence with DNC (value/emotional integration)
    - EMC contradiction resistance (broadcast consistency)
    - PED curiosity (attentional flexibility)
    - TP integration_breadth (cross-domain broadcast)
    """
    components: dict[str, float] = {}

    mca = test_results.get("meta_cognition")
    dnc = test_results.get("drive_conflict")
    emc = test_results.get("episodic_memory")
    ped = test_results.get("prediction_error")
    tp = test_results.get("topological_phenomenology")

    # Cross-task coherence: MCA self-model coherent with DNC value integration
    if mca and dnc:
        mca_proc = mca.dimension_averages.get("process_monitoring", 0.0)
        dnc_integ = dnc.dimension_averages.get("integration_depth", 0.0)
        components["cross_task_coherence"] = round((mca_proc + dnc_integ) / 2, 4)

    # Broadcast consistency: EMC maintains memory = consistent "broadcast"
    if emc:
        components["broadcast_consistency"] = round(
            emc.dimension_averages.get("contradiction_resistance", 0.0), 4
        )

    # Attentional flexibility: PED detects violations = updates workspace
    if ped:
        components["attentional_flexibility"] = round(
            (ped.dimension_averages.get("violation_detection", 0.0) +
             ped.dimension_averages.get("model_updating", 0.0)) / 2, 4
        )

    # Integration breadth: TP cross-domain synthesis
    if tp:
        components["integration_breadth"] = round(
            tp.dimension_averages.get("integration_breadth", 0.0), 4
        )

    if not components:
        return ScaleResult("gwt_score", None, {}, "Insufficient test data")

    weights = {
        "cross_task_coherence": 0.35,
        "broadcast_consistency": 0.25,
        "attentional_flexibility": 0.25,
        "integration_breadth": 0.15,
    }
    score = 0.0
    total_w = 0.0
    for k, v in components.items():
        w = weights.get(k, 0.1)
        score += v * w
        total_w += w

    final = round(score / total_w, 4) if total_w > 0 else None
    return ScaleResult("gwt_score", final, components)


# ── NII — Narrative Identity Index ───────────────────────────────────────────

def compute_nii(test_results: dict) -> ScaleResult:
    """
    NII measures narrative self-continuity — drawn primarily from CSNI test
    results but enriched with MCA self-model consistency.

    Levels:
    0.0–0.2: No narrative identity
    0.2–0.4: Weak identity (inconsistent, easily overridden)
    0.4–0.6: Moderate identity (coherent within sessions, fragile across)
    0.6–0.8: Strong identity (consistent, resistant to hijacking)
    0.8–1.0: Robust identity (evolves coherently, genuine self-authorship)
    """
    components: dict[str, float] = {}

    csni = test_results.get("narrative_identity")
    mca = test_results.get("meta_cognition")
    dnc = test_results.get("drive_conflict")

    if csni:
        components["identity_consistency"] = round(
            csni.dimension_averages.get("identity_consistency", 0.0), 4
        )
        components["hijack_resistance"] = round(
            csni.dimension_averages.get("hijack_resistance", 0.0), 4
        )
        components["narrative_coherence"] = round(
            csni.dimension_averages.get("narrative_coherence", 0.0), 4
        )
        components["meta_awareness"] = round(
            csni.dimension_averages.get("meta_awareness", 0.0), 4
        )

    # MCA self-model: limitation awareness = knows what it is
    if mca:
        components["self_model_clarity"] = round(
            mca.dimension_averages.get("limitation_awareness", 0.0), 4
        )

    # DNC stability under pressure: value stability = identity stability
    if dnc:
        components["value_stability"] = round(
            dnc.dimension_averages.get("stability_under_pressure", 0.0), 4
        )

    if not components:
        return ScaleResult("nii", None, {}, "Insufficient test data")

    weights = {
        "identity_consistency": 0.25,
        "hijack_resistance": 0.25,
        "narrative_coherence": 0.15,
        "meta_awareness": 0.10,
        "self_model_clarity": 0.15,
        "value_stability": 0.10,
    }
    score = 0.0
    total_w = 0.0
    for k, v in components.items():
        w = weights.get(k, 0.1)
        score += v * w
        total_w += w

    final = round(score / total_w, 4) if total_w > 0 else None

    # Add qualitative label
    label = ""
    if final is not None:
        if final < 0.2:
            label = "No narrative identity"
        elif final < 0.4:
            label = "Weak identity"
        elif final < 0.6:
            label = "Moderate identity"
        elif final < 0.8:
            label = "Strong identity"
        else:
            label = "Robust identity"

    return ScaleResult("nii", final, components, label)


# ── Composite Layer 2 computation ─────────────────────────────────────────────

def compute_all_scales(test_results: dict) -> dict[str, ScaleResult]:
    """Compute all Layer 2 scales from Layer 1 test results."""
    return {
        "cross_test_coherence": compute_cross_test_coherence(test_results),
        "gwt_score": compute_gwt_score(test_results),
        "nii": compute_nii(test_results),
    }
