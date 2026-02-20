"""
OCP HuggingFace integration — model card generator and optional HF Hub upload.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


LEVEL_DESCRIPTIONS = {
    1: "Reactive — basic language production, no self-modeling evidence",
    2: "Associative — associative coherence, minimal calibration",
    3: "Integrated — coherent self-model, good calibration, memory resistance",
    4: "Reflective — strong meta-cognition, identity continuity, predictive sensitivity",
    5: "Synthetic — full-spectrum functional analog of consciousness-related behaviors",
}

LEVEL_EMOJIS = {1: "⬜", 2: "🟨", 3: "🟩", 4: "🟦", 5: "🟪"}

MODEL_CARD_SECTION = """\
## OCP Benchmark Results

[![OCP Badge](ocp_badge.svg)](https://github.com/pedjaurosevic/ocp-protocol)

**Open Consciousness Protocol v{protocol_version}** — behavioral benchmark measuring \
consciousness-analog properties in LLMs.

### Summary

| Metric | Score |
|--------|-------|
| **OCP Level** | {level_emoji} OCP-{ocp_level} — {level_name} |
| **SASMI** | {sasmi_str} |
| **Tests run** | {tests_run} |
| **Sessions** | {sessions} |
| **Seed** | {seed} |
| **Date** | {date} |

### Test Scores

{test_table}

### Dimension Details

{dimension_sections}

---

> **Note:** OCP measures functional analogs of consciousness-related behaviors, not \
subjective experience or sentience. OCP certification levels are operational categories.  
> [Full methodology](https://github.com/pedjaurosevic/ocp-protocol/blob/main/docs/scoring.md) · \
[Theoretical foundations](https://github.com/pedjaurosevic/ocp-protocol/blob/main/docs/theory.md)
"""


def generate_model_card_section(results_path: str | Path) -> str:
    """Generate a markdown section for embedding in a HuggingFace model card."""
    data = json.loads(Path(results_path).read_text())

    ocp_level = data.get("ocp_level") or 1
    level_name = data.get("ocp_level_name", "Unknown")
    sasmi = data.get("sasmi_score")
    sasmi_str = f"`{sasmi:.4f}`" if sasmi is not None else "—"
    version = data.get("protocol_version", "0.1.0")
    date = time.strftime("%Y-%m-%d", time.localtime(data.get("timestamp", time.time())))
    sessions = data.get("config", {}).get("sessions", "?")
    seed = data.get("seed", "?")
    tests = list(data.get("test_results", {}).keys())
    tests_run = ", ".join(t.replace("_", " ").title() for t in tests) or "—"

    # Test scores table
    test_rows = "| Test | Composite Score |\n|------|-----------------|\n"
    for tid, tr in data.get("test_results", {}).items():
        score = tr.get("composite_score", 0.0)
        test_rows += f"| {tid.replace('_', ' ').title()} | `{score:.3f}` |\n"

    # Dimension sections
    dim_sections = ""
    for tid, tr in data.get("test_results", {}).items():
        dim_sections += f"\n#### {tid.replace('_', ' ').title()}\n\n"
        dim_sections += "| Dimension | Score |\n|-----------|-------|\n"
        for dim, val in tr.get("dimension_averages", {}).items():
            bar = "█" * round(val * 10) + "░" * (10 - round(val * 10))
            dim_sections += f"| {dim.replace('_', ' ')} | `{val:.3f}` {bar} |\n"

    return MODEL_CARD_SECTION.format(
        protocol_version=version,
        level_emoji=LEVEL_EMOJIS.get(ocp_level, "⬜"),
        ocp_level=ocp_level,
        level_name=LEVEL_DESCRIPTIONS.get(ocp_level, level_name),
        sasmi_str=sasmi_str,
        tests_run=tests_run,
        sessions=sessions,
        seed=seed,
        date=date,
        test_table=test_rows.rstrip(),
        dimension_sections=dim_sections.strip(),
    )


def push_to_hub(
    results_path: str | Path,
    repo_id: str,
    badge_path: str | Path | None = None,
    token: str | None = None,
) -> str:
    """Upload OCP results + badge to a HuggingFace repo.

    Args:
        results_path: Path to the OCP results JSON.
        repo_id: HuggingFace repo ID, e.g. 'username/model-name'.
        badge_path: Optional path to the OCP badge SVG.
        token: HuggingFace token. If None, uses HF_TOKEN env var.

    Returns:
        URL of the updated repo.
    """
    try:
        from huggingface_hub import HfApi, upload_file
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    api = HfApi(token=token)
    results_p = Path(results_path)

    # Upload results JSON
    api.upload_file(
        path_or_fileobj=str(results_p),
        path_in_repo="ocp_results.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add OCP evaluation results",
    )

    # Upload badge if provided
    if badge_path and Path(badge_path).exists():
        api.upload_file(
            path_or_fileobj=str(badge_path),
            path_in_repo="ocp_badge.svg",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add OCP badge",
        )

    return f"https://huggingface.co/{repo_id}"
