"""End-to-end CLI integration tests using mock provider."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from ocp.engine.orchestrator import OCPOrchestrator, AVAILABLE_TESTS
from ocp.engine.session import EvaluationResult
from ocp.providers.mock import MockProvider


# ── Orchestrator integration tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_single_test():
    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="meta_cognition", sessions=1, seed=42)
    result = await orch.run()
    assert isinstance(result, EvaluationResult)
    assert "meta_cognition" in result.test_results
    assert result.ocp_level is not None
    assert 1 <= result.ocp_level <= 5


@pytest.mark.asyncio
async def test_orchestrator_all_tests():
    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="all", sessions=1, seed=42)
    result = await orch.run()
    assert len(result.test_results) == 5
    assert result.sasmi_score is not None
    assert 0.0 <= result.sasmi_score <= 1.0


@pytest.mark.asyncio
async def test_orchestrator_sasmi_normalized():
    """Partial SASMI (only 2 tests) should still return a valid score."""
    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="meta_cognition,episodic_memory", sessions=1, seed=42)
    result = await orch.run()
    assert result.sasmi_score is not None
    assert 0.0 <= result.sasmi_score <= 1.0


@pytest.mark.asyncio
async def test_orchestrator_seed_reproducible():
    """Two runs with same seed should produce identical SASMI."""
    provider1 = MockProvider("v1")
    provider2 = MockProvider("v1")
    orch1 = OCPOrchestrator(provider1, tests="meta_cognition", sessions=1, seed=42)
    orch2 = OCPOrchestrator(provider2, tests="meta_cognition", sessions=1, seed=42)
    r1 = await orch1.run()
    r2 = await orch2.run()
    assert r1.sasmi_score == r2.sasmi_score


@pytest.mark.asyncio
async def test_unknown_test_raises():
    provider = MockProvider("v1")
    with pytest.raises(ValueError, match="Unknown tests"):
        OCPOrchestrator(provider, tests="nonexistent_test")


# ── Save/load roundtrip ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_result_save_load_roundtrip():
    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="meta_cognition", sessions=1, seed=99)
    result = await orch.run()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)

    try:
        result.save(tmp)
        data = json.loads(tmp.read_text())
        assert data["provider"] == "mock"
        assert data["model"] == "v1"
        assert "meta_cognition" in data["test_results"]
        assert data["ocp_level"] is not None
    finally:
        tmp.unlink(missing_ok=True)


# ── HTML report ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_report_generates():
    from ocp.cli.report import generate_report

    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="meta_cognition,drive_conflict", sessions=1, seed=42)
    result = await orch.run()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        report_path = Path(f.name)

    try:
        result.save(results_path)
        out = generate_report(results_path, report_path)
        html = out.read_text()
        assert "OCP Evaluation Report" in html
        assert "Performance Radar" in html
        assert "svg" in html.lower()
    finally:
        results_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)


# ── Badge ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_badge_generates():
    from ocp.cli.badge import generate_badge

    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="meta_cognition", sessions=1, seed=42)
    result = await orch.run()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        results_path = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        badge_path = Path(f.name)

    try:
        result.save(results_path)
        out = generate_badge(results_path, badge_path)
        svg = out.read_text()
        assert "<svg" in svg
        assert "OCP" in svg
    finally:
        results_path.unlink(missing_ok=True)
        badge_path.unlink(missing_ok=True)


# ── Leaderboard DB ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_leaderboard_db_insert_and_query():
    from ocp.server.db import LeaderboardDB

    provider = MockProvider("v1")
    orch = OCPOrchestrator(provider, tests="meta_cognition", sessions=1, seed=42)
    result = await orch.run()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    try:
        db = LeaderboardDB(db_path)
        rid = db.insert_result(result.to_dict(), submitter="pytest")
        assert rid  # got an ID back

        rows = db.get_leaderboard()
        assert len(rows) == 1
        assert rows[0]["model"] == "v1"

        full = db.get_result(rid)
        assert full is not None
        assert full["provider"] == "mock"

        stats = db.get_stats()
        assert stats["total"] == 1
    finally:
        db_path.unlink(missing_ok=True)
