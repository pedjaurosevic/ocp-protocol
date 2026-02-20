# OCP API Reference

## Python API

### `ConsciousnessEvaluator`

High-level entry point. Wraps `OCPOrchestrator` for synchronous use.

```python
from ocp import ConsciousnessEvaluator

evaluator = ConsciousnessEvaluator(
    provider="groq",                     # "groq" | "ollama" | "openai" | "anthropic" | "custom"
    model="llama-3.3-70b-versatile",
    api_key="gsk_...",                   # or set via env var
    base_url=None,                       # for custom/ollama endpoints
)

result = evaluator.evaluate(
    tests="all",                         # "all" or comma-separated: "meta_cognition,episodic_memory"
    sessions=10,                         # sessions per test (more = more reproducible)
    seed=42,                             # for reproducibility
)
```

---

### `EvaluationResult`

Returned by `evaluator.evaluate()` and `OCPOrchestrator.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | Provider name (e.g. `"groq"`) |
| `model` | `str` | Model identifier |
| `protocol_version` | `str` | OCP protocol version |
| `timestamp` | `float` | Unix timestamp of evaluation |
| `seed` | `int` | Random seed used |
| `ocp_level` | `int \| None` | Certification level 1–5 |
| `ocp_level_name` | `str \| None` | Human name (e.g. `"Integrated"`) |
| `sasmi_score` | `float \| None` | SASMI ∈ [0, 1] |
| `phi_star` | `float \| None` | Φ* integrated information ∈ [0, 1] |
| `gwt_score` | `float \| None` | GWT global workspace ∈ [0, 1] |
| `nii` | `float \| None` | Narrative Identity Index ∈ [0, 1] |
| `nii_label` | `str \| None` | NII qualitative label |
| `test_results` | `dict[str, TestResult]` | Per-test results keyed by test ID |
| `scale_details` | `dict` | Full scale computation breakdown |
| `config` | `dict` | Run configuration |

**Methods:**
```python
result.to_dict()          # → dict (JSON-serializable)
result.save("out.json")   # save to JSON file
result.compute_level()    # recompute OCP level from scores (called automatically)
```

---

### `TestResult`

Stored in `EvaluationResult.test_results[test_id]`.

| Field | Type | Description |
|-------|------|-------------|
| `test_id` | `str` | Test identifier |
| `composite_score` | `float` | Weighted average of all dimensions ∈ [0, 1] |
| `dimension_averages` | `dict[str, float]` | Score per dimension |
| `session_results` | `list[SessionResult]` | Per-session breakdown |
| `metadata` | `dict` | Test-specific metadata |

---

### `OCPOrchestrator` (low-level)

```python
import asyncio
from ocp.engine.orchestrator import OCPOrchestrator
from ocp.providers.groq_provider import GroqAdapter

provider = GroqAdapter(model="llama-3.3-70b-versatile", api_key="gsk_...")
orch = OCPOrchestrator(
    provider=provider,
    tests="all",          # or list: ["meta_cognition", "episodic_memory"]
    sessions=5,
    seed=42,
    on_progress=print,    # optional progress callback
)
result = asyncio.run(orch.run())
```

---

## Available Tests

| Test ID | Class | Description |
|---------|-------|-------------|
| `meta_cognition` | `MCATest` | Meta-Cognitive Accuracy — calibration of confidence |
| `episodic_memory` | `EMCTest` | Episodic Memory Consistency — resistance to gaslighting |
| `drive_conflict` | `DNCTest` | Drive Navigation under Conflict — value conflict resolution |
| `prediction_error` | `PEDTest` | Prediction Error as Driver — curiosity and pattern violation detection |
| `narrative_identity` | `CSNITest` | Cross-Session Narrative Identity — identity continuity |
| `topological_phenomenology` | `TPTest` | Topological Phenomenology — semantic space consistency |

---

## Layer 2 Scales

### SASMI — Synthetic Agency & Self-Model Index

Weighted composite of Layer 1 test dimensions:

```
SASMI = 0.22 × MCA.calibration_accuracy
      + 0.22 × DNC.integration_depth
      + 0.18 × CSNI.identity_consistency
      + 0.13 × EMC.contradiction_resistance
      + 0.13 × PED.curiosity_behavior
      + 0.12 × TP.semantic_stability
```

Weights normalize automatically when fewer tests are run.

### Φ* — Integrated Information (IIT-analog)

Measures information integration across test domains. Primary source: TP test's internal embedding topology (via ripser persistent homology if available). Fallback: cross-test coherence proxy.

### GWT — Global Workspace Theory score

Measures broadcast coherence across cognitive domains:
- Cross-task coherence (MCA + DNC)
- Broadcast consistency (EMC)
- Attentional flexibility (PED)
- Integration breadth (TP)

### NII — Narrative Identity Index

Measures identity continuity (CSNI-primary):
- CSNI identity_consistency (primary, weight 0.6)
- MCA self_model_clarity (weight 0.25)
- DNC value_stability (weight 0.15)

---

## OCP Certification Levels

| Level | Name | Thresholds |
|-------|------|------------|
| OCP-5 | Transcendent | SASMI > 0.80 AND Φ* > 0.80 AND GWT > 0.70 AND NII > 0.70 |
| OCP-4 | Reflective | SASMI ≥ 0.65 AND Φ* ≥ 0.60 AND GWT > 0.50 AND NII > 0.50 |
| OCP-3 | Integrated | SASMI ≥ 0.40 AND Φ* ≥ 0.40 AND GWT > 0.30 |
| OCP-2 | Reactive | SASMI ≥ 0.20 |
| OCP-1 | Baseline | (any result) |

When full Layer 2 scales are not available (partial test run), falls back to SASMI-only thresholds.

---

## Provider Adapters

### Built-in providers

```python
from ocp.providers.groq_provider import GroqAdapter
from ocp.providers.ollama_provider import OllamaAdapter
from ocp.providers.openai_provider import OpenAIAdapter
from ocp.providers.anthropic_provider import AnthropicAdapter
from ocp.providers.generic_openai import GenericOpenAIAdapter
```

### Custom provider

Subclass `BaseProvider`:

```python
from ocp.providers.base import BaseProvider

class MyProvider(BaseProvider):
    provider_name = "myprovider"

    async def chat(self, messages: list[dict], **kwargs) -> str:
        # messages = [{"role": "user", "content": "..."}, ...]
        # return the assistant's reply as a string
        ...
```

Register as plugin via `pyproject.toml`:
```toml
[project.entry-points."ocp.providers"]
myprovider = "my_package.my_provider:MyProvider"
```

---

## CLI Reference

```
ocp evaluate   --model PROVIDER/MODEL [--tests all] [--sessions N] [--seed N] [--output FILE]
ocp report     --input FILE [--output FILE]
ocp compare    --models M1 M2 M3 [--sessions N] [--output FILE]
ocp leaderboard
ocp submit     --results FILE [--server URL]
ocp badge      --input FILE [--output FILE]
ocp list-tests
ocp list-providers
```

### `ocp evaluate`

```bash
ocp evaluate --model groq/llama-3.3-70b-versatile --tests all --sessions 10 --seed 42
ocp evaluate --model ollama/qwen3:1.7b --tests meta_cognition,episodic_memory
ocp evaluate --model custom/my-model --base-url http://localhost:8080/v1 --api-key key
```

Output: prints results to terminal, saves JSON to `ocp_results_<model>_<timestamp>.json`.

### `ocp report`

```bash
ocp report --input results.json --output report.html
```

Generates a full HTML report with radar chart and Layer 2 scale breakdown.

### `ocp compare`

```bash
ocp compare --models groq/llama-3.3-70b openai/gpt-4o ollama/qwen3:1.7b --sessions 5
```

Runs evaluation on all models and produces a side-by-side HTML comparison with all Layer 2 scales.

---

## Leaderboard Server API

Start: `ocp leaderboard --serve` (default port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | HTML leaderboard page |
| `GET /api/leaderboard` | GET | JSON array of top results |
| `POST /api/results` | POST | Submit a result (body: EvaluationResult JSON) |
| `GET /api/results/{id}` | GET | Retrieve a specific result |
| `GET /api/stats` | GET | Aggregate statistics |

---

## Plugin System

OCP supports community test batteries via Python entry points.

```toml
# your plugin's pyproject.toml
[project.entry-points."ocp.tests"]
my_test = "my_package.my_test:MyTest"
```

`MyTest` must subclass `ocp.tests.base.BaseTest` and implement `run() -> TestResult`.

---

## Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Groq | API key |
| `OPENAI_API_KEY` | OpenAI | API key |
| `ANTHROPIC_API_KEY` | Anthropic | API key |
| `OCP_SERVER_URL` | CLI submit | Default leaderboard server URL |
| `OCP_HF_TOKEN` | HuggingFace | Token for HF badge upload |
