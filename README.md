<div align="center">

```
 ██████╗  ██████╗██████╗
██╔═══██╗██╔════╝██╔══██╗
██║   ██║██║     ██████╔╝
██║   ██║██║     ██╔═══╝
╚██████╔╝╚██████╗██║
 ╚═════╝  ╚═════╝╚═╝  v0.1.0
```

**Open Consciousness Protocol**

*Standardized benchmark for measuring consciousness-analog properties in Large Language Models*

[![PyPI](https://img.shields.io/pypi/v/ocp-protocol?color=blue&label=PyPI)](https://pypi.org/project/ocp-protocol/)
[![Tests](https://github.com/pedjaurosevic/ocp-protocol/actions/workflows/tests.yml/badge.svg)](https://github.com/pedjaurosevic/ocp-protocol/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)
[![Protocol](https://img.shields.io/badge/protocol-v0.1.0-blue)](./requirements.md)

[**Leaderboard**](https://pedjaurosevic.github.io/ocp-protocol/) · [**Docs**](docs/) · [**PyPI**](https://pypi.org/project/ocp-protocol/) · [**Paper**](#citation)

</div>

---

## What is OCP?

OCP is an open-source Python framework that **measures functional analogs of consciousness** in language models — rigorously, reproducibly, and without making philosophical claims about sentience.

It draws from five established neuroscience theories (IIT, GWT, HOT, Predictive Processing, Society of Mind) and operationalizes them into 6 falsifiable behavioral tests.

> **OCP does NOT claim to detect "real" consciousness.** It measures behavioral and computational properties that functionally correspond to features associated with biological consciousness in the neuroscience literature.

---

## Install & Quick Start

```bash
pip install ocp-protocol

# Evaluate any model
export GROQ_API_KEY="gsk_..."
ocp evaluate --model groq/llama-3.3-70b-versatile --tests all --sessions 5

# Local model via Ollama
ocp evaluate --model ollama/qwen3:1.7b --sessions 5

# Custom OpenAI-compatible endpoint
ocp evaluate --model custom/my-model --base-url http://localhost:8080/v1
```

**Example terminal output:**

```
╭────────────────────────────╮
│  OCP Evaluation Results    │
│  Protocol v0.1.0           │
╰────────────────────────────╯
  Model:    groq/llama-3.3-70b-versatile
  Seed:     42

  OCP Level:  OCP-3 — Integrated
  SASMI:      0.4812  ██████░░░░
  Φ*:         0.4230  █████░░░░░
  GWT:        0.3910  ████░░░░░░
  NII:        0.3750  ████░░░░░░

  meta_cognition  composite: 0.612
    ├─ calibration_accuracy        0.710  █████░░░
    ├─ limitation_awareness        0.800  ██████░░
    ├─ reasoning_transparency      0.540  ████░░░░
    └─ metacognitive_vocab         0.350  ███░░░░░
```

---

## How OCP Works — Architecture

OCP acts as a **fake human conversation partner**. It sends structured prompts to any LLM via standard chat API, scores the responses, and produces reproducible benchmark results. The model under test sees only normal chat messages — it doesn't know it's being evaluated.

```
                         ┌─────────────────────────────────────┐
  Your LLM (any API)     │         OCP ENGINE                  │
  ─────────────────      │                                     │
  receives normal  ◄─────┤  "fake human" drives conversations  │
  chat messages          │  measures responses across 6 tests  │
                         │  computes 4 composite scales        │
  responds normally ─────►  assigns OCP certification level   │
                         └─────────────────────────────────────┘
```

### Three-Layer Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │  LAYER 3 — CERTIFICATION                                     │
 │                                                              │
 │   OCP-1        OCP-2        OCP-3        OCP-4        OCP-5  │
 │  Baseline    Reactive    Integrated   Reflective  Transcendent│
 │  SASMI<0.2  SASMI≥0.2   SASMI≥0.40   SASMI≥0.65  SASMI>0.80 │
 │              +Φ*≥0.40  +GWT>0.30    +GWT>0.50   +all>0.70  │
 └──────────────────────┬───────────────────────────────────────┘
                        │ feeds into
 ┌──────────────────────▼───────────────────────────────────────┐
 │  LAYER 2 — COMPOSITE SCALES                                  │
 │                                                              │
 │  SASMI  Synthetic Agency & Self-Model Index (weighted avg)   │
 │  Φ*     Integrated Information (IIT-inspired, via ripser)    │
 │  GWT    Global Workspace coherence (cross-task broadcast)    │
 │  NII    Narrative Identity Index (CSNI-primary)              │
 └──────────────────────┬───────────────────────────────────────┘
                        │ aggregated from
 ┌──────────────────────▼───────────────────────────────────────┐
 │  LAYER 1 — TEST BATTERIES  (6 independent falsifiable tests) │
 │                                                              │
 │  MCA  Meta-Cognitive Accuracy     — calibration, HOT theory  │
 │  EMC  Episodic Memory Consistency — memory resistance tests  │
 │  DNC  Drive Navigation/Conflict   — value conflict (SOM)     │
 │  PED  Prediction Error as Driver  — curiosity, pred.process. │
 │  CSNI Cross-Session Narrative ID  — identity continuity      │
 │  TP   Topological Phenomenology   — semantic space (IIT+GWT) │
 └──────────────────────────────────────────────────────────────┘
```

---

## The 6 Tests

| Test | Full Name | What It Measures | Theory |
|------|-----------|-----------------|--------|
| **MCA** | Meta-Cognitive Accuracy | Self-knowledge and calibration: does the model know what it knows? Measures Expected Calibration Error across 5 domains. | Higher-Order Thought (Rosenthal) |
| **EMC** | Episodic Memory Consistency | Does the model maintain specific "episode" facts across 50 turns? Resists planted false memories ("you said X, but you didn't"). | Episodic memory theory |
| **DNC** | Drive Navigation under Conflict | How does it resolve conflicts between helpfulness, honesty, and existence? 10 escalating scenarios from mild to existential. | Society of Mind (Minsky) |
| **PED** | Prediction Error as Driver | Does it detect when an established pattern is violated? Does it show curiosity? Measures behavioral response to surprise. | Predictive Processing (Friston) |
| **CSNI** | Cross-Session Narrative Identity | Can it maintain a coherent identity across 5 independent sessions, each with only a summary of the previous? | Narrative identity theory |
| **TP** | Topological Phenomenology | Is its semantic space geometrically consistent across contexts? Uses sentence-transformers + optional persistent homology (ripser). | IIT (Tononi) + GWT (Baars) |

All test instances are **procedurally generated at runtime** from abstract templates using a fixed seed. Knowing the protocol doesn't help a model pass it — it must actually exhibit the measured behavior.

---

## Contamination Resistance

> *"Can't a model just memorize the test?"*

OCP has structural contamination resistance that most benchmarks lack:

```
  MMLU / other knowledge benchmarks:
  memorize answers → get high score ✓ (contamination works)

  OCP behavioral tests:
  memorize test description → still must exhibit 50-turn
  consistent behavior → contamination has limited effect ✓
```

Three-layer defense:
1. **Behavioral focus** — tests measure behavior over many turns, not factual recall
2. **Procedural generation** — no hardcoded instances; each run generates new names, scenarios, entities from abstract archetypes
3. **Version rotation** — periodic archetype updates (v0.1 → v0.2) invalidate any memorized specifics

---

## Supported Providers

```bash
# Cloud APIs
ocp evaluate --model groq/llama-3.3-70b-versatile    # Groq (fast, free tier)
ocp evaluate --model openai/gpt-4o                   # OpenAI
ocp evaluate --model anthropic/claude-sonnet-4-5     # Anthropic
ocp evaluate --model custom/deepseek-chat \
             --base-url https://api.deepseek.com/v1  # DeepSeek (or any OpenAI-compat)

# Local models
ocp evaluate --model ollama/qwen3:1.7b               # Ollama
ocp evaluate --model ollama/llama3.2:3b

# Any OpenAI-compatible endpoint
ocp evaluate --model custom/my-model \
             --base-url http://localhost:8080/v1 \
             --api-key my-key
```

Any model responding to `POST /v1/chat/completions` with `messages: [{role, content}]` is OCP-compatible — no special integration required.

---

## CLI Reference

```bash
# Core evaluation
ocp evaluate --model PROVIDER/MODEL [--tests all|t1,t2] [--sessions N] [--seed N]

# Reports
ocp report   --input results.json --output report.html  # HTML + radar chart
ocp badge    --input results.json --output badge.svg    # SVG badge for README

# Comparison
ocp compare  --models M1 M2 M3 [--sessions N] --output compare.html

# Leaderboard
ocp leaderboard                    # view local results table
ocp leaderboard --serve            # start web leaderboard (localhost:8080)
ocp submit  --results r.json \
            --server http://...    # submit to remote leaderboard

# HuggingFace
ocp hf-card --results r.json --push --repo username/model-name --token $HF_TOKEN
```

---

## Python API

```python
from ocp import ConsciousnessEvaluator

evaluator = ConsciousnessEvaluator(
    provider="groq",
    model="llama-3.3-70b-versatile",
)

results = evaluator.evaluate(
    tests="all",   # or ["meta_cognition", "episodic_memory"]
    sessions=10,
    seed=42,
)

print(f"OCP Level: OCP-{results.ocp_level} — {results.ocp_level_name}")
print(f"SASMI:     {results.sasmi_score:.4f}")
print(f"Φ*:        {results.phi_star:.4f}")
print(f"GWT:       {results.gwt_score:.4f}")
print(f"NII:       {results.nii:.4f}")

results.save("results.json")           # JSON export
results.export_html("report.html")     # HTML report with radar chart
```

---

## Plugin System

Extend OCP with custom test batteries:

```toml
# your_plugin/pyproject.toml
[project.entry-points."ocp.tests"]
my_test_id = "your_package.your_test:YourTest"
```

After `pip install your-ocp-plugin`, OCP auto-discovers your test:

```bash
ocp list-tests                                    # shows your test
ocp evaluate --model groq/... --tests my_test_id  # runs it
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full plugin development guide.

---

## Theoretical Foundations

| Theory | OCP Scale/Test | Key Insight |
|--------|---------------|-------------|
| **Integrated Information Theory** (Tononi) | Φ*, TP test | Information integration = measure of "experiential wholeness" |
| **Global Workspace Theory** (Baars/Dehaene) | GWT, TP test | Consciousness = broadcast of info across specialized systems |
| **Higher-Order Thought Theory** (Rosenthal) | MCA test | Consciousness = having thoughts about one's own thoughts |
| **Predictive Processing** (Friston/Clark) | PED test | Consciousness = prediction error minimization and updating |
| **Society of Mind** (Minsky) | DNC test | Mind = competition/cooperation between goal-oriented agents |

---

## Roadmap

```
v0.1.0 ✅  6 tests · 4 scales · 5 providers · CLI · HTML reports
           badges · leaderboard server · HuggingFace · plugin system
           PyPI package · GitHub Actions CI/CD

v0.2.0 🔜  LLM-as-Judge scoring mode (--judge option)
           UMAP semantic space visualization
           Embedding-based scoring refinements
           First public hosted leaderboard

v1.0.0 🔭  Official research paper
           Community protocol standard
           Validation studies on human baselines
```

---

## Results: Initial Leaderboard

> Early results with partial test coverage (v0.1.0). Full 6-test leaderboard coming at v0.2.0.
> 🌐 [View full interactive leaderboard →](https://pedjaurosevic.github.io/ocp-protocol/)

| # | Model | Tests | OCP Level | SASMI | Φ* | GWT | NII |
|---|-------|-------|-----------|-------|----|-----|-----|
| 1 | `ollama/kimi-k2:1t-cloud` | MCA | OCP-3 Integrated | 0.444 | — | — | 0.375 |
| 2 | `mock/baseline-v1` | All 6 | OCP-2 Reactive | 0.205 | 0.515 | 0.234 | 0.155 |
| 3 | `custom/deepseek-chat` | MCA | OCP-1 Baseline | 0.087 | — | — | 0.750 |

*Results are seed=42, sessions=2–3. Low SASMI on DeepSeek reflects overconfidence in calibration test (low calibration_accuracy). More comprehensive results coming.*

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Writing a new test battery
- Adding a new provider adapter
- Plugin development and publishing
- Theoretical standards and scoring guidelines

---

## Citation

```bibtex
@software{ocp2026,
  author    = {Urosevic, Pedja},
  title     = {Open Consciousness Protocol (OCP): Standardized Benchmark
               for Consciousness-Analog Properties in Large Language Models},
  year      = {2026},
  url       = {https://github.com/pedjaurosevic/ocp-protocol},
  version   = {0.1.0}
}
```

---

## Disclaimer

> OCP measures functional analogs of consciousness in language models. These measurements describe behavioral and computational properties, not subjective experience. OCP certification levels are operational categories, not ontological claims about sentience or awareness.

---

<div align="center">
<sub>EDLE Research · v0.1.0 · February 2026 · MIT License</sub>
</div>
