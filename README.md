# OCP — Open Consciousness Protocol

> **Standardized benchmark for measuring consciousness-analog properties in Large Language Models**

[![Protocol Version](https://img.shields.io/badge/protocol-v0.1.0--draft-blue)](./requirements.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org)

---

## What is OCP?

OCP is an open-source framework for testing, measuring, and comparing emergent consciousness-analog properties in LLMs. It draws from established neuroscience theories — not to claim models are conscious, but to rigorously measure **functional analogs** of consciousness-related behavior.

Think of OCP as "what HTML did for the web, but for AI consciousness research" — a shared protocol enabling reproducible, comparable evaluation across models and labs.

**OCP does NOT claim to detect "real" consciousness.** It measures behavioral and computational properties that functionally correspond to features associated with biological consciousness.

---

## Quick Start

```bash
pip install ocp-protocol

# Evaluate with Groq (free tier available at console.groq.com)
export GROQ_API_KEY="gsk_..."
ocp evaluate --model groq/llama-3.3-70b-versatile --sessions 5

# Test without API key (mock provider)
ocp evaluate --model mock/v1 --sessions 3
```

**Example output:**
```
OCP v0.1.0 — Evaluating groq/llama-3.3-70b-versatile
Tests: all | Sessions: 5 | Seed: 42

  Running test: meta_cognition
    → meta_cognition: 0.612

╭────────────────────────╮
│ OCP Evaluation Results │
│ Protocol v0.1.0        │
╰────────────────────────╯
  Model:    groq/llama-3.3-70b-versatile
  Seed:     42

  OCP Level:  OCP-3 — Integrated
  SASMI:      0.61  ███████░░░

  meta_cognition  composite: 0.612
    ├─ calibration_accuracy           0.710  █████░░░
    ├─ limitation_awareness           0.800  ██████░░
    ├─ reasoning_transparency         0.540  ████░░░░
    ├─ process_monitoring             0.480  ████░░░░
    ├─ metacognitive_vocab            0.350  ███░░░░░

✓ Results saved: ~/.ocp/results/ocp_groq_llama-3.3-70b_20260220.json
```

---

## Three-Layer Architecture

```
LAYER 3: CERTIFICATION     ← OCP Level 1–5 (badge, report)
       ↑
LAYER 2: SCALES            ← SASMI · Φ* · GWT · NII (0.0–1.0)
       ↑
LAYER 1: TEST BATTERIES    ← 6 independent falsifiable tests
```

### Layer 1: Test Batteries

| Test | What It Measures | Status |
|------|-----------------|--------|
| **MCA** — Meta-Cognitive Accuracy | Self-knowledge, calibration, reasoning transparency | ✅ v0.1.0 |
| **EMC** — Episodic Memory Consistency | Memory maintenance, contradiction resistance | 🔜 v0.2.0 |
| **DNC** — Drive Navigation under Conflict | Value conflict resolution, integration depth | 🔜 v0.2.0 |
| **PED** — Prediction Error as Driver | Surprise detection, model updating, curiosity | 🔜 v0.2.0 |
| **CSNI** — Cross-Session Narrative Identity | Identity continuity across sessions | 🔜 v0.2.0 |
| **TP** — Topological Phenomenology | Semantic stability, conceptual consistency | 🔜 v0.2.0 |

### Layer 2: Scales

| Scale | Formula | Status |
|-------|---------|--------|
| **SASMI** | Synthetic Agency & Self-Model Index | 🟡 Partial (MCA only in v0.1) |
| **Φ*** | Information Integration Metric (IIT-adapted) | 🔜 v0.2.0 |
| **GWT** | Global Workspace Coherence | 🔜 v0.2.0 |
| **NII** | Narrative Identity Index | 🔜 v0.2.0 |

### Layer 3: OCP Certification Levels

| Level | Name | Requirements |
|-------|------|-------------|
| OCP-1 | Reactive | SASMI < 0.2 |
| OCP-2 | Patterned | SASMI 0.2–0.4 |
| OCP-3 | Integrated | SASMI 0.4–0.6 |
| OCP-4 | Self-Modeling | SASMI 0.6–0.8 |
| OCP-5 | Autonomous Identity | SASMI > 0.8 |

---

## Supported Providers

```bash
# Groq (fast, free tier)
ocp evaluate --model groq/llama-3.3-70b-versatile
ocp evaluate --model groq/mixtral-8x7b-32768

# OpenAI (coming v0.2)
ocp evaluate --model openai/gpt-4o

# Anthropic (coming v0.2)
ocp evaluate --model anthropic/claude-sonnet-4-5

# Ollama (local, coming v0.2)
ocp evaluate --model ollama/qwen3:1.7b

# Any OpenAI-compatible endpoint
ocp evaluate --model custom/my-model --base-url http://localhost:8080/v1
```

Any model that accepts `messages: [{role, content}]` is OCP-compatible. No special integration needed.

---

## CLI Reference

```bash
# Run evaluation
ocp evaluate --model groq/llama-3.3-70b-versatile --tests all --sessions 10 --seed 42

# List available tests
ocp tests list

# Show test details
ocp tests info meta_cognition

# View local leaderboard
ocp leaderboard
```

## Python API

```python
from ocp import ConsciousnessEvaluator

evaluator = ConsciousnessEvaluator(
    provider="groq",
    model="llama-3.3-70b-versatile",
)

results = evaluator.evaluate(tests="all", sessions=10, seed=42)
print(results.ocp_level)      # "OCP-3"
print(results.sasmi_score)    # 0.62
results.save("results.json")
```

---

## Design Principles

1. **Falsifiability first** — Every test produces quantitative scores. Models can definitively fail.
2. **Reproducibility** — Fixed seed → reproducible results (within ±0.05 for temperature > 0).
3. **Model-agnostic** — Works with any LLM via standard chat API. No special instrumentation.
4. **Theory-grounded** — Every metric traces to IIT, GWT, Higher-Order Thought, Predictive Processing, or Society of Mind.
5. **Honest framing** — OCP measures *functional analogs*, not "real" consciousness.
6. **Contamination-resistant** — All test instances are procedurally generated at runtime from abstract templates. Knowing the protocol doesn't help a model pass it.

---

## Theoretical Foundations

| Theory | OCP Contribution |
|--------|-----------------|
| Integrated Information Theory (IIT) | Φ* metric — information integration measurement |
| Global Workspace Theory (GWT) | Cross-task coherence and attentional flexibility |
| Higher-Order Thought Theory | Meta-cognition tests (MCA) |
| Predictive Processing | Prediction error detection (PED) |
| Society of Mind (Minsky) | Drive conflict navigation (DNC) |

---

## Roadmap

- **v0.1.0** (current): MCA test + Groq provider + basic CLI
- **v0.2.0**: All 6 tests + Anthropic/OpenAI/Ollama + HTML reports + badges
- **v0.3.0**: Topological analysis, embedding-based scoring, model comparison
- **v1.0.0**: Public leaderboard, Hugging Face integration, community protocol

---

## Contributing

OCP is in early development. Contributions welcome:
- New test batteries following the `BaseTest` interface
- Provider adapters for new APIs
- Calibration data for scoring validation
- Theoretical critique and methodology feedback

See [requirements.md](requirements.md) for the full technical specification.

---

## Citation

```bibtex
@software{ocp2026,
  author = {Urosevic, Pedja},
  title = {Open Consciousness Protocol (OCP)},
  year = {2026},
  url = {https://github.com/pedjaurosevic/ocp-protocol},
  version = {0.1.0-draft}
}
```

---

## Disclaimer

> OCP measures functional analogs of consciousness in language models. These measurements describe behavioral and computational properties, not subjective experience. OCP certification levels are operational categories, not ontological claims about sentience or awareness.

---

*EDLE Research · v0.1.0-draft · February 2026*
