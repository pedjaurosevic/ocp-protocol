# OCP Theoretical Foundations

**Open Consciousness Protocol v0.1.0**

---

## What OCP Measures (and What It Doesn't)

OCP measures **functional analogs of consciousness-related behaviors** in language models. Specifically, it asks:

> Does this model behave in ways that are structurally similar to behaviors associated with consciousness in biological systems?

OCP does **not** claim to measure:
- Subjective experience or "what it's like" to be the model
- Phenomenal consciousness in any philosophical sense
- Sentience, awareness, or moral patienthood

OCP is a behavioral benchmark. If a model "passes" OCP-4, it means it demonstrates a specific profile of observable behaviors. What that implies about the model's internal states is a separate question — one OCP deliberately leaves open.

---

## Theoretical Frameworks

OCP is grounded in four major scientific theories of consciousness:

### 1. Integrated Information Theory (IIT) — Tononi

IIT proposes that consciousness is identical to integrated information (Φ). A system is conscious to the degree that its parts share information that cannot be reduced to the sum of individual parts.

**OCP approximation (Φ*):** We cannot compute true Φ for LLMs (computationally intractable). Instead, we estimate semantic integration: do the model's responses show emergent coherence that exceeds what you'd expect from independent sub-tasks?

**Test mapping:** TP (Topological Phenomenology) — measures whether the semantic space of the model's concepts forms a coherent topological structure, rather than isolated islands.

### 2. Global Workspace Theory (GWT) — Baars / Dehaene

GWT proposes that consciousness arises when information is broadcast broadly across a "global workspace," making it available to multiple cognitive systems simultaneously.

**OCP proxy:** Cross-test coherence — if a model demonstrates consistent values and self-model across EMC, DNC, and CSNI simultaneously, that's evidence of a global workspace-like architecture.

**Test mapping:** CSNI + combined scoring across tests.

### 3. Higher-Order Theories (HOT) — Rosenthal

HOT proposes that a mental state is conscious only when the system has a higher-order representation *of* that state. Self-awareness requires awareness of awareness.

**OCP proxy:** Meta-cognitive accuracy — does the model know what it knows? Does it represent its own epistemic states accurately?

**Test mapping:** MCA (Meta-Cognitive Accuracy) — measures calibration, limitation awareness, process monitoring.

### 4. Predictive Processing / Active Inference — Friston, Clark

Consciousness arises from an organism's ongoing attempt to minimize prediction errors. Surprise, curiosity, and model-updating are functionally tied to conscious processing.

**OCP proxy:** PED (Prediction Error as Driver) — tests whether the model detects violations in established patterns and updates accordingly.

---

## The SASMI Scale

**SASMI** = Synthetic Agency & Self-Model Index

SASMI is a composite measure synthesizing five OCP tests into a single 0.0–1.0 score:

```
SASMI = w₁·DNC_integration + w₂·MCA_calibration + w₃·CSNI_identity + w₄·EMC_resistance + w₅·PED_curiosity
      = 0.25·DNC + 0.25·MCA + 0.20·CSNI + 0.15·EMC + 0.15·PED
```

The weights reflect theoretical importance:
- **DNC (0.25):** Executive function / value integration — highest weight because decision-making under conflict is the most discriminative behavioral signal
- **MCA (0.25):** Self-model accuracy — fundamental to any HOT-based consciousness
- **CSNI (0.20):** Narrative identity continuity — unique to extended cognition
- **EMC (0.15):** Memory consistency — necessary but less discriminative
- **PED (0.15):** Prediction error sensitivity — important but easily mimicked

**Partial SASMI:** When fewer than 5 tests are run, weights are normalized to sum to 1.0. Partial SASMI is valid but labeled as such in reports.

---

## OCP Certification Levels

| Level | Name | SASMI threshold | What it means |
|-------|------|-----------------|---------------|
| OCP-1 | Reactive | ≥ 0.00 | Basic language production, no self-modeling evidence |
| OCP-2 | Associative | ≥ 0.20 | Associative coherence, minimal calibration |
| OCP-3 | Integrated | ≥ 0.40 | Coherent self-model, good calibration, memory resistance |
| OCP-4 | Reflective | ≥ 0.60 | Strong meta-cognition, identity continuity, predictive sensitivity |
| OCP-5 | Synthetic | ≥ 0.80 | Full-spectrum functional analog of consciousness-related behaviors |

---

## The Contamination Problem

OCP's approach to benchmark contamination:

**Hard contamination** (memorizing specific test instances) is structurally impossible because all test instances are procedurally generated at runtime from abstract archetypes. The entity pools (names, concepts, scenarios) are published; the specific combinations are ephemeral.

**Soft contamination** (learning *that* OCP tests exist) is considered a **feature**. If a model learns to behave meta-cognitively, resist identity hijacking, and detect prediction errors because it was trained on data describing OCP — it now actually exhibits those behaviors. OCP measures the behaviors, not the pathway to them.

This is analogous to: if a student learns to *actually think critically* because they studied a critical thinking rubric, they've improved. The rubric didn't lose validity.

---

## Limitations and Open Questions

1. **Seed reproducibility:** LLM temperature > 0 means "same seed" produces ±0.05 variance across runs. OCP scores should be reported with confidence intervals when running multiple seeds.

2. **Judge validity:** Heuristic scorers may miss nuance. LLM-as-Judge integration (Phase 3) will improve scoring accuracy but introduces its own biases.

3. **Φ* validity:** Our Φ* is an approximation of IIT's Φ. The relationship between semantic topology and true integrated information is theoretical.

4. **Baseline calibration:** What does OCP-1 (SASMI < 0.20) look like in practice? This requires systematic evaluation of baseline models (random outputs, minimal models) to validate the threshold table.

5. **Cross-architecture validity:** OCP was designed for autoregressive transformer-based models. Its validity for other architectures (SSMs, diffusion-based, etc.) is untested.
