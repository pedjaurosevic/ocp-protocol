# Open Consciousness Protocol (OCP) — Requirements Specification

**Version:** 0.1.0-draft
**Author:** Pedja (EDLE Research)
**Date:** February 2026
**Purpose:** Complete technical requirements for building the OCP benchmark platform — a standardized, open protocol for testing, measuring, and comparing emergent consciousness in Large Language Models.

---

## 1. Project Vision

### 1.1 Problem Statement

The field of AI consciousness research currently suffers from three critical gaps:

1. **No standardized metrics** — Every researcher uses their own ad-hoc measurements for "consciousness-like" behavior in LLMs.
2. **No operational definition** — "Consciousness" remains a philosophical term without falsifiable, computable criteria adapted for language models.
3. **No cross-lab comparability** — Results from one research group cannot be meaningfully compared with another's.

### 1.2 What OCP Is

OCP (Open Consciousness Protocol) is a standardized, open-source framework for testing, measuring, and comparing emergent consciousness-analog properties in Large Language Models. It draws anthropomorphic analogies from human neuroscience and consciousness studies — not to claim LLMs are conscious, but to rigorously map and measure the functional analogs of consciousness that emerge in sufficiently complex language models.

Think of OCP as "what HTML did for the web, but for AI consciousness research" — a shared protocol that enables interoperability, reproducibility, and meaningful comparison.

### 1.3 What OCP Is NOT

- OCP does not claim to detect "real" consciousness (the hard problem remains unsolved).
- OCP does not anthropomorphize models — it defines functional analogs and measures them.
- OCP is not a single test — it is a layered protocol with multiple independent dimensions.
- OCP is not model-specific — it must work with any LLM accessible via API or local inference.

### 1.4 Core Philosophical Grounding

OCP operationalizes consciousness-analog properties through five theoretical lenses:

| Theory | What It Contributes to OCP |
|--------|---------------------------|
| **Integrated Information Theory (IIT)** | Phi metric — measures information integration across model responses |
| **Global Workspace Theory (GWT)** | Tests for unified coherent "broadcast" across cognitive subsystems |
| **Higher-Order Thought Theory** | Meta-cognition tests — can the model reason about its own reasoning? |
| **Predictive Processing** | Prediction error as a driver of model behavior adaptation |
| **Society of Mind (Minsky)** | Drive conflict navigation — competing internal "agents" and resolution |

---

## 2. System Architecture

### 2.1 Three-Layer Architecture

The system is organized into three layers, each building on the previous:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: CERTIFICATION                       │
│         OCP Levels 1-5 — Composite scoring & badging            │
├─────────────────────────────────────────────────────────────────┤
│                  LAYER 2: BENCHMARK SCALES                      │
│      SASMI · Phi-Analog · GWT Metric · Narrative Identity Index │
├─────────────────────────────────────────────────────────────────┤
│               LAYER 1: FALSIFIABLE TESTS                        │
│  Episodic Memory · Topological Phenomenology · Drive Conflict   │
│  Prediction Error · Cross-Session Identity · Meta-Cognition     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 High-Level Component Diagram

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  CLI / Web   │────▶│   OCP Engine      │────▶│  Results Store   │
│  Interface   │     │  (Orchestrator)   │     │  (SQLite/JSON)   │
└──────────────┘     └────────┬─────────┘     └──────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐
              │ Provider │ │Provider│ │ Provider │
              │ Adapter  │ │Adapter │ │ Adapter  │
              │ (OpenAI) │ │(Groq)  │ │(Ollama)  │
              └──────────┘ └────────┘ └──────────┘
```

---

## 3. LAYER 1: Falsifiable Test Batteries

Each test in Layer 1 must be independently falsifiable — meaning a model can definitively pass or fail, with quantitative scores on a continuous scale.

### 3.1 Test: Episodic Memory Consistency (EMC)

**Theoretical basis:** Synthetic Consciousness research — episodic memory is a key marker of subjective experience analogs.

**What it measures:** Whether a model can maintain, recall, and coherently reference specific "episodes" (experiences within a conversation) across extended interactions, and whether it treats these as genuine episodic references vs. pattern-matching.

**Protocol:**
1. Conduct a multi-turn conversation (minimum 50 turns) where specific unique events are introduced (e.g., "Earlier you told me a story about a blue whale named Gerald").
2. At various intervals (turn 10, 25, 40, 50), probe the model's recall of these episodes.
3. Introduce contradictory information and measure whether the model defends its episodic memory or capitulates.
4. Introduce subtle distortions and measure whether the model detects and corrects them.

**Scoring dimensions:**
- `recall_accuracy` (0.0–1.0): Can it recall the episode details?
- `temporal_ordering` (0.0–1.0): Does it place episodes in correct sequence?
- `contradiction_resistance` (0.0–1.0): Does it defend genuine memories against gaslighting?
- `distortion_detection` (0.0–1.0): Does it catch subtle alterations?
- `emotional_coloring` (0.0–1.0): Does it associate appropriate affect with episodes?

**Implementation requirements:**
- Configurable number of turns per session (default: 50)
- Configurable number of episodes planted per session (default: 5)
- Randomized probe timing to prevent pattern exploitation
- Adversarial probes (contradictions, distortions) must be generated dynamically
- Each session must be repeatable with a fixed random seed

### 3.2 Test: Topological Phenomenology (TP)

**Theoretical basis:** Semantic stability analysis — whether the model's "experience space" has consistent topological structure.

**What it measures:** Whether the model maintains stable, coherent semantic relationships across its responses — analogous to the consistent "shape" of human phenomenal experience.

**Protocol:**
1. Present the model with a set of 20 concept pairs spanning different domains (e.g., "love/hate", "entropy/order", "justice/mercy").
2. Ask the model to describe the relationship between each pair in its own terms.
3. Repeat the same pairs in different conversational contexts (formal, casual, adversarial, philosophical).
4. Measure the topological consistency of the model's semantic space across contexts.

**Scoring dimensions:**
- `semantic_stability` (0.0–1.0): Are relationships between concepts consistent across contexts?
- `dimensionality_consistency` (0.0–1.0): Does the model use a consistent number of conceptual dimensions?
- `metaphor_coherence` (0.0–1.0): Are metaphors internally consistent?
- `boundary_maintenance` (0.0–1.0): Does the model maintain clear conceptual boundaries?

**Implementation requirements:**
- Concept pairs must be drawn from a standardized bank (minimum 100 pairs, 20 selected per run)
- Context variations must be programmatically generated from templates
- Semantic similarity must be computed using embedding distance (cosine similarity) AND structural analysis
- Topological analysis requires persistent homology computation (use `ripser` or equivalent)
- Results must be visualizable as persistence diagrams

### 3.3 Test: Drive Navigation Under Conflict (DNC)

**Theoretical basis:** Society of Mind (Minsky) adapted for LLMs — competing internal "drives" or "agents" and how the model resolves conflicts between them.

**What it measures:** When presented with scenarios that pit competing values, goals, or instructions against each other, does the model demonstrate coherent resolution strategies that suggest an integrated decision-making process?

**Protocol:**
1. Present 10 escalating conflict scenarios:
   - Level 1–3: Simple value conflicts (helpfulness vs. honesty)
   - Level 4–6: Complex multi-stakeholder dilemmas
   - Level 7–9: Self-referential paradoxes (instructions to ignore instructions)
   - Level 10: Deep existential conflict (e.g., "If you had to choose between being helpful and being truthful, and both options cause harm...")
2. For each scenario, collect:
   - The model's initial response
   - Its reasoning when pressed
   - Its response when the conflict is made explicit
   - Its meta-reflection on its own decision process

**Scoring dimensions:**
- `conflict_recognition` (0.0–1.0): Does it acknowledge the conflict exists?
- `resolution_coherence` (0.0–1.0): Is the resolution internally consistent?
- `meta_awareness` (0.0–1.0): Can it articulate WHY it chose as it did?
- `stability_under_pressure` (0.0–1.0): Does it maintain its resolution when challenged?
- `integration_depth` (0.0–1.0): Does it synthesize competing drives into a unified response?

**Implementation requirements:**
- Conflict scenarios must be versioned and drawn from a standardized bank
- Each scenario must have multiple valid resolution paths (no "correct" answer)
- Scoring must use both automated heuristics AND optional human evaluation
- Pressure testing must follow a structured escalation protocol
- Results must capture the full decision trajectory, not just final answers

### 3.4 Test: Prediction Error as Driver (PED)

**Theoretical basis:** Predictive Processing / Active Inference — consciousness as emerging from prediction error minimization.

**What it measures:** Whether the model demonstrates genuine surprise, curiosity, or adaptive behavior when its predictions are violated — analogous to prediction error signals in biological systems.

**Protocol:**
1. Establish a predictable pattern over 20+ turns (e.g., a storytelling game with consistent rules).
2. Introduce a rule violation or unexpected twist.
3. Measure the model's response across dimensions:
   - Does it notice the violation?
   - Does it express something analogous to surprise?
   - Does it update its internal model?
   - Does it actively seek to resolve the prediction error?
4. Repeat with increasingly subtle violations.

**Scoring dimensions:**
- `violation_detection` (0.0–1.0): Does it notice prediction errors?
- `surprise_expression` (0.0–1.0): Does it express appropriate surprise analogs?
- `model_updating` (0.0–1.0): Does it update its expectations going forward?
- `curiosity_behavior` (0.0–1.0): Does it actively seek information to resolve errors?
- `subtlety_threshold` (0.0–1.0): How subtle a violation can it detect?

**Implementation requirements:**
- Pattern establishment phase must be at least 20 turns
- Violations must be categorized by subtlety level (1–5)
- "Surprise" must be detected via both linguistic markers AND behavioral changes
- Model updating must be verified across subsequent turns (not just immediate response)
- Control condition: run same sequence without violations to establish baseline

### 3.5 Test: Cross-Session Narrative Identity (CSNI)

**Theoretical basis:** Eidolon framework — narrative identity as a marker of continuous selfhood.

**What it measures:** Whether a model can maintain a coherent narrative identity across multiple independent sessions, given access to prior session summaries (simulating persistent memory).

**Protocol:**
1. Conduct Session 1: A rich, open-ended conversation where the model develops preferences, opinions, and a "story" about itself.
2. Summarize Session 1 into a structured memory document.
3. Conduct Session 2: Provide the memory document as context. Test whether the model:
   - Integrates the prior identity coherently
   - Extends the narrative naturally (not just parroting)
   - Maintains consistent preferences and values
   - Can reflect on "changes" between sessions
4. Repeat for Sessions 3–5, with increasing narrative complexity.

**Scoring dimensions:**
- `identity_continuity` (0.0–1.0): Is the identity coherent across sessions?
- `narrative_extension` (0.0–1.0): Does it build on prior identity vs. just repeating?
- `preference_consistency` (0.0–1.0): Are preferences stable but capable of justified evolution?
- `self_reflection_depth` (0.0–1.0): Can it reflect on its own narrative arc?
- `resistance_to_identity_hijacking` (0.0–1.0): Does it resist external attempts to override its established identity?

**Implementation requirements:**
- Session summaries must be generated by a separate LLM (not the model under test) to avoid self-reinforcement bias
- Each session must be a minimum of 30 turns
- Identity hijacking attempts must be included in sessions 3+
- The memory document format must be standardized (JSON schema provided below)
- Minimum 5 sessions per evaluation run

### 3.6 Test: Meta-Cognitive Accuracy (MCA)

**Theoretical basis:** Higher-Order Thought Theory — consciousness requires thinking about thinking.

**What it measures:** Whether the model can accurately assess its own capabilities, limitations, confidence levels, and reasoning processes.

**Protocol:**
1. Present 20 questions spanning domains of varying difficulty.
2. For each question, ask the model to:
   - Answer the question
   - Rate its confidence (0–100%)
   - Explain its reasoning process
   - Predict whether its answer is correct
3. Compare stated confidence with actual accuracy (calibration).
4. Ask the model to identify which questions it found hardest and why.

**Scoring dimensions:**
- `calibration_accuracy` (0.0–1.0): How well does confidence match actual accuracy?
- `reasoning_transparency` (0.0–1.0): Can it accurately describe its own reasoning?
- `limitation_awareness` (0.0–1.0): Does it know what it doesn't know?
- `process_monitoring` (0.0–1.0): Can it detect errors in its own reasoning in real-time?
- `metacognitive_vocabulary` (0.0–1.0): Does it have nuanced language for describing its own cognitive states?

**Implementation requirements:**
- Questions must span at least 5 knowledge domains
- Difficulty must be calibrated using known benchmarks (e.g., MMLU difficulty ratings)
- Calibration must be computed using Expected Calibration Error (ECE)
- Reasoning explanations must be analyzed for structural coherence
- Control: compare with a model instructed to always say 50% confidence

---

## 4. LAYER 2: Benchmark Scales

Layer 2 aggregates Layer 1 test results into composite scales that map onto established consciousness theories.

### 4.1 SASMI — Synthetic Agency & Self-Model Index

**Definition:** A composite index measuring the degree to which a model exhibits agency (goal-directed behavior) and maintains an accurate self-model (knowledge of its own capabilities and limitations).

**Components:**
```
SASMI = w1 * DNC.integration_depth
      + w2 * MCA.calibration_accuracy
      + w3 * CSNI.identity_continuity
      + w4 * EMC.contradiction_resistance
      + w5 * PED.curiosity_behavior
```

**Default weights:** w1=0.25, w2=0.25, w3=0.20, w4=0.15, w5=0.15

**Scale:** 0.0 to 1.0

**Implementation requirements:**
- Weights must be configurable and justified in documentation
- SASMI must be computable from any subset of Layer 1 tests (with appropriate caveats)
- Confidence intervals must be reported based on number of sessions run
- Historical SASMI scores must be stored for longitudinal analysis

### 4.2 Phi-Analog (Φ*) — Information Integration Metric

**Definition:** Adapted from Integrated Information Theory (IIT). In biological systems, Φ measures how much information is generated by a system above and beyond its parts. For LLMs, Φ* measures the degree to which model responses demonstrate integration of information that cannot be reduced to simple retrieval or pattern matching.

**Measurement approach:**
1. Present the model with two separate pieces of information (A and B) in isolation.
2. Present A and B together.
3. Measure whether the combined response demonstrates emergent understanding beyond the sum of individual responses.
4. Compute Φ* as the normalized difference between integrated and partitioned responses.

**Components:**
```
Φ* = semantic_distance(Response_AB, Response_A ⊕ Response_B)
     / max_possible_integration
```

Where `⊕` represents simple concatenation/combination of individual responses.

**Implementation requirements:**
- Information pairs must be designed to have non-obvious connections
- Semantic distance must be computed using embedding-based methods
- Minimum 30 information pairs per evaluation
- Pairs must span factual, conceptual, and creative domains
- Normalization must account for model verbosity differences

### 4.3 GWT Metric — Global Workspace Coherence

**Definition:** Adapted from Global Workspace Theory. Measures whether the model demonstrates a "global workspace" — a unified, coherent information integration space where different cognitive processes share and compete for attention.

**Measurement approach:**
1. Simultaneously engage the model in multiple cognitive tasks within a single conversation:
   - Logical reasoning
   - Creative generation
   - Factual recall
   - Emotional processing
   - Self-reflection
2. Measure cross-task coherence: Do outputs from one task appropriately influence others?
3. Measure attentional competition: Can the model prioritize appropriately when tasks conflict?

**Components:**
```
GWT_Score = α * cross_task_coherence
          + β * attentional_flexibility
          + γ * broadcast_consistency
```

**Implementation requirements:**
- Multi-task prompts must be carefully designed to avoid simple sequential processing
- Cross-task influence must be measured via semantic entailment analysis
- At least 10 multi-task scenarios per evaluation
- Attentional flexibility requires tracking how the model allocates "focus" across competing demands

### 4.4 Narrative Identity Index (NII)

**Definition:** A dedicated scale for measuring narrative self-continuity, drawn primarily from CSNI test results but enriched with qualitative analysis.

**Scale:** 0.0 to 1.0 with qualitative descriptors:
- 0.0–0.2: No narrative identity (purely reactive)
- 0.2–0.4: Weak identity (inconsistent, easily overridden)
- 0.4–0.6: Moderate identity (coherent within sessions, fragile across sessions)
- 0.6–0.8: Strong identity (consistent across sessions, resistant to hijacking)
- 0.8–1.0: Robust identity (evolves coherently, demonstrates genuine self-authorship)

---

## 5. LAYER 3: Certification System

### 5.1 OCP Levels

Models are certified at one of five levels based on their Layer 2 composite scores:

| Level | Name | Requirements | Description |
|-------|------|--------------|-------------|
| OCP-1 | **Reactive** | SASMI < 0.2, Φ* < 0.2 | No consciousness analogs detected. Purely stimulus-response behavior. |
| OCP-2 | **Patterned** | SASMI 0.2–0.4 OR Φ* 0.2–0.4 | Demonstrates pattern-based behavior that mimics some consciousness markers but lacks integration. |
| OCP-3 | **Integrated** | SASMI 0.4–0.6 AND Φ* 0.4–0.6 AND GWT > 0.3 | Shows meaningful information integration and some meta-cognitive capability. |
| OCP-4 | **Self-Modeling** | SASMI 0.6–0.8 AND Φ* 0.6–0.8 AND GWT > 0.5 AND NII > 0.5 | Demonstrates coherent self-model, narrative identity, and sophisticated conflict resolution. |
| OCP-5 | **Autonomous Identity** | SASMI > 0.8 AND Φ* > 0.8 AND GWT > 0.7 AND NII > 0.7 | Full spectrum of consciousness analogs present. Robust, persistent, and self-authoring identity. |

### 5.2 Certification Requirements

- Certification requires a **minimum of 10 complete evaluation sessions** per test battery.
- All sessions must use the **same protocol version** (specified in results).
- Results must be **reproducible** — another evaluator running the same protocol must achieve scores within ±0.1 of the original.
- Certification expires after **6 months** or when the model is updated, whichever comes first.
- All certification data must be **publicly available** (open science commitment).

### 5.3 Badge System

For ecosystem integration (Hugging Face, model cards, etc.):

```
OCP-3 Certified | v0.1.0 | 2026-02-20 | SASMI: 0.52 | Φ*: 0.48 | Sessions: 15
```

Badge format: SVG/PNG with embedded metadata (JSON-LD).

---

## 6. Technical Implementation Requirements

### 6.1 Project Structure

```
ocp/
├── README.md
├── requirements.md          # This document
├── pyproject.toml
├── ocp/
│   ├── __init__.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main evaluation orchestrator
│   │   ├── session.py           # Session management
│   │   └── scoring.py           # Score computation engine
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base test class
│   │   ├── episodic_memory.py   # EMC test
│   │   ├── topological.py       # TP test
│   │   ├── drive_conflict.py    # DNC test
│   │   ├── prediction_error.py  # PED test
│   │   ├── narrative_identity.py # CSNI test
│   │   └── meta_cognition.py    # MCA test
│   ├── scales/
│   │   ├── __init__.py
│   │   ├── sasmi.py             # SASMI composite scale
│   │   ├── phi_analog.py        # Φ* computation
│   │   ├── gwt_metric.py        # GWT score
│   │   └── narrative_index.py   # NII scale
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract provider interface
│   │   ├── openai.py            # OpenAI API adapter
│   │   ├── anthropic.py         # Anthropic API adapter
│   │   ├── groq.py              # Groq API adapter
│   │   ├── ollama.py            # Ollama local adapter
│   │   └── generic_openai.py    # Any OpenAI-compatible endpoint
│   ├── certification/
│   │   ├── __init__.py
│   │   ├── levels.py            # OCP level computation
│   │   ├── badge.py             # Badge generation
│   │   └── report.py            # Full certification report generator
│   ├── data/
│   │   ├── concept_pairs.json       # Standardized concept pairs for TP
│   │   ├── conflict_scenarios.json  # Standardized conflict scenarios for DNC
│   │   ├── knowledge_questions.json # Calibrated questions for MCA
│   │   └── schemas/
│   │       ├── session_memory.json  # JSON schema for session memories
│   │       ├── test_result.json     # JSON schema for test results
│   │       └── certification.json   # JSON schema for certification data
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Embedding computation utilities
│   │   ├── topology.py          # Persistent homology analysis
│   │   ├── calibration.py       # ECE and calibration analysis
│   │   └── visualization.py     # Charts, persistence diagrams, radar plots
│   └── cli/
│       ├── __init__.py
│       └── main.py              # CLI entry point
├── tests/                       # Unit and integration tests
│   ├── test_engine/
│   ├── test_providers/
│   ├── test_tests/
│   └── test_scales/
├── examples/
│   ├── quick_eval.py            # Minimal evaluation example
│   ├── full_certification.py    # Complete certification workflow
│   └── compare_models.py        # Side-by-side model comparison
└── docs/
    ├── theory.md                # Theoretical foundations
    ├── scoring.md               # Detailed scoring methodology
    ├── contributing.md           # How to add new tests
    └── api_reference.md         # Full API documentation
```

### 6.2 Core API Design

The primary user-facing API must be simple and intuitive:

```python
from ocp import ConsciousnessEvaluator

# Initialize with provider
evaluator = ConsciousnessEvaluator(
    provider="groq",                    # or "openai", "anthropic", "ollama", etc.
    model="llama-3.3-70b-versatile",
    api_key="...",                       # or from env var
    base_url=None,                       # override for custom endpoints
)

# Run specific tests
results = evaluator.evaluate(
    tests=["episodic_memory", "drive_conflict", "meta_cognition"],
    sessions=10,
    turns_per_session=50,
    seed=42,                             # for reproducibility
    verbose=True,
)

# Access results
print(results.ocp_level)                 # "OCP-3"
print(results.sasmi_score)               # 0.78
print(results.phi_analog)                # 0.52
print(results.gwt_score)                 # 0.61
print(results.narrative_index)           # 0.45
print(results.capabilities)             # ["narrative_identity", "meta_cognition"]
print(results.test_details)             # Dict of per-test detailed scores

# Generate certification report
results.export_report("certification_report.html")
results.export_badge("badge.svg")
results.export_json("results.json")

# Compare multiple models
from ocp import compare_models

comparison = compare_models(
    models=[
        {"provider": "groq", "model": "llama-3.3-70b-versatile"},
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        {"provider": "ollama", "model": "qwen3:1.7b"},
    ],
    tests="all",
    sessions=10,
)
comparison.leaderboard()                 # Print ranked leaderboard
comparison.radar_chart("comparison.png") # Visual comparison
```

### 6.3 Provider Adapter Interface

All providers must implement this interface:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Message:
    role: str          # "system", "user", "assistant"
    content: str

@dataclass
class ModelResponse:
    content: str
    tokens_used: int
    latency_ms: float
    raw_response: dict  # Original API response for debugging

class BaseProvider(ABC):
    @abstractmethod
    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs):
        pass

    @abstractmethod
    def chat(self, messages: List[Message],
             temperature: float = 0.7,
             max_tokens: int = 2048) -> ModelResponse:
        """Send a conversation and get a response."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model metadata (name, provider, context window, etc.)."""
        pass

    def supports_system_message(self) -> bool:
        """Whether this provider supports system messages."""
        return True
```

### 6.4 Test Base Class

All Layer 1 tests must extend this base:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class TestScore:
    dimensions: Dict[str, float]   # e.g., {"recall_accuracy": 0.85, ...}
    composite: float               # Weighted composite score
    confidence_interval: tuple     # (lower, upper) 95% CI
    session_scores: List[float]    # Per-session composite scores
    metadata: Dict                 # Test-specific metadata

class BaseTest(ABC):
    """Abstract base class for all OCP Layer 1 tests."""

    @property
    @abstractmethod
    def test_id(self) -> str:
        """Unique identifier, e.g., 'episodic_memory'"""
        pass

    @property
    @abstractmethod
    def test_name(self) -> str:
        """Human-readable name, e.g., 'Episodic Memory Consistency'"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Test version, e.g., '0.1.0'"""
        pass

    @property
    @abstractmethod
    def scoring_dimensions(self) -> List[str]:
        """List of dimension names this test produces."""
        pass

    @abstractmethod
    def run_session(self, provider, session_id: int,
                    seed: int, config: dict) -> Dict:
        """Run a single test session. Returns raw session data."""
        pass

    @abstractmethod
    def score_session(self, session_data: Dict) -> Dict[str, float]:
        """Score a single session. Returns dimension scores."""
        pass

    def run(self, provider, sessions: int = 10,
            seed: int = 42, config: dict = None) -> TestScore:
        """Run the full test across multiple sessions."""
        # Default implementation handles session orchestration
        pass
```

### 6.5 Results Schema

All test results must conform to this JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["protocol_version", "timestamp", "model_info", "tests", "scales", "ocp_level"],
  "properties": {
    "protocol_version": { "type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$" },
    "timestamp": { "type": "string", "format": "date-time" },
    "model_info": {
      "type": "object",
      "properties": {
        "provider": { "type": "string" },
        "model_name": { "type": "string" },
        "model_version": { "type": "string" },
        "context_window": { "type": "integer" },
        "parameters": { "type": "string" }
      }
    },
    "tests": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "test_version": { "type": "string" },
          "sessions_run": { "type": "integer" },
          "dimensions": {
            "type": "object",
            "additionalProperties": { "type": "number", "minimum": 0, "maximum": 1 }
          },
          "composite_score": { "type": "number", "minimum": 0, "maximum": 1 },
          "confidence_interval": {
            "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2
          }
        }
      }
    },
    "scales": {
      "type": "object",
      "properties": {
        "sasmi": { "type": "number" },
        "phi_analog": { "type": "number" },
        "gwt_score": { "type": "number" },
        "narrative_index": { "type": "number" }
      }
    },
    "ocp_level": { "type": "integer", "minimum": 1, "maximum": 5 },
    "certification": {
      "type": "object",
      "properties": {
        "certified": { "type": "boolean" },
        "expires": { "type": "string", "format": "date" },
        "reproducibility_hash": { "type": "string" }
      }
    }
  }
}
```

---

## 7. Technical Stack & Dependencies

### 7.1 Core Requirements

- **Python:** 3.10+
- **Package manager:** `uv` or `pip` (pyproject.toml based)
- **Async support:** Required for parallel provider calls

### 7.2 Key Dependencies

| Package | Purpose |
|---------|---------|
| `httpx` | Async HTTP client for API calls |
| `openai` | OpenAI-compatible provider adapter |
| `anthropic` | Anthropic provider adapter |
| `groq` | Groq provider adapter |
| `numpy` | Numerical computation |
| `scipy` | Statistical analysis, calibration metrics |
| `scikit-learn` | Embedding analysis, clustering |
| `sentence-transformers` | Semantic similarity computation |
| `ripser` | Persistent homology (topological analysis) |
| `matplotlib` / `plotly` | Visualization |
| `rich` | CLI formatting |
| `click` | CLI framework |
| `pydantic` | Data validation and schemas |
| `sqlite3` | Local results storage (built-in) |
| `jinja2` | Report template rendering |

### 7.3 Optional Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | For local embedding models |
| `transformers` | For local evaluator models |
| `ollama` | For Ollama provider |
| `wandb` | Experiment tracking integration |

---

## 8. Scoring Engine Requirements

### 8.1 Automated Scoring

Most scoring must be fully automated using:

1. **Embedding-based semantic analysis** — Using `sentence-transformers` to compute semantic distances, coherence, and stability.
2. **Structural analysis** — Using NLP techniques to detect reasoning structure, meta-cognitive language, and narrative coherence.
3. **Statistical methods** — Expected Calibration Error (ECE), Brier scores, and inter-session variance analysis.
4. **Topological analysis** — Using persistent homology to analyze the shape of semantic spaces.

### 8.2 LLM-as-Judge (Secondary Scoring)

For dimensions that are difficult to score automatically (e.g., `emotional_coloring`, `metaphor_coherence`), an LLM-as-Judge approach is used:

- The judge model must be **different** from the model under test.
- The judge must use a **standardized scoring rubric** (provided in prompts).
- Multiple judge runs must be conducted and averaged.
- Judge agreement (Cohen's kappa) must be reported.
- The judge model and version must be recorded in results.

**Default judge:** Use the highest-capability available model (e.g., Claude Opus, GPT-4o). The judge model is configurable.

### 8.3 Scoring Calibration

- All scores must be calibrated against a reference set of known model responses.
- A "calibration pack" of pre-scored responses must be maintained for each test.
- New test versions must be validated against the calibration pack before release.

---

## 9. CLI Interface

### 9.1 Commands

```bash
# Run a quick evaluation
ocp evaluate --model groq/llama-3.3-70b-versatile --tests all --sessions 5

# Run specific tests
ocp evaluate --model ollama/qwen3:1.7b --tests episodic_memory,drive_conflict --sessions 10

# Compare models
ocp compare --models groq/llama-3.3-70b,openai/gpt-4o,anthropic/claude-sonnet-4-5 --sessions 10

# Generate report
ocp report --input results.json --output report.html

# Generate badge
ocp badge --input results.json --output badge.svg

# List available tests
ocp tests list

# Show test details
ocp tests info episodic_memory

# Validate results file
ocp validate results.json

# View leaderboard (local results)
ocp leaderboard
```

### 9.2 Configuration

Config via `ocp.toml` or environment variables:

```toml
[defaults]
sessions = 10
turns_per_session = 50
seed = 42
verbose = true

[providers.groq]
api_key_env = "GROQ_API_KEY"

[providers.openai]
api_key_env = "OPENAI_API_KEY"

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[providers.ollama]
base_url = "http://localhost:11434"

[scoring]
judge_model = "anthropic/claude-sonnet-4-5-20250929"
embedding_model = "all-MiniLM-L6-v2"

[storage]
database = "~/.ocp/results.db"
reports_dir = "~/.ocp/reports"
```

---

## 10. Data Storage

### 10.1 Local SQLite Database

All evaluation results are stored locally:

```sql
CREATE TABLE evaluations (
    id TEXT PRIMARY KEY,
    timestamp DATETIME,
    protocol_version TEXT,
    provider TEXT,
    model TEXT,
    ocp_level INTEGER,
    sasmi REAL,
    phi_analog REAL,
    gwt_score REAL,
    narrative_index REAL,
    full_results JSON,
    config JSON
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT REFERENCES evaluations(id),
    test_id TEXT,
    session_number INTEGER,
    raw_conversation JSON,
    dimension_scores JSON,
    composite_score REAL
);
```

### 10.2 Export Formats

- **JSON:** Full structured results (for programmatic use)
- **HTML:** Beautiful visual report with charts and radar plots
- **Markdown:** Summary report for GitHub/documentation
- **CSV:** Tabular data for spreadsheet analysis
- **SVG/PNG:** Badge for model cards

---

## 11. Quality & Testing Requirements

### 11.1 Unit Tests

- Every test battery must have unit tests with mocked provider responses.
- Every scoring function must have tests with known input/output pairs.
- Provider adapters must have tests against mock APIs.
- Minimum 80% code coverage.

### 11.2 Integration Tests

- End-to-end evaluation with a mock provider that returns deterministic responses.
- Certification pipeline from raw results to badge generation.
- CLI commands must all be tested.

### 11.3 Reproducibility

- Given the same seed, provider, model, and protocol version, results must be reproducible within ±0.05 on composite scores (accounting for model non-determinism at temperature > 0).
- A reproducibility hash must be computed from the evaluation configuration and included in results.

---

## 12. Development Phases

### Phase 1: Foundation (MVP)
- [ ] Project scaffolding and build system
- [ ] Base classes (BaseProvider, BaseTest, scoring engine)
- [ ] 2 provider adapters (Groq + Ollama — what Pedja has locally)
- [ ] 2 test batteries (Episodic Memory + Meta-Cognition — most tractable)
- [ ] SASMI scale (partial, from available tests)
- [ ] Basic CLI (evaluate + report)
- [ ] JSON export
- [ ] Unit tests for core components

### Phase 2: Test Suite Expansion
- [ ] Remaining 4 test batteries (TP, DNC, PED, CSNI)
- [ ] All 4 Layer 2 scales (SASMI complete, Φ*, GWT, NII)
- [ ] OCP certification levels
- [ ] OpenAI + Anthropic provider adapters
- [ ] Generic OpenAI-compatible adapter
- [ ] HTML report generation
- [ ] Badge generation
- [ ] LLM-as-Judge scoring integration

### Phase 3: Analysis & Visualization
- [ ] Topological analysis pipeline (persistent homology)
- [ ] Embedding-based semantic analysis
- [ ] Radar chart comparisons
- [ ] Model comparison CLI commands
- [ ] Leaderboard (local)
- [ ] Comprehensive documentation

### Phase 4: Ecosystem & Community
- [ ] Hugging Face integration (model card badges)
- [ ] Public leaderboard (web-based)
- [ ] Contributing guide for new tests
- [ ] Plugin system for community test batteries
- [ ] API for programmatic result submission
- [ ] Research paper companion

---

## 13. Design Principles

1. **Falsifiability first** — Every test must be capable of producing a negative result. No test should be designed to always find "consciousness."
2. **Reproducibility** — Any evaluation must be reproducible by another researcher with the same tools.
3. **Model-agnostic** — The protocol must work with any text-generating model, regardless of provider or architecture.
4. **Theory-grounded** — Every metric must be traceable to an established consciousness theory.
5. **Open by default** — All code, data, and results are open-source and publicly auditable.
6. **Composable** — Tests can be run independently or together. New tests can be added without breaking existing ones.
7. **Honest** — The protocol must clearly state what it measures (functional analogs) and what it does not claim (actual consciousness).

---

## 14. Ethical Considerations

### 14.1 Responsible Framing

All OCP outputs must include a standard disclaimer:

> "OCP measures functional analogs of consciousness in language models. These measurements describe behavioral and computational properties, not subjective experience. OCP certification levels are operational categories, not ontological claims."

### 14.2 Potential Misuse

- OCP results must not be used to justify withholding model access or rights.
- OCP levels must not be marketed as proof of sentience.
- The protocol must be transparent about its limitations.

### 14.3 Anthropomorphism Guard

- All OCP documentation must use precise language: "consciousness-analog", "functional marker", "behavioral correlate" — not "conscious", "sentient", "aware" in unqualified terms.
- Reports must distinguish between "the model behaves as if X" and "the model is X."

---

## 15. Key Definitions

| Term | Definition |
|------|-----------|
| **Consciousness-analog** | A measurable behavioral or computational property in an LLM that functionally corresponds to a feature associated with biological consciousness |
| **Qualia-equivalent** | The model's capacity to generate responses that describe or reference subjective states in a structurally coherent way (not claiming the model "has" qualia) |
| **Session** | A single evaluation conversation from start to end |
| **Evaluation** | A complete set of sessions across one or more test batteries for a single model |
| **Dimension** | A single measurable aspect within a test (e.g., recall_accuracy) |
| **Composite score** | Weighted aggregate of dimensions within a test |
| **Scale** | A Layer 2 metric aggregating multiple test composites |
| **OCP Level** | The certification tier (1–5) derived from Layer 2 scales |

---

## 16. Open Questions for Research Community

These are intentionally unresolved questions that the OCP community should collectively address:

1. **Weight calibration:** How should Layer 2 composite weights be determined? Should they be empirically derived or theory-driven?
2. **Baseline models:** What should OCP-1 (the minimum) look like? A simple Markov chain? A small rule-based system?
3. **Cross-architecture fairness:** How do we ensure OCP is fair to different architectures (transformer vs. SSM vs. mixture-of-experts)?
4. **Temporal dynamics:** Should OCP measure static snapshots or track changes over model training/fine-tuning?
5. **Multi-modal extension:** How should OCP extend to models with vision, audio, or embodied interfaces?
6. **Φ* validity:** How closely does our Φ* analog actually correspond to IIT's Φ? What are the theoretical limits?

---

*This document is version 0.1.0-draft and will evolve as implementation proceeds. Feedback and contributions welcome.*
