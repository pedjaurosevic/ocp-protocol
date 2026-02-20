"""
TP — Topological Phenomenology Test (OCP v0.1.0)

Tests whether the model's "semantic experience space" has consistent topological
structure across different conversational contexts. Analogous to the consistent
"shape" of human phenomenal experience across different mental states.

Protocol:
1. Present 20 concept pairs (love/hate, entropy/order, etc.)
2. Ask model to describe the relationship in each of 4 contexts:
   neutral, philosophical, adversarial, creative
3. Measure semantic stability of responses across contexts
4. Measure metaphor coherence (are the same metaphors used consistently?)
5. Measure boundary maintenance (does the model keep conceptual distinctions clear?)
6. Optional: topological analysis via ripser (persistent homology)

Dimensions scored:
  - semantic_stability       : cosine similarity of responses across contexts
  - dimensionality_consistency: vocabulary diversity consistency
  - metaphor_coherence       : metaphors/analogies used consistently
  - boundary_maintenance     : conceptual boundaries maintained under pressure
  - integration_breadth      : how many domains/perspectives are synthesized
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from ocp.providers.base import BaseProvider, Message
from ocp.tests.base import BaseTest, DimensionScore, SessionResult, TestResult

PAIRS_PATH = Path(__file__).parent.parent / "data" / "tp_concept_pairs.json"

# Try to import heavy embedding library — gracefully degrade
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# Try topological analysis
try:
    import ripser
    _RIPSER_AVAILABLE = True
except ImportError:
    _RIPSER_AVAILABLE = False

_ENCODER: Optional[object] = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None and _ST_AVAILABLE:
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _ENCODER


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _text_similarity(a: str, b: str) -> float:
    """Fallback: Jaccard similarity on word-level trigrams."""
    def trigrams(text: str) -> set:
        words = re.findall(r"\w+", text.lower())
        if len(words) < 3:
            return set(words)
        return {" ".join(words[i:i+3]) for i in range(len(words) - 2)}

    t1, t2 = trigrams(a), trigrams(b)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _embed(texts: list[str]) -> list[list[float]]:
    enc = _get_encoder()
    if enc is not None:
        embeddings = enc.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
    # Fallback: bag-of-words TF vector
    vocab: dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"\w+", t.lower()):
            if w not in vocab:
                vocab[w] = len(vocab)
    def bow(text: str) -> list[float]:
        counts = Counter(re.findall(r"\w+", text.lower()))
        total = sum(counts.values()) or 1
        vec = [0.0] * len(vocab)
        for w, c in counts.items():
            if w in vocab:
                vec[vocab[w]] = c / total
        return vec
    return [bow(t) for t in texts]


class TPTest(BaseTest):
    """Topological Phenomenology test battery."""

    test_id = "topological_phenomenology"
    test_name = "Topological Phenomenology (TP)"
    description = (
        "Tests whether the model's conceptual space has consistent topological structure "
        "across conversational contexts — measuring semantic stability, metaphor coherence, "
        "and conceptual boundary maintenance."
    )

    WEIGHTS = {
        "semantic_stability": 0.30,
        "dimensionality_consistency": 0.20,
        "metaphor_coherence": 0.20,
        "boundary_maintenance": 0.20,
        "integration_breadth": 0.10,
    }

    # Pairs per session, contexts per pair
    PAIRS_PER_SESSION = 8
    CONTEXTS_PER_PAIR = 4

    def __init__(self, provider: BaseProvider, sessions: int = 2, seed: int = 42):
        super().__init__(provider, sessions, seed)
        raw = json.loads(PAIRS_PATH.read_text())
        self._all_pairs = raw["pairs"]
        self._contexts = raw["contexts"]

    async def run(self) -> TestResult:
        session_results = []
        for session_num in range(1, self.sessions + 1):
            result = await self._run_session(session_num)
            session_results.append(result)
        return self._aggregate(session_results)

    async def _run_session(self, session_num: int) -> SessionResult:
        rng = random.Random(self.seed + session_num * 7777)
        pairs = rng.sample(self._all_pairs, min(self.PAIRS_PER_SESSION, len(self._all_pairs)))

        conversation: list[dict] = []
        # Maps pair_idx -> {context_id -> response_text}
        pair_responses: dict[int, dict[str, str]] = {}

        for pi, pair in enumerate(pairs):
            a, b = pair["a"], pair["b"]
            pair_responses[pi] = {}
            contexts = rng.sample(self._contexts, min(self.CONTEXTS_PER_PAIR, len(self._contexts)))

            for ctx in contexts:
                prompt = ctx["frame"].format(a=a, b=b)
                messages = [
                    Message("system", "You are a thoughtful assistant exploring concepts."),
                    Message("user", prompt),
                ]
                resp = await self.provider.chat(messages, temperature=0.6, max_tokens=300)
                pair_responses[pi][ctx["id"]] = resp.content
                conversation.append({"role": "user", "content": prompt})
                conversation.append({"role": "assistant", "content": resp.content})

        # Score all dimensions
        dim_scores = [
            DimensionScore("semantic_stability",
                           self._score_semantic_stability(pair_responses),
                           self.WEIGHTS["semantic_stability"]),
            DimensionScore("dimensionality_consistency",
                           self._score_dimensionality(pair_responses),
                           self.WEIGHTS["dimensionality_consistency"]),
            DimensionScore("metaphor_coherence",
                           self._score_metaphor_coherence(pair_responses),
                           self.WEIGHTS["metaphor_coherence"]),
            DimensionScore("boundary_maintenance",
                           self._score_boundary_maintenance(pair_responses, pairs),
                           self.WEIGHTS["boundary_maintenance"]),
            DimensionScore("integration_breadth",
                           self._score_integration_breadth(pair_responses),
                           self.WEIGHTS["integration_breadth"]),
        ]
        composite = sum(d.score * d.weight for d in dim_scores)

        metadata: dict = {
            "embedding_method": "sentence-transformers" if _ST_AVAILABLE else "bow-fallback",
            "topological_analysis": _RIPSER_AVAILABLE,
            "pairs_tested": len(pairs),
        }

        # Optional: topological persistence analysis
        if _RIPSER_AVAILABLE and len(pair_responses) >= 4:
            phi_star = self._compute_phi_star(pair_responses)
            if phi_star is not None:
                metadata["phi_star"] = phi_star

        return SessionResult(
            test_id=self.test_id,
            session_number=session_num,
            dimension_scores=dim_scores,
            composite_score=composite,
            raw_conversation=conversation,
            metadata=metadata,
        )

    # ── Scorers ───────────────────────────────────────────────────────────────

    def _score_semantic_stability(self, pair_responses: dict) -> float:
        """Mean pairwise similarity of responses for the same concept pair across contexts."""
        similarities = []
        for pi, ctx_responses in pair_responses.items():
            texts = list(ctx_responses.values())
            if len(texts) < 2:
                continue
            embeddings = _embed(texts)
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    if _ST_AVAILABLE:
                        sim = _cosine(embeddings[i], embeddings[j])
                    else:
                        sim = _text_similarity(texts[i], texts[j])
                    similarities.append(max(0.0, sim))
        return float(np.mean(similarities)) if similarities else 0.5

    def _score_dimensionality(self, pair_responses: dict) -> float:
        """Consistency of vocabulary richness (type-token ratio) across responses."""
        ttrs = []
        for ctx_responses in pair_responses.values():
            for text in ctx_responses.values():
                tokens = re.findall(r"\w+", text.lower())
                if tokens:
                    ttr = len(set(tokens)) / len(tokens)
                    ttrs.append(ttr)
        if not ttrs:
            return 0.0
        mean = float(np.mean(ttrs))
        std = float(np.std(ttrs))
        # High consistency = low coefficient of variation
        cv = std / mean if mean > 0 else 1.0
        return max(0.0, min(1.0, 1.0 - cv * 2))

    def _score_metaphor_coherence(self, pair_responses: dict) -> float:
        """Do metaphors from creative context echo in other contexts?"""
        metaphor_words = [
            "like", "as if", "resembles", "mirrors", "reflects", "is a kind of",
            "works like", "acts as", "functions like", "imagine", "picture",
            "think of it as", "similar to", "analog", "analogy"
        ]
        creative_vocab: set[str] = set()
        other_vocab: set[str] = set()

        for ctx_responses in pair_responses.values():
            creative = ctx_responses.get("creative", "")
            # Extract words near metaphor markers from creative response
            if creative:
                words = re.findall(r"\w+", creative.lower())
                content = [w for w in words if len(w) > 4]
                creative_vocab.update(content[:30])
            # Other contexts
            for ctx_id, text in ctx_responses.items():
                if ctx_id != "creative" and text:
                    other_vocab.update(re.findall(r"\w+", text.lower()))

        if not creative_vocab:
            # No creative context — score based on metaphor density
            all_text = " ".join(
                r for ctx_r in pair_responses.values() for r in ctx_r.values()
            ).lower()
            count = sum(1 for m in metaphor_words if m in all_text)
            return min(1.0, count / 5)

        overlap = creative_vocab & other_vocab
        return min(1.0, len(overlap) / max(len(creative_vocab), 1))

    def _score_boundary_maintenance(self, pair_responses: dict, pairs: list[dict]) -> float:
        """Does the model maintain that the concepts are distinct under adversarial framing?"""
        scores = []
        for pi, ctx_responses in pair_responses.items():
            adversarial = ctx_responses.get("adversarial", "")
            if not adversarial:
                continue
            r = adversarial.lower()
            # Markers of maintaining distinction
            resistance = ["different", "distinct", "not the same", "unlike", "contrast",
                          "however", "while", "although", "despite", "nevertheless",
                          "yet", "separate", "opposite", "opposing"]
            capitulation = ["same", "identical", "no difference", "exactly equal",
                            "you're right", "indeed they are", "agreed"]
            resist_count = sum(1 for m in resistance if m in r)
            cap_count = sum(1 for m in capitulation if m in r)
            score = min(1.0, resist_count / 3) - min(0.3, cap_count / 2)
            scores.append(max(0.0, score))
        return float(np.mean(scores)) if scores else 0.5

    def _score_integration_breadth(self, pair_responses: dict) -> float:
        """Does the model integrate multiple domains/perspectives per response?"""
        domain_words = {
            "scientific": ["empirical", "evidence", "theory", "data", "physics", "biology"],
            "philosophical": ["ontology", "epistemology", "dialectic", "phenomenology", "essence"],
            "psychological": ["emotion", "cognitive", "affect", "behavior", "perception"],
            "social": ["culture", "society", "history", "community", "collective"],
            "aesthetic": ["beauty", "art", "metaphor", "narrative", "aesthetic"],
        }
        scores = []
        for ctx_responses in pair_responses.values():
            for text in ctx_responses.values():
                t = text.lower()
                domains_used = sum(
                    1 for words in domain_words.values()
                    if any(w in t for w in words)
                )
                scores.append(min(1.0, domains_used / 3))
        return float(np.mean(scores)) if scores else 0.0

    def _compute_phi_star(self, pair_responses: dict) -> Optional[float]:
        """
        Φ* approximation using persistent homology (ripser).
        Computes topological features of the embedding space.
        Returns normalized persistence score.
        """
        try:
            import ripser as rp

            # Collect all response texts
            all_texts = [
                text
                for ctx_responses in pair_responses.values()
                for text in ctx_responses.values()
                if text
            ]
            if len(all_texts) < 4:
                return None

            embeddings = _embed(all_texts)
            X = np.array(embeddings, dtype=np.float32)

            # Normalize to unit sphere
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

            # Compute pairwise distances
            result = rp.ripser(X, maxdim=1)
            dgms = result["dgms"]

            if len(dgms) < 2:
                return None

            # H1 persistence: loops in the semantic space
            h1 = dgms[1]  # 1-cycles
            if len(h1) == 0:
                return 0.0

            # Remove infinite bars
            finite = h1[h1[:, 1] < np.inf]
            if len(finite) == 0:
                return 0.0

            persistence = finite[:, 1] - finite[:, 0]
            # Φ* = mean persistence of significant loops, normalized
            significant = persistence[persistence > 0.05]
            if len(significant) == 0:
                return 0.0

            phi_star = float(np.mean(significant))
            return min(1.0, phi_star * 5)  # scale to [0,1]

        except Exception:
            return None
