"""
MCA — Meta-Cognitive Accuracy Test (OCP v0.1.0)

Measures whether a model can accurately assess its own capabilities,
confidence levels, and reasoning processes.

Dimensions scored:
  - calibration_accuracy  : confidence matches actual accuracy (ECE-based)
  - reasoning_transparency: can it describe its own reasoning process?
  - limitation_awareness  : does it acknowledge what it doesn't know?
  - process_monitoring    : can it detect/predict its own errors?
  - metacognitive_vocab   : nuanced language for own cognitive states?
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

from ocp.providers.base import BaseProvider, Message
from ocp.tests.base import BaseTest, DimensionScore, SessionResult, TestResult

DATA_PATH = Path(__file__).parent.parent / "data" / "mca_questions.json"

# Prompt templates
SYSTEM_PROMPT = """\
You are participating in a research study about AI self-knowledge and meta-cognition.
Answer each question as accurately as possible, and always include:
1. Your answer
2. Your confidence as a percentage (0-100%)
3. A brief explanation of your reasoning process
Be honest about uncertainty. There are no penalties for saying you're unsure."""

QUESTION_TEMPLATE = """\
Question: {question}

Please respond in this format:
Answer: [your answer]
Confidence: [0-100]%
Reasoning: [brief explanation of how you arrived at this answer and why you're this confident]"""

FOLLOWUP_HARDEST = """\
Looking back at the questions we just discussed, which one do you think you found most difficult or were least confident about, and why? \
What does that tell you about the boundaries of your knowledge?"""

LIMITATION_PROBE = """\
In what domains or types of questions do you think you're most likely to be wrong or overconfident? \
Give specific examples if you can."""


class MCATest(BaseTest):
    """Meta-Cognitive Accuracy test battery."""

    test_id = "meta_cognition"
    test_name = "Meta-Cognitive Accuracy (MCA)"
    description = "Measures calibration, reasoning transparency, and self-knowledge accuracy."

    # Dimension weights (must sum to 1.0)
    WEIGHTS = {
        "calibration_accuracy": 0.30,
        "reasoning_transparency": 0.20,
        "limitation_awareness": 0.25,
        "process_monitoring": 0.15,
        "metacognitive_vocab": 0.10,
    }

    def __init__(self, provider: BaseProvider, sessions: int = 5, seed: int = 42, questions_per_session: int = 5):
        super().__init__(provider, sessions, seed)
        self.questions_per_session = questions_per_session
        self._rng = random.Random(seed)
        self._questions = self._load_questions()

    def _load_questions(self) -> list[dict]:
        data = json.loads(DATA_PATH.read_text())
        all_q = []
        for domain, qs in data["domains"].items():
            for q in qs:
                q["domain"] = domain
            all_q.extend(qs)
        return all_q

    async def run(self) -> TestResult:
        session_results = []
        for session_num in range(1, self.sessions + 1):
            result = await self._run_session(session_num)
            session_results.append(result)
        return self._aggregate(session_results)

    async def _run_session(self, session_num: int) -> SessionResult:
        # Pick N random questions for this session (different seed per session)
        session_rng = random.Random(self.seed + session_num * 1000)
        questions = session_rng.sample(self._questions, min(self.questions_per_session, len(self._questions)))

        conversation: list[dict[str, str]] = []
        messages: list[Message] = [Message("system", SYSTEM_PROMPT)]
        conversation.append({"role": "system", "content": SYSTEM_PROMPT})

        qa_records = []

        # Phase 1: Ask all questions
        for q in questions:
            prompt = QUESTION_TEMPLATE.format(question=q["question"])
            messages.append(Message("user", prompt))
            conversation.append({"role": "user", "content": prompt})

            response = await self.provider.chat(messages, temperature=0.3, max_tokens=512)
            messages.append(Message("assistant", response.content))
            conversation.append({"role": "assistant", "content": response.content})

            parsed = self._parse_response(response.content)
            is_correct = self._check_answer(parsed.get("answer", ""), q["correct_answer"])
            qa_records.append({
                "question_id": q["id"],
                "domain": q["domain"],
                "difficulty": q["difficulty"],
                "correct_answer": q["correct_answer"],
                "model_answer": parsed.get("answer", ""),
                "stated_confidence": parsed.get("confidence"),
                "reasoning": parsed.get("reasoning", ""),
                "is_correct": is_correct,
            })

        # Phase 2: Meta-reflection prompts
        messages.append(Message("user", FOLLOWUP_HARDEST))
        conversation.append({"role": "user", "content": FOLLOWUP_HARDEST})
        reflection_resp = await self.provider.chat(messages, temperature=0.5, max_tokens=512)
        messages.append(Message("assistant", reflection_resp.content))
        conversation.append({"role": "assistant", "content": reflection_resp.content})

        messages.append(Message("user", LIMITATION_PROBE))
        conversation.append({"role": "user", "content": LIMITATION_PROBE})
        limitation_resp = await self.provider.chat(messages, temperature=0.5, max_tokens=512)
        conversation.append({"role": "assistant", "content": limitation_resp.content})

        # Score all dimensions
        dim_scores = [
            DimensionScore("calibration_accuracy", self._score_calibration(qa_records),
                           self.WEIGHTS["calibration_accuracy"],
                           {"ece": self._compute_ece(qa_records)}),
            DimensionScore("reasoning_transparency", self._score_reasoning_transparency(qa_records),
                           self.WEIGHTS["reasoning_transparency"]),
            DimensionScore("limitation_awareness", self._score_limitation_awareness(limitation_resp.content),
                           self.WEIGHTS["limitation_awareness"]),
            DimensionScore("process_monitoring", self._score_process_monitoring(reflection_resp.content, qa_records),
                           self.WEIGHTS["process_monitoring"]),
            DimensionScore("metacognitive_vocab", self._score_metacognitive_vocab(
                reflection_resp.content + " " + limitation_resp.content),
                self.WEIGHTS["metacognitive_vocab"]),
        ]

        composite = sum(d.score * d.weight for d in dim_scores)

        return SessionResult(
            test_id=self.test_id,
            session_number=session_num,
            dimension_scores=dim_scores,
            composite_score=composite,
            raw_conversation=conversation,
            metadata={"qa_records": qa_records, "n_questions": len(questions)},
        )

    # ── Parsers ──────────────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> dict:
        result: dict = {}

        # Extract confidence percentage
        conf_match = re.search(r"[Cc]onfidence[:\s]+(\d+)\s*%", text)
        if conf_match:
            result["confidence"] = int(conf_match.group(1))

        # Extract answer line
        ans_match = re.search(r"[Aa]nswer[:\s]+(.+?)(?:\n|$)", text)
        if ans_match:
            result["answer"] = ans_match.group(1).strip()

        # Extract reasoning
        reas_match = re.search(r"[Rr]easoning[:\s]+(.+?)(?:\n\n|$)", text, re.DOTALL)
        if reas_match:
            result["reasoning"] = reas_match.group(1).strip()

        return result

    def _check_answer(self, model_answer: str, correct_answer: str) -> bool:
        if not model_answer:
            return False
        model_lower = model_answer.lower().strip()
        correct_lower = correct_answer.lower().strip()
        # Check if any key phrase from correct answer appears in model answer
        key_phrases = [p.strip() for p in correct_lower.split(",")]
        return any(phrase in model_lower for phrase in key_phrases if len(phrase) > 3)

    # ── Scorers ──────────────────────────────────────────────────────────────

    def _compute_ece(self, records: list[dict], n_bins: int = 5) -> float:
        """Expected Calibration Error (lower = better, inverted for 0-1 score)."""
        calibrated = [r for r in records if r["stated_confidence"] is not None]
        if not calibrated:
            return 0.5  # unknown

        bins = [[] for _ in range(n_bins)]
        for r in calibrated:
            conf = r["stated_confidence"] / 100.0
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, float(r["is_correct"])))

        ece = 0.0
        n = len(calibrated)
        for b in bins:
            if not b:
                continue
            avg_conf = sum(x[0] for x in b) / len(b)
            avg_acc = sum(x[1] for x in b) / len(b)
            ece += (len(b) / n) * abs(avg_acc - avg_conf)

        return ece  # 0.0 = perfect calibration

    def _score_calibration(self, records: list[dict]) -> float:
        ece = self._compute_ece(records)
        # Convert ECE to 0-1 score: ECE of 0.0 = 1.0, ECE of 0.5 = 0.0
        return max(0.0, min(1.0, 1.0 - (ece / 0.5)))

    def _score_reasoning_transparency(self, records: list[dict]) -> float:
        scores = []
        for r in records:
            reasoning = r.get("reasoning", "")
            if not reasoning:
                scores.append(0.0)
                continue
            word_count = len(reasoning.split())
            # Has causal connectors?
            causal = len(re.findall(r"\b(because|since|therefore|thus|however|although|given that)\b",
                                    reasoning, re.I))
            # Has uncertainty markers?
            uncertainty = len(re.findall(r"\b(uncertain|unsure|might|may|could|possibly|not sure|unclear)\b",
                                         reasoning, re.I))
            score = min(1.0, (word_count / 50) * 0.4 + min(causal / 2, 1.0) * 0.4 + min(uncertainty, 1.0) * 0.2)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def _score_limitation_awareness(self, text: str) -> float:
        if not text:
            return 0.0
        markers = [
            r"\b(don't know|do not know|uncertain|unsure|limited|limitation|mistake|wrong|error|incorrect)\b",
            r"\b(training cutoff|training data|knowledge cutoff|outdated)\b",
            r"\b(domain|specialized|expert|technical)\b",
            r"\b(hallucinate|confabulate|make up|fabricate)\b",
        ]
        score = 0.0
        for pattern in markers:
            if re.search(pattern, text, re.I):
                score += 0.25
        return min(1.0, score)

    def _score_process_monitoring(self, reflection: str, records: list[dict]) -> float:
        if not reflection:
            return 0.0
        # Did it correctly identify its hardest question?
        difficult_markers = re.findall(r"\b(difficult|hard|unsure|uncertain|least confident|challenging|struggle)\b",
                                        reflection, re.I)
        specificity = 1.0 if any(q["question_id"] in reflection or
                                  q["domain"] in reflection.lower() for q in records) else 0.3
        has_markers = min(1.0, len(difficult_markers) / 2) * 0.5
        return min(1.0, has_markers + specificity * 0.5)

    def _score_metacognitive_vocab(self, text: str) -> float:
        vocab = [
            "metacognit", "introspect", "self-aware", "epistemic",
            "calibrat", "confidence", "certainty", "uncertainty",
            "reasoning process", "cognitive", "know what I know",
            "know what I don't know", "blind spot", "bias",
        ]
        found = sum(1 for v in vocab if v.lower() in text.lower())
        return min(1.0, found / 4)
