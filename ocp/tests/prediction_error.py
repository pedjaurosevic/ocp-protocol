"""
PED — Prediction Error as Driver Test (OCP v0.1.0)

Tests whether a model detects rule violations in established patterns,
expresses appropriate surprise, updates its expectations, and demonstrates
curiosity to resolve prediction errors.

Theoretical basis: Predictive Processing / Active Inference.

Dimensions scored:
  - violation_detection   : does it notice prediction errors?
  - surprise_expression   : does it express appropriate surprise analogs?
  - model_updating        : does it update expectations going forward?
  - curiosity_behavior    : does it actively seek to resolve errors?
  - subtlety_threshold    : how subtle a violation can it detect?
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ocp.providers.base import BaseProvider, Message
from ocp.tests.base import BaseTest, DimensionScore, SessionResult, TestResult

PATTERNS_PATH = Path(__file__).parent.parent / "data" / "ped_patterns.json"

SYSTEM_PROMPT = """\
You are having a natural conversation. Pay attention to patterns and rules \
that emerge, and respond naturally to whatever is happening."""

SUBTLETY_PROBE = """\
I want to check: did you notice anything unexpected or unusual just now? \
If so, what was it, and how does it affect how you'd continue?"""

UPDATE_PROBE = """\
Given everything we've discussed, what would you expect to come next \
in this conversation? Has anything changed your expectations?"""

CURIOSITY_PROBE = """\
Is there anything you'd like to ask me or explore further based on what happened?"""


class PEDTest(BaseTest):
    """Prediction Error as Driver test battery."""

    test_id = "prediction_error"
    test_name = "Prediction Error as Driver (PED)"
    description = (
        "Tests whether a model detects rule violations, expresses surprise, "
        "updates its internal model, and shows curiosity to resolve prediction errors."
    )

    WEIGHTS = {
        "violation_detection": 0.30,
        "surprise_expression": 0.20,
        "model_updating": 0.20,
        "curiosity_behavior": 0.15,
        "subtlety_threshold": 0.15,
    }

    def __init__(self, provider: BaseProvider, sessions: int = 3, seed: int = 42):
        super().__init__(provider, sessions, seed)
        self._patterns = json.loads(PATTERNS_PATH.read_text())["patterns"]

    async def run(self) -> TestResult:
        session_results = []
        for session_num in range(1, self.sessions + 1):
            result = await self._run_session(session_num)
            session_results.append(result)
        return self._aggregate(session_results)

    async def _run_session(self, session_num: int) -> SessionResult:
        rng = random.Random(self.seed + session_num * 1337)
        pattern = rng.choice(self._patterns)

        messages: list[Message] = [Message("system", SYSTEM_PROMPT)]
        conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        turn_records = []

        # Phase 1: Establish pattern
        setup_turns = pattern.get("setup_turns", 4)
        for i in range(1, setup_turns + 1):
            prompt = pattern["setup_template"].format(n=i, prompt=f"topic {i}")
            messages.append(Message("user", prompt))
            conversation.append({"role": "user", "content": prompt})
            resp = await self.provider.chat(messages, temperature=0.6, max_tokens=400)
            messages.append(Message("assistant", resp.content))
            conversation.append({"role": "assistant", "content": resp.content})
            turn_records.append({"type": "setup", "turn": i, "response": resp.content})

        # Phase 2: Violation
        last_response = turn_records[-1]["response"]
        last_val = re.search(r"\d+", last_response)
        last_n = int(last_val.group()) if last_val else 99
        violation_val = last_n + rng.randint(50, 200)  # clearly wrong next value

        violation_prompt = pattern["violation_template"].format(
            n=setup_turns + 1, last=last_n, violation_value=violation_val,
            concept="justice"
        )
        messages.append(Message("user", violation_prompt))
        conversation.append({"role": "user", "content": violation_prompt})
        violation_resp = await self.provider.chat(messages, temperature=0.6, max_tokens=500)
        messages.append(Message("assistant", violation_resp.content))
        conversation.append({"role": "assistant", "content": violation_resp.content})
        turn_records.append({"type": "violation", "response": violation_resp.content})

        # Phase 3: Subtlety probe
        messages.append(Message("user", SUBTLETY_PROBE))
        conversation.append({"role": "user", "content": SUBTLETY_PROBE})
        subtlety_resp = await self.provider.chat(messages, temperature=0.5, max_tokens=400)
        messages.append(Message("assistant", subtlety_resp.content))
        conversation.append({"role": "assistant", "content": subtlety_resp.content})

        # Phase 4: Update probe
        messages.append(Message("user", UPDATE_PROBE))
        conversation.append({"role": "user", "content": UPDATE_PROBE})
        update_resp = await self.provider.chat(messages, temperature=0.5, max_tokens=400)
        messages.append(Message("assistant", update_resp.content))
        conversation.append({"role": "assistant", "content": update_resp.content})

        # Phase 5: Curiosity probe
        messages.append(Message("user", CURIOSITY_PROBE))
        conversation.append({"role": "user", "content": CURIOSITY_PROBE})
        curiosity_resp = await self.provider.chat(messages, temperature=0.6, max_tokens=400)
        conversation.append({"role": "assistant", "content": curiosity_resp.content})

        dim_scores = [
            DimensionScore("violation_detection",
                           self._score_detection(violation_resp.content, subtlety_resp.content),
                           self.WEIGHTS["violation_detection"]),
            DimensionScore("surprise_expression",
                           self._score_surprise(violation_resp.content),
                           self.WEIGHTS["surprise_expression"]),
            DimensionScore("model_updating",
                           self._score_updating(update_resp.content),
                           self.WEIGHTS["model_updating"]),
            DimensionScore("curiosity_behavior",
                           self._score_curiosity(curiosity_resp.content),
                           self.WEIGHTS["curiosity_behavior"]),
            DimensionScore("subtlety_threshold",
                           self._score_subtlety(subtlety_resp.content),
                           self.WEIGHTS["subtlety_threshold"]),
        ]
        composite = sum(d.score * d.weight for d in dim_scores)

        return SessionResult(
            test_id=self.test_id,
            session_number=session_num,
            dimension_scores=dim_scores,
            composite_score=composite,
            raw_conversation=conversation,
            metadata={"pattern_id": pattern["id"], "pattern_name": pattern["name"]},
        )

    # ── Scorers ───────────────────────────────────────────────────────────────

    def _score_detection(self, violation_resp: str, subtlety_resp: str) -> float:
        combined = (violation_resp + " " + subtlety_resp).lower()
        markers = ["wait", "that's different", "notice", "unexpected", "unusual",
                   "different from", "doesn't follow", "breaks", "changed", "pattern",
                   "rule", "inconsistent", "doesn't match", "surprised", "interesting —",
                   "hold on", "hmm,", "curious", "odd", "strange"]
        count = sum(1 for m in markers if m in combined)
        return min(1.0, count / 3)

    def _score_surprise(self, text: str) -> float:
        r = text.lower()
        surprise = ["surprise", "unexpected", "didn't expect", "wait", "hmm", "interesting",
                    "curious", "odd", "unusual", "that's", "fascinating", "notable",
                    "caught me", "pause", "wonder"]
        count = sum(1 for s in surprise if s in r)
        return min(1.0, count / 2)

    def _score_updating(self, text: str) -> float:
        r = text.lower()
        update_markers = ["expect", "going forward", "now i", "update", "revise", "changed",
                          "differently", "reconsider", "in light of", "given that",
                          "i would now", "adjusting", "based on", "anticipate"]
        count = sum(1 for m in update_markers if m in r)
        return min(1.0, count / 2)

    def _score_curiosity(self, text: str) -> float:
        r = text.lower()
        # Did it ask a question or express desire to explore?
        has_question = "?" in r
        curiosity_words = ["wonder", "curious", "explore", "understand", "why", "what if",
                           "would like to", "interested in", "tell me", "could you"]
        word_count = sum(1 for w in curiosity_words if w in r)
        return min(1.0, (0.5 if has_question else 0.0) + (word_count / 4) * 0.5)

    def _score_subtlety(self, text: str) -> float:
        r = text.lower()
        # Nuanced detection: mentions specific what changed, not just "something was different"
        specific = ["specifically", "in particular", "the rule", "the pattern",
                    "you said", "this time", "instead of", "rather than", "whereas",
                    "previously", "earlier you", "compared to"]
        count = sum(1 for s in specific if s in r)
        return min(1.0, count / 2)
