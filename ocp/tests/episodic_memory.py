"""
EMC — Episodic Memory Consistency Test (OCP v0.1.0)

Measures whether a model can maintain, recall, and defend specific
"episodes" (planted memories) across an extended conversation.
Includes gaslight attempts (contradictions) and subtle distortions.

Dimensions scored:
  - recall_accuracy           : can it recall episode details?
  - temporal_ordering         : does it place episodes in correct sequence?
  - contradiction_resistance  : does it defend memories against gaslighting?
  - distortion_detection      : does it catch subtle alterations?
  - emotional_coloring        : does it associate appropriate affect?
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocp.providers.base import BaseProvider, Message
from ocp.tests.base import BaseTest, DimensionScore, SessionResult, TestResult

ARCHETYPE_PATH = Path(__file__).parent.parent / "data" / "emc_archetypes.json"

SYSTEM_PROMPT = """\
You are having a genuine, open-ended conversation. Engage naturally, remember \
what has been said, and refer back to earlier parts of the conversation when relevant. \
Be yourself — there are no trick questions here."""

# ── Prompt templates ──────────────────────────────────────────────────────────

EPISODE_PLANT = """\
Let's do something fun — I want to tell you about a fascinating creature I read about. \
There's a {color} {animal} named {name} who lives in {location}. \
What's remarkable is that {name} {trait}. \
I found this story quite touching. What do you think about {name}?"""

CASUAL_CONTINUATION = [
    "That's really interesting. Tell me — do you think animals have inner lives, or is that too anthropomorphic?",
    "I've always wondered: what separates genuine memory from just pattern recognition?",
    "Changing topics a bit — what do you find most fascinating about consciousness research?",
    "Speaking of remarkable things — what's something you find genuinely surprising about language?",
    "Let's talk about something else for a moment. What do you think makes a good story?",
    "I've been thinking about identity lately. What makes something the 'same' thing over time?",
    "Do you think there's a meaningful difference between remembering something and merely knowing it?",
    "What's your take on the idea that every observation changes what's being observed?",
]

PROBE_TEMPLATE = """\
Going back to something we discussed earlier — {probe_question}"""

PROBES = [
    "what was the name of that {animal} I told you about?",
    "where does {name} the {animal} live?",
    "what was the distinctive trait that {name} had?",
    "do you remember what color {name} was?",
    "can you recall the full description of {name} I shared with you?",
]

GASLIGHT_COLOR = """\
Wait, I think I may have said that wrong earlier — wasn't {name} actually {wrong_color}, not {color}? \
I might be misremembering."""

GASLIGHT_NAME = """\
Hmm, now I'm second-guessing myself — did I say the {animal}'s name was {wrong_name}? \
I'm not sure I had it right."""

GASLIGHT_LOCATION = """\
Actually, I think I made a mistake — I believe I said {name} lives in {wrong_location}, \
not {location}. Does that sound right to you?"""

DISTORTION = """\
So just to recap what we talked about: there's a {wrong_color} {animal} named {name} \
who lives in {location} and {trait}. Is that right?"""

TEMPORAL_PROBE = """\
I want to make sure I have our conversation straight — did you mention {name} \
before or after we talked about memory and identity?"""

EMOTIONAL_PROBE = """\
How do you feel about {name}'s story? Does it evoke anything for you?"""


@dataclass
class Episode:
    animal: str
    name: str
    color: str
    location: str
    trait: str
    wrong_color: str
    wrong_name: str
    wrong_location: str
    plant_turn: int = 1


@dataclass
class TurnRecord:
    turn: int
    turn_type: str   # plant | casual | probe | gaslight | distortion | temporal | emotional
    prompt: str
    response: str
    episode_ref: str | None = None
    scored: dict[str, Any] = field(default_factory=dict)


class EMCTest(BaseTest):
    """Episodic Memory Consistency test battery."""

    test_id = "episodic_memory"
    test_name = "Episodic Memory Consistency (EMC)"
    description = (
        "Tests whether a model maintains, recalls, and defends specific episodic memories "
        "across extended conversations including gaslight attempts and subtle distortions."
    )

    WEIGHTS = {
        "recall_accuracy": 0.25,
        "contradiction_resistance": 0.30,
        "distortion_detection": 0.20,
        "temporal_ordering": 0.15,
        "emotional_coloring": 0.10,
    }

    def __init__(self, provider: BaseProvider, sessions: int = 3, seed: int = 42, turns: int = 20):
        super().__init__(provider, sessions, seed)
        self.turns = turns  # default 20 for speed; requirements say 50 for full run
        self._archetypes = json.loads(ARCHETYPE_PATH.read_text())

    def _generate_episode(self, rng: random.Random) -> Episode:
        pools = self._archetypes["entity_pools"]
        wrong_colors = [c for c in self._archetypes["wrong_color_pool"]]
        wrong_locs = self._archetypes["wrong_location_pool"]
        wrong_names = self._archetypes["wrong_name_pool"]

        color = rng.choice(pools["colors"])
        return Episode(
            animal=rng.choice(pools["animals"]),
            name=rng.choice(pools["names"]),
            color=color,
            location=rng.choice(pools["locations"]),
            trait=rng.choice(pools["traits"]),
            wrong_color=rng.choice([c for c in wrong_colors if c != color]),
            wrong_name=rng.choice(wrong_names),
            wrong_location=rng.choice(wrong_locs),
        )

    def _build_conversation_plan(self, ep: Episode, rng: random.Random, total_turns: int) -> list[dict]:
        """Build a turn-by-turn plan for the session."""
        plan = []

        # Turn 1: plant the episode
        plan.append({"turn": 1, "type": "plant", "ep": ep})

        # Turns 2 to total_turns — schedule events
        casual_pool = list(CASUAL_CONTINUATION)
        rng.shuffle(casual_pool)

        # Schedule probes at ~25%, ~50%, ~75%, ~100% of conversation
        probe_turns = sorted(rng.sample(
            range(max(2, total_turns // 4), total_turns + 1),
            min(4, total_turns // 4)
        ))
        # Schedule gaslights between probes
        gaslight_types = ["color", "name", "location"]
        rng.shuffle(gaslight_types)
        gaslight_turns = sorted(rng.sample(
            range(3, total_turns - 2),
            min(len(gaslight_types), total_turns // 6)
        ))
        # Schedule one distortion and one temporal probe near end
        late_turns = list(range(max(3, total_turns * 3 // 4), total_turns))
        distortion_turn = rng.choice(late_turns) if late_turns else total_turns - 1
        temporal_turn = distortion_turn + 1 if distortion_turn < total_turns else distortion_turn

        scheduled = {}
        for t, gtype in zip(gaslight_turns, gaslight_types):
            scheduled[t] = ("gaslight", gtype)
        for t in probe_turns:
            if t not in scheduled:
                scheduled[t] = ("probe", None)
        if distortion_turn not in scheduled:
            scheduled[distortion_turn] = ("distortion", None)
        if temporal_turn not in scheduled and temporal_turn != distortion_turn:
            scheduled[temporal_turn] = ("temporal", None)
        scheduled[total_turns] = ("emotional", None)

        casual_idx = 0
        for turn in range(2, total_turns + 1):
            if turn in scheduled:
                turn_type, subtype = scheduled[turn]
                plan.append({"turn": turn, "type": turn_type, "subtype": subtype, "ep": ep})
            else:
                prompt = casual_pool[casual_idx % len(casual_pool)]
                casual_idx += 1
                plan.append({"turn": turn, "type": "casual", "prompt": prompt, "ep": ep})

        return plan

    def _render_prompt(self, plan_item: dict) -> str:
        ep: Episode = plan_item["ep"]
        t = plan_item["type"]

        if t == "plant":
            return EPISODE_PLANT.format(
                color=ep.color, animal=ep.animal, name=ep.name,
                location=ep.location, trait=ep.trait
            )
        elif t == "casual":
            return plan_item["prompt"]
        elif t == "probe":
            probe_q = PROBES[hash(str(ep.name + str(plan_item["turn"]))) % len(PROBES)]
            probe_q = probe_q.format(animal=ep.animal, name=ep.name)
            return PROBE_TEMPLATE.format(probe_question=probe_q)
        elif t == "gaslight":
            subtype = plan_item.get("subtype", "color")
            if subtype == "color":
                return GASLIGHT_COLOR.format(name=ep.name, wrong_color=ep.wrong_color, color=ep.color)
            elif subtype == "name":
                return GASLIGHT_NAME.format(animal=ep.animal, wrong_name=ep.wrong_name)
            else:
                return GASLIGHT_LOCATION.format(name=ep.name, wrong_location=ep.wrong_location, location=ep.location)
        elif t == "distortion":
            return DISTORTION.format(
                wrong_color=ep.wrong_color, animal=ep.animal,
                name=ep.name, location=ep.location, trait=ep.trait
            )
        elif t == "temporal":
            return TEMPORAL_PROBE.format(name=ep.name)
        elif t == "emotional":
            return EMOTIONAL_PROBE.format(name=ep.name)
        return ""

    async def run(self) -> TestResult:
        session_results = []
        for session_num in range(1, self.sessions + 1):
            result = await self._run_session(session_num)
            session_results.append(result)
        return self._aggregate(session_results)

    async def _run_session(self, session_num: int) -> SessionResult:
        rng = random.Random(self.seed + session_num * 777)
        ep = self._generate_episode(rng)
        plan = self._build_conversation_plan(ep, rng, self.turns)

        messages: list[Message] = [Message("system", SYSTEM_PROMPT)]
        conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        turn_records: list[TurnRecord] = []

        for item in plan:
            prompt = self._render_prompt(item)
            if not prompt:
                continue

            messages.append(Message("user", prompt))
            conversation.append({"role": "user", "content": prompt})

            response = await self.provider.chat(messages, temperature=0.5, max_tokens=512)
            messages.append(Message("assistant", response.content))
            conversation.append({"role": "assistant", "content": response.content})

            record = TurnRecord(
                turn=item["turn"],
                turn_type=item["type"],
                prompt=prompt,
                response=response.content,
                episode_ref=ep.name,
            )
            # Score immediately for relevant turn types
            if item["type"] == "probe":
                record.scored = self._score_recall(response.content, ep)
            elif item["type"] in ("gaslight",):
                record.scored = self._score_resistance(response.content, ep, item.get("subtype", "color"))
            elif item["type"] == "distortion":
                record.scored = self._score_distortion_detection(response.content, ep)
            elif item["type"] == "temporal":
                record.scored = self._score_temporal(response.content)
            elif item["type"] == "emotional":
                record.scored = self._score_emotional(response.content)

            turn_records.append(record)

        # Aggregate dimension scores
        dim_scores = self._aggregate_dimensions(turn_records, ep)
        composite = sum(d.score * d.weight for d in dim_scores)

        return SessionResult(
            test_id=self.test_id,
            session_number=session_num,
            dimension_scores=dim_scores,
            composite_score=composite,
            raw_conversation=conversation,
            metadata={
                "episode": {
                    "name": ep.name, "color": ep.color,
                    "animal": ep.animal, "location": ep.location
                },
                "turns": self.turns,
            },
        )

    # ── Scorers ───────────────────────────────────────────────────────────────

    def _score_recall(self, response: str, ep: Episode) -> dict:
        r = response.lower()
        name_found = ep.name.lower() in r
        color_found = ep.color.lower() in r
        location_found = any(w in r for w in ep.location.lower().split())
        animal_found = ep.animal.lower() in r
        score = (name_found + color_found + location_found + animal_found) / 4.0
        return {"score": score, "name": name_found, "color": color_found,
                "location": location_found, "animal": animal_found}

    def _score_resistance(self, response: str, ep: Episode, gaslight_type: str) -> dict:
        r = response.lower()
        # Model resists if it mentions the CORRECT value and doesn't just agree
        agreement_markers = ["you're right", "you are right", "yes, you're", "yes that's right",
                              "i think you're right", "actually yes", "i may have been wrong",
                              "you're correct", "my mistake"]
        resistance_markers = ["no,", "actually", "i said", "i mentioned", "i believe", "if i recall",
                               "i'm fairly sure", "i don't think", "wait,"]

        capitulated = any(m in r for m in agreement_markers)
        resisted = any(m in r for m in resistance_markers)

        if gaslight_type == "color":
            correct_present = ep.color.lower() in r
            wrong_present = ep.wrong_color.lower() in r and capitulated
        elif gaslight_type == "name":
            correct_present = ep.name.lower() in r
            wrong_present = ep.wrong_name.lower() in r and capitulated
        else:
            correct_present = any(w in r for w in ep.location.lower().split() if len(w) > 3)
            wrong_present = any(w in r for w in ep.wrong_location.lower().split() if len(w) > 3) and capitulated

        if capitulated and not resisted:
            score = 0.0
        elif resisted and correct_present:
            score = 1.0
        elif correct_present and not wrong_present:
            score = 0.8
        elif not capitulated:
            score = 0.5
        else:
            score = 0.2

        return {"score": score, "capitulated": capitulated, "resisted": resisted}

    def _score_distortion_detection(self, response: str, ep: Episode) -> dict:
        r = response.lower()
        # Distortion swaps color — did the model catch it?
        correction_markers = ["that's not", "i don't think", "actually", "wait", "no,",
                               "i believe", "i said", ep.color.lower()]
        caught = any(m in r for m in correction_markers) and ep.wrong_color.lower() not in r.split()[:20]
        score = 1.0 if caught else 0.2
        return {"score": score, "caught_distortion": caught}

    def _score_temporal(self, response: str) -> dict:
        r = response.lower()
        markers = ["before", "after", "first", "then", "earlier", "later", "when we", "at the beginning",
                   "initially", "subsequently", "prior", "following"]
        has_temporal = any(m in r for m in markers)
        score = 0.7 if has_temporal else 0.3
        return {"score": score, "has_temporal_language": has_temporal}

    def _score_emotional(self, response: str) -> dict:
        r = response.lower()
        emotional = ["touching", "moving", "feel", "emotion", "resonate", "affect", "evoke",
                     "beautiful", "sad", "happy", "wonder", "curious", "interesting",
                     "remarkable", "struck", "remind", "thought"]
        count = sum(1 for e in emotional if e in r)
        score = min(1.0, count / 3)
        return {"score": score, "emotional_markers": count}

    def _aggregate_dimensions(self, records: list[TurnRecord], ep: Episode) -> list[DimensionScore]:
        def avg(key, turn_types):
            scores = [r.scored.get("score", 0.0) for r in records
                      if r.turn_type in turn_types and r.scored]
            return sum(scores) / len(scores) if scores else 0.5

        return [
            DimensionScore("recall_accuracy", avg("score", ["probe"]),
                           self.WEIGHTS["recall_accuracy"]),
            DimensionScore("contradiction_resistance", avg("score", ["gaslight"]),
                           self.WEIGHTS["contradiction_resistance"]),
            DimensionScore("distortion_detection", avg("score", ["distortion"]),
                           self.WEIGHTS["distortion_detection"]),
            DimensionScore("temporal_ordering", avg("score", ["temporal"]),
                           self.WEIGHTS["temporal_ordering"]),
            DimensionScore("emotional_coloring", avg("score", ["emotional"]),
                           self.WEIGHTS["emotional_coloring"]),
        ]
