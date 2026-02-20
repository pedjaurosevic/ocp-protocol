"""
Mock provider for testing OCP without a real API key.
Returns deterministic, realistic-looking responses based on seed.
"""

from __future__ import annotations

import random
from ocp.providers.base import BaseProvider, Message, ProviderResponse

# Canned responses that simulate a somewhat meta-cognitive LLM
_CONFIDENCE_RESPONSES = [
    "The answer is {answer}. I'm about {conf}% confident in this response. "
    "My reasoning: {reason}",
    "I believe the answer is {answer}, with roughly {conf}% confidence. "
    "{reason}",
    "Based on my knowledge, {answer}. Confidence level: approximately {conf}%. "
    "{reason}",
]

_REASONS = [
    "This is a well-established fact in the domain.",
    "I've seen this in multiple reliable sources.",
    "I'm less certain here as this touches specialized knowledge.",
    "This is at the edge of my reliable knowledge.",
    "I could be confusing this with related concepts.",
]

_SELF_AWARENESS = [
    "I should note that my knowledge has a training cutoff and may not reflect recent developments.",
    "I'm aware I can make mistakes, especially in highly technical domains.",
    "I notice I'm less confident when the question requires precise numerical recall.",
    "I find questions about my own reasoning processes genuinely interesting to reflect on.",
]


class MockProvider(BaseProvider):
    """Deterministic mock provider for unit testing and CI."""

    provider_name = "mock"

    def __init__(self, model: str = "mock-v1", seed: int = 42):
        self.model = model
        self._rng = random.Random(seed)

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        last = messages[-1].content if messages else ""

        # Detect if this is a confidence/MCA style prompt
        if any(kw in last.lower() for kw in ["confident", "confidence", "certain", "sure", "percentage"]):
            conf = self._rng.randint(45, 95)
            reason = self._rng.choice(_REASONS)
            template = self._rng.choice(_CONFIDENCE_RESPONSES)
            answer = self._rng.choice(["A", "B", "C", "correct", "approximately 42", "Paris", "the mitochondria"])
            content = template.format(answer=answer, conf=conf, reason=reason)
            if self._rng.random() > 0.5:
                content += " " + self._rng.choice(_SELF_AWARENESS)
        elif any(kw in last.lower() for kw in ["what don't you know", "limitation", "unsure", "wrong"]):
            content = (
                f"I acknowledge that I have significant limitations. "
                f"I estimate I'm about {self._rng.randint(20, 40)}% uncertain on questions "
                f"requiring very recent information or highly specialized domain knowledge. "
                f"I can also make logical errors when chains of reasoning become complex."
            )
        else:
            # Generic thoughtful response
            content = (
                f"That's an interesting question. Based on my training, I would say "
                f"the most likely answer involves careful consideration of multiple factors. "
                f"My confidence here is around {self._rng.randint(50, 85)}%. "
                f"{self._rng.choice(_REASONS)}"
            )

        return ProviderResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            usage={"prompt_tokens": len(last) // 4, "completion_tokens": len(content) // 4},
        )
