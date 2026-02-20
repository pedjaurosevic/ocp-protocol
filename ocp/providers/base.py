"""
Base classes for OCP providers.
A provider wraps a model API and exposes a uniform chat interface.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ProviderResponse:
    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class BaseProvider(abc.ABC):
    """Abstract base for all OCP model providers."""

    provider_name: str = "base"

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        """Send a chat completion request and return a ProviderResponse."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={getattr(self, 'model', '?')})"
