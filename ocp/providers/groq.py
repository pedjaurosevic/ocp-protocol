"""
Groq provider adapter.
Uses httpx async client against Groq's OpenAI-compatible API.
"""

from __future__ import annotations

import os
import httpx

from ocp.providers.base import BaseProvider, Message, ProviderResponse

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(BaseProvider):
    provider_name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Export it or pass api_key=.")

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        return ProviderResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            provider=self.provider_name,
            usage=data.get("usage", {}),
            raw=data,
        )
