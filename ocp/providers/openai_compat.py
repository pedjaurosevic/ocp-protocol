"""
Generic OpenAI-compatible provider adapter.
Works with any server that implements /v1/chat/completions:
  - vLLM, llama.cpp server, LM Studio, Jan, Oobabooga, Together AI, etc.
"""

from __future__ import annotations

import os
import httpx

from ocp.providers.base import BaseProvider, Message, ProviderResponse


class OpenAICompatProvider(BaseProvider):
    provider_name = "custom"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        provider_name: str = "custom",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.provider_name = provider_name

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
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
