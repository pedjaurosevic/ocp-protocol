"""
Ollama provider adapter.
Talks to a locally running Ollama instance (default: http://localhost:11434).
Uses Ollama's OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

import os
import httpx

from ocp.providers.base import BaseProvider, Message, ProviderResponse

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def __init__(self, model: str = "llama3.2:3b", base_url: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")

    async def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        content = data.get("message", {}).get("content", "")
        return ProviderResponse(
            content=content,
            model=data.get("model", self.model),
            provider=self.provider_name,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )
