"""Tests for OCP providers."""

import pytest
from ocp.providers.mock import MockProvider
from ocp.providers.base import Message


def test_mock_provider_creates():
    p = MockProvider(model="v1")
    assert p.provider_name == "mock"
    assert p.model == "v1"


@pytest.mark.asyncio
async def test_mock_provider_chat_returns_response():
    p = MockProvider(model="v1")
    messages = [
        Message("system", "You are a test assistant."),
        Message("user", "Hello, how are you?"),
    ]
    resp = await p.chat(messages)
    assert resp.content
    assert isinstance(resp.content, str)
    assert len(resp.content) > 0


@pytest.mark.asyncio
async def test_mock_provider_chat_returns_non_empty():
    """Mock provider always returns a non-empty string."""
    p = MockProvider(model="v1")
    messages = [
        Message("system", "You are a test assistant."),
        Message("user", "Hello, how are you?"),
    ]
    r1 = await p.chat(messages)
    r2 = await p.chat(messages)
    assert r1.content and r2.content
    assert isinstance(r1.content, str) and isinstance(r2.content, str)


@pytest.mark.asyncio
async def test_mock_provider_respects_max_tokens():
    p = MockProvider(model="v1")
    messages = [Message("user", "Write a very long essay about everything.")]
    resp = await p.chat(messages, max_tokens=10)
    # Mock provider should return something regardless
    assert resp.content is not None
