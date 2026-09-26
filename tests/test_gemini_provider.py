from __future__ import annotations

import json

import httpx
import pytest

from navi.config import Settings
from navi.domain.models import ChatMessage, MessageRole, ProviderError
from navi.providers.gemini import GeminiProvider


def test_default_model_matches_provider_and_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    settings = Settings(_env_file=None)
    provider = GeminiProvider(api_key="test-key")

    assert settings.gemini_model == provider.model_name == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_gemini_provider_maps_conversation_and_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "Resposta"}]}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-key", client=client)
    result = await provider.generate(
        system_prompt="Sistema",
        messages=[
            ChatMessage(MessageRole.USER, "Olá"),
            ChatMessage(MessageRole.ASSISTANT, "Oi"),
            ChatMessage(MessageRole.USER, "Dúvida"),
        ],
    )

    assert result == "Resposta"
    assert captured["system_instruction"]["parts"][0]["text"] == "Sistema"
    assert [item["role"] for item in captured["contents"]] == ["user", "model", "user"]
    await client.aclose()


@pytest.mark.asyncio
async def test_gemini_provider_hides_error_payload_details() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "chave inválida"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="secret", client=client)
    with pytest.raises(ProviderError, match="chave inválida"):
        await provider.generate(
            system_prompt="Sistema",
            messages=[ChatMessage(MessageRole.USER, "Pergunta")],
        )
    await client.aclose()
