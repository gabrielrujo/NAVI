from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from navi.api.app import create_app
from navi.domain.models import KnowledgeChunk
from navi.infrastructure.memory import InMemoryConversationStore
from navi.services.assistant import AssistantService

from .test_assistant_service import FakeKnowledgeBase, FakeProvider


@pytest.mark.asyncio
async def test_fastapi_is_an_independent_channel_for_assistant() -> None:
    provider = FakeProvider()
    knowledge = FakeKnowledgeBase([KnowledgeChunk("Definição", "Tributos.pdf", page=3)])
    assistant = AssistantService(
        llm=provider,
        knowledge_base=knowledge,
        conversations=InMemoryConversationStore(),
        human_contact="Fale com o NAF.",
    )

    async def close() -> None:
        return None

    container = SimpleNamespace(
        assistant=assistant,
        knowledge_base=knowledge,
        llm=provider,
        aclose=close,
    )
    transport = httpx.ASGITransport(app=create_app(container))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        response = await client.post(
            "/v1/chat",
            json={"session_id": "totem-1", "message": "O que é tributo?"},
        )
        cleared = await client.delete("/v1/chat/totem-1")

    assert health.json()["rag_ready"] is True
    assert response.status_code == 200
    assert response.json()["sources"] == [{"name": "Tributos.pdf", "page": 3}]
    assert cleared.status_code == 204
