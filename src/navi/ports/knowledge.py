from __future__ import annotations

from typing import Protocol

from navi.domain.models import KnowledgeChunk


class KnowledgeBase(Protocol):
    """Recupera evidencia; nao gera nem sintetiza respostas."""

    @property
    def ready(self) -> bool: ...

    async def retrieve(self, query: str, *, limit: int | None = None) -> list[KnowledgeChunk]: ...

