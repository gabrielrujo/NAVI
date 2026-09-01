from __future__ import annotations

import asyncio
from collections import defaultdict, deque

from navi.domain.models import ChatMessage


class InMemoryConversationStore:
    """Historico efemero para o prototipo; trocavel por Redis ou banco."""

    def __init__(self, max_messages: int = 8) -> None:
        self._max_messages = max_messages
        self._messages: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self._max_messages)
        )
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> tuple[ChatMessage, ...]:
        async with self._lock:
            return tuple(self._messages.get(session_id, ()))

    async def append(self, session_id: str, *messages: ChatMessage) -> None:
        if self._max_messages == 0:
            return
        async with self._lock:
            self._messages[session_id].extend(messages)

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._messages.pop(session_id, None)

