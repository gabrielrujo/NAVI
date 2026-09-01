from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from navi.domain.models import ChatMessage


class ConversationStore(Protocol):
    async def get(self, session_id: str) -> Sequence[ChatMessage]: ...

    async def append(self, session_id: str, *messages: ChatMessage) -> None: ...

    async def clear(self, session_id: str) -> None: ...
