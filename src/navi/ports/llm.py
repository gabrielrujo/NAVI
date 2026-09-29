from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from navi.domain.models import ChatMessage


class LLMProvider(Protocol):
    """Contrato a ser implementado pelo Gemini e pelo futuro modelo local."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def network_required(self) -> bool: ...

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: Sequence[ChatMessage],
    ) -> str: ...


class ClosableLLMProvider(LLMProvider, Protocol):
    async def aclose(self) -> None: ...
