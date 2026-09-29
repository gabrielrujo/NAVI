from __future__ import annotations

from navi.bootstrap import ApplicationContainer, RuntimeStatus
from navi.domain.models import AssistantResponse


class DesktopController:
    """Adapta ações da interface sem acoplar Qt ao núcleo da aplicação."""

    def __init__(
        self,
        container: ApplicationContainer,
        *,
        session_id: str = "desktop:local",
    ) -> None:
        self._container = container
        self._session_id = session_id

    @property
    def status(self) -> RuntimeStatus:
        return self._container.status

    async def answer(self, question: str) -> AssistantResponse:
        return await self._container.assistant.answer(
            session_id=self._session_id,
            question=question,
        )

    async def new_conversation(self) -> None:
        await self._container.assistant.clear_history(self._session_id)
