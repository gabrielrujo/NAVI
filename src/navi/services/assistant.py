from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence

from navi.domain.models import (
    AssistantResponse,
    ChatMessage,
    InvalidQuestionError,
    KnowledgeChunk,
    MessageRole,
    SourceReference,
)
from navi.ports.conversations import ConversationStore
from navi.ports.knowledge import KnowledgeBase
from navi.ports.llm import LLMProvider
from navi.services.prompts import build_system_prompt


class AssistantService:
    """Nucleo de atendimento reutilizavel por Telegram, API, totem e outros canais."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        knowledge_base: KnowledgeBase,
        conversations: ConversationStore,
        human_contact: str,
        max_question_chars: int = 3000,
    ) -> None:
        self._llm = llm
        self._knowledge_base = knowledge_base
        self._conversations = conversations
        self._human_contact = human_contact
        self._max_question_chars = max_question_chars
        self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def answer(self, *, session_id: str, question: str) -> AssistantResponse:
        clean_question = question.strip()
        if not clean_question:
            raise InvalidQuestionError("Envie uma pergunta em texto.")
        if len(clean_question) > self._max_question_chars:
            raise InvalidQuestionError(
                f"A pergunta deve ter no maximo {self._max_question_chars} caracteres."
            )

        async with self._session_locks[session_id]:
            chunks = await self._knowledge_base.retrieve(clean_question)
            if not chunks:
                return AssistantResponse(
                    answer=(
                        "Não encontrei essa informação nos materiais disponíveis do NAF. "
                        f"{self._human_contact}"
                    ),
                    provider=self._llm.provider_name,
                    model=self._llm.model_name,
                )

            history = await self._conversations.get(session_id)
            user_message = ChatMessage(MessageRole.USER, clean_question)
            answer = await self._llm.generate(
                system_prompt=build_system_prompt(
                    chunks=chunks,
                    human_contact=self._human_contact,
                ),
                messages=(*history, user_message),
            )
            assistant_message = ChatMessage(MessageRole.ASSISTANT, answer)
            await self._conversations.append(session_id, user_message, assistant_message)

            return AssistantResponse(
                answer=answer,
                sources=self._deduplicate_sources(chunks),
                provider=self._llm.provider_name,
                model=self._llm.model_name,
            )

    async def clear_history(self, session_id: str) -> None:
        await self._conversations.clear(session_id)

    @staticmethod
    def _deduplicate_sources(chunks: Sequence[KnowledgeChunk]) -> tuple[SourceReference, ...]:
        seen: set[tuple[str, int | None]] = set()
        result: list[SourceReference] = []
        for chunk in chunks:
            key = (chunk.source_name, chunk.page)
            if key not in seen:
                seen.add(key)
                result.append(SourceReference(name=chunk.source_name, page=chunk.page))
        return tuple(result)
