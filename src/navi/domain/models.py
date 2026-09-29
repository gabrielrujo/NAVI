from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    text: str
    source_name: str
    page: int | None = None
    score: float | None = None

    @property
    def label(self) -> str:
        return f"{self.source_name}, p. {self.page}" if self.page else self.source_name


@dataclass(frozen=True, slots=True)
class SourceReference:
    name: str
    page: int | None = None

    @property
    def label(self) -> str:
        return f"{self.name}, p. {self.page}" if self.page else self.name


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    answer: str
    sources: tuple[SourceReference, ...] = field(default_factory=tuple)
    provider: str = ""
    model: str = ""


class NaviError(Exception):
    """Erro esperado na aplicacao."""


class InvalidQuestionError(NaviError):
    """Pergunta vazia ou maior que o limite aceito."""


class KnowledgeBaseNotReadyError(NaviError):
    """O indice RAG ainda nao foi criado."""


class ProviderError(NaviError):
    """Falha ao obter uma resposta do provider de IA."""


class EmbeddingError(NaviError):
    """Falha ao gerar embeddings ou consultar um indice que depende deles."""


class ConfigurationError(NaviError):
    """Configuracao obrigatoria ausente ou invalida."""
