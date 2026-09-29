from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from navi.config import Settings
from navi.infrastructure.memory import InMemoryConversationStore
from navi.infrastructure.rag.llama_index_kb import LlamaIndexKnowledgeBase
from navi.ports.embeddings import EmbeddingProvider
from navi.ports.llm import LLMProvider
from navi.providers.embeddings import GeminiEmbeddingProvider, LocalEmbeddingProvider
from navi.providers.gemini import GeminiProvider
from navi.providers.local import LocalLLMProvider
from navi.services.assistant import AssistantService


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    provider: str
    model: str
    embeddings_provider: str
    embeddings_model: str
    network_required: bool
    index_ready: bool
    model_ready: bool
    index_dir: Path
    data_root: Path


def build_gemini_embedding_provider(settings: Settings) -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider(
        api_key=settings.require_gemini_api_key(),
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def build_local_embedding_provider(settings: Settings) -> LocalEmbeddingProvider:
    return LocalEmbeddingProvider(
        model=settings.local_embedding_model,
        cache_dir=settings.resolved_local_embedding_cache_dir,
        batch_size=settings.local_embedding_batch_size,
        threads=settings.local_embedding_threads,
        local_files_only=settings.local_embedding_local_files_only,
    )


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.resolved_embedding_provider == "gemini":
        return build_gemini_embedding_provider(settings)
    return build_local_embedding_provider(settings)


def build_embedding_model(settings: Settings, *, retries: int = 3) -> Any:
    return build_embedding_provider(settings).build_model(retries=retries)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "gemini":
        return GeminiProvider(
            api_key=settings.require_gemini_api_key(),
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_output_tokens=settings.gemini_max_output_tokens,
        )
    return LocalLLMProvider(
        model_path=settings.resolved_local_model_path,
        context_size=settings.local_context_size,
        max_output_tokens=settings.local_max_output_tokens,
        threads=settings.local_threads,
        chat_format=settings.local_chat_format,
    )


@dataclass(slots=True)
class ApplicationContainer:
    assistant: AssistantService
    knowledge_base: LlamaIndexKnowledgeBase
    llm: LLMProvider
    embeddings: EmbeddingProvider
    data_root: Path

    @property
    def status(self) -> RuntimeStatus:
        model_ready = bool(getattr(self.llm, "model_available", True))
        return RuntimeStatus(
            provider=self.llm.provider_name,
            model=self.llm.model_name,
            embeddings_provider=self.embeddings.provider_name,
            embeddings_model=self.embeddings.model_name,
            network_required=self.llm.network_required or self.embeddings.network_required,
            index_ready=self.knowledge_base.ready,
            model_ready=model_ready,
            index_dir=self.knowledge_base.storage_dir,
            data_root=self.data_root,
        )

    async def aclose(self) -> None:
        close = getattr(self.llm, "aclose", None)
        if close is not None:
            await close()


def build_container(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> ApplicationContainer:
    settings.ensure_local_mode_is_offline()
    selected_llm = llm or build_llm_provider(settings)
    selected_embeddings = embeddings or build_embedding_provider(settings)
    index_dir = (
        settings.resolved_local_index_dir
        if selected_embeddings.provider_name == "local"
        else settings.index_dir
    )
    knowledge_base = LlamaIndexKnowledgeBase(
        storage_dir=index_dir,
        embed_model=selected_embeddings.build_model(),
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
        expected_embedding_provider=selected_embeddings.provider_name,
        expected_embedding_model=selected_embeddings.model_name,
    )
    conversations = InMemoryConversationStore(max_messages=settings.max_history_messages)
    assistant = AssistantService(
        llm=selected_llm,
        knowledge_base=knowledge_base,
        conversations=conversations,
        human_contact=settings.human_contact,
        max_question_chars=settings.max_question_chars,
    )
    return ApplicationContainer(
        assistant=assistant,
        knowledge_base=knowledge_base,
        llm=selected_llm,
        embeddings=selected_embeddings,
        data_root=settings.data_root,
    )
