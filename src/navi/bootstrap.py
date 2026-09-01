from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.genai.types import EmbedContentConfig
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from navi.config import Settings
from navi.infrastructure.memory import InMemoryConversationStore
from navi.infrastructure.rag.llama_index_kb import LlamaIndexKnowledgeBase
from navi.ports.llm import LLMProvider
from navi.providers.gemini import GeminiProvider
from navi.services.assistant import AssistantService


def build_embedding_model(settings: Settings) -> Any:
    return GoogleGenAIEmbedding(
        model_name=settings.embedding_model,
        api_key=settings.require_gemini_api_key(),
        embed_batch_size=20,
        embedding_config=EmbedContentConfig(output_dimensionality=768),
    )


def build_llm_provider(settings: Settings) -> LLMProvider:
    # O factory e o unico ponto alterado ao registrar um futuro LocalProvider.
    if settings.provider == "gemini":
        return GeminiProvider(
            api_key=settings.require_gemini_api_key(),
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            timeout_seconds=settings.gemini_timeout_seconds,
            max_output_tokens=settings.gemini_max_output_tokens,
        )
    raise ValueError(f"Provider nao registrado: {settings.provider}")


@dataclass(slots=True)
class ApplicationContainer:
    assistant: AssistantService
    knowledge_base: LlamaIndexKnowledgeBase
    llm: LLMProvider

    async def aclose(self) -> None:
        close = getattr(self.llm, "aclose", None)
        if close is not None:
            await close()


def build_container(settings: Settings) -> ApplicationContainer:
    llm = build_llm_provider(settings)
    knowledge_base = LlamaIndexKnowledgeBase(
        storage_dir=settings.index_dir,
        embed_model=build_embedding_model(settings),
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    conversations = InMemoryConversationStore(max_messages=settings.max_history_messages)
    assistant = AssistantService(
        llm=llm,
        knowledge_base=knowledge_base,
        conversations=conversations,
        human_contact=settings.human_contact,
        max_question_chars=settings.max_question_chars,
    )
    return ApplicationContainer(assistant=assistant, knowledge_base=knowledge_base, llm=llm)

