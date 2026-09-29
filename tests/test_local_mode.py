from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding

import navi.bootstrap as bootstrap
from navi.bootstrap import RuntimeStatus, build_container, build_llm_provider
from navi.channels.desktop.controller import DesktopController
from navi.config import Settings
from navi.domain.models import (
    AssistantResponse,
    ChatMessage,
    ConfigurationError,
    KnowledgeBaseNotReadyError,
    MessageRole,
    ProviderError,
)
from navi.infrastructure.memory import InMemoryConversationStore
from navi.infrastructure.rag.llama_index_kb import LlamaIndexKnowledgeBase
from navi.providers.embeddings import LocalEmbeddingProvider
from navi.providers.local import LocalLLMProvider
from navi.services.assistant import AssistantService


class FakeLocalBackend:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.messages = kwargs["messages"]
        return {"choices": [{"message": {"content": "Resposta totalmente local."}}]}


class FakeEmbeddingProvider:
    provider_name = "local"
    model_name = "fake-local-embeddings"
    network_required = False

    def build_model(self, *, retries: int = 3) -> MockEmbedding:
        del retries
        return MockEmbedding(embed_dim=8)


class FakeFastEmbedBackend:
    def embed(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        return [[float(len(text)), float(batch_size)] for text in texts]

    def query_embed(self, query: str) -> list[list[float]]:
        return [[float(len(query)), 1.0]]


class FakeLocalProvider:
    provider_name = "local"
    model_name = "fake.gguf"
    network_required = False
    model_available = True

    async def generate(self, *, system_prompt: str, messages: Sequence[ChatMessage]) -> str:
        del system_prompt, messages
        return "Resposta local simulada."


def test_data_root_resolves_portable_paths(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        provider="local",
        data_root=tmp_path,
        local_model_path=Path("models/teste.gguf"),
    )

    assert settings.resolved_local_model_path == tmp_path / "models/teste.gguf"
    assert settings.resolved_local_documents_dir == tmp_path / "documents"
    assert settings.resolved_local_index_dir == tmp_path / "index"
    assert settings.resolved_local_embedding_cache_dir == tmp_path / "embeddings"
    assert settings.resolved_embedding_provider == "local"


def test_local_provider_is_selected_without_gemini_key(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        provider="local",
        data_root=tmp_path,
        local_model_path=Path("models/teste.gguf"),
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, LocalLLMProvider)
    assert provider.network_required is False


def test_local_embeddings_use_cpu_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY", raising=False)
    captured_options: dict[str, Any] = {}

    def factory(**options: Any) -> FakeFastEmbedBackend:
        captured_options.update(options)
        return FakeFastEmbedBackend()

    provider = LocalEmbeddingProvider(
        model="modelo-multilingue",
        cache_dir=tmp_path / "embeddings",
        batch_size=4,
        local_files_only=True,
        backend_factory=factory,
    )
    model = provider.build_model()

    assert model.get_text_embedding_batch(["um", "dois"]) == [[2.0, 4.0], [4.0, 4.0]]
    assert model.get_query_embedding("pergunta") == [8.0, 1.0]
    assert captured_options["cuda"] is False
    assert captured_options["local_files_only"] is True
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert provider.network_required is False


def test_local_mode_rejects_gemini_embeddings() -> None:
    settings = Settings(
        _env_file=None,
        provider="local",
        embedding_provider="gemini",
    )

    with pytest.raises(ConfigurationError, match="exige embeddings locais"):
        settings.ensure_local_mode_is_offline()


@pytest.mark.asyncio
async def test_missing_local_model_has_friendly_error(tmp_path: Path) -> None:
    provider = LocalLLMProvider(model_path=tmp_path / "ausente.gguf")

    with pytest.raises(ProviderError, match="Modelo local não encontrado"):
        await provider.generate(
            system_prompt="Sistema",
            messages=[ChatMessage(MessageRole.USER, "Pergunta")],
        )


@pytest.mark.asyncio
async def test_local_provider_generates_with_injected_backend(tmp_path: Path) -> None:
    model_path = tmp_path / "modelo.gguf"
    model_path.write_bytes(b"fake")
    backend = FakeLocalBackend()
    captured_options: dict[str, Any] = {}

    def factory(**options: Any) -> FakeLocalBackend:
        captured_options.update(options)
        return backend

    provider = LocalLLMProvider(model_path=model_path, backend_factory=factory)
    answer = await provider.generate(
        system_prompt="Use apenas as fontes.",
        messages=[ChatMessage(MessageRole.USER, "Pergunta")],
    )

    assert answer == "Resposta totalmente local."
    assert captured_options["n_gpu_layers"] == 0
    assert backend.messages[0] == {"role": "system", "content": "Use apenas as fontes."}
    assert backend.messages[1] == {"role": "user", "content": "Pergunta"}


@pytest.mark.asyncio
async def test_missing_local_index_is_reported(tmp_path: Path) -> None:
    knowledge = LlamaIndexKnowledgeBase(
        storage_dir=tmp_path / "index",
        embed_model=MockEmbedding(embed_dim=8),
        expected_embedding_provider="local",
        expected_embedding_model="modelo-local",
    )

    assert knowledge.ready is False
    with pytest.raises(KnowledgeBaseNotReadyError, match="navi ingest-local"):
        await knowledge.retrieve("tributo")


@pytest.mark.asyncio
async def test_local_index_loads_and_retrieves_source_metadata(tmp_path: Path) -> None:
    storage_dir = tmp_path / "index"
    embed_model = MockEmbedding(embed_dim=8)
    index = VectorStoreIndex.from_documents(
        [
            Document(
                text="Imposto e taxa são espécies tributárias.",
                metadata={"source_name": "Tributos.pdf", "page_number": 7},
            )
        ],
        embed_model=embed_model,
    )
    index.storage_context.persist(persist_dir=str(storage_dir))
    knowledge = LlamaIndexKnowledgeBase(
        storage_dir=storage_dir,
        embed_model=embed_model,
        min_score=-1,
    )

    chunks = await knowledge.retrieve("Qual a diferença entre imposto e taxa?")

    assert knowledge.ready is True
    assert chunks[0].source_name == "Tributos.pdf"
    assert chunks[0].page == 7


@pytest.mark.asyncio
async def test_local_index_rejects_gemini_embedding_manifest(tmp_path: Path) -> None:
    storage_dir = tmp_path / "index"
    embed_model = MockEmbedding(embed_dim=8)
    index = VectorStoreIndex.from_documents(
        [Document(text="Texto", metadata={"source_name": "base.pdf", "page_number": 1})],
        embed_model=embed_model,
    )
    index.storage_context.persist(persist_dir=str(storage_dir))
    (storage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "embedding_provider": "gemini",
                "embedding_model": "gemini-embedding-2",
            }
        ),
        encoding="utf-8",
    )
    knowledge = LlamaIndexKnowledgeBase(
        storage_dir=storage_dir,
        embed_model=embed_model,
        expected_embedding_provider="local",
        expected_embedding_model="modelo-local",
    )

    assert knowledge.ready is False
    with pytest.raises(KnowledgeBaseNotReadyError, match="outro provider"):
        await knowledge.retrieve("texto")


def test_local_container_requires_no_network_and_uses_local_index(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, provider="local", data_root=tmp_path)
    container = build_container(
        settings,
        llm=FakeLocalProvider(),
        embeddings=FakeEmbeddingProvider(),
    )

    assert container.status.provider == "local"
    assert container.status.embeddings_provider == "local"
    assert container.status.network_required is False
    assert container.status.index_dir == tmp_path / "index"
    assert container.status.index_ready is False


def test_local_container_never_constructs_gemini_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_gemini(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        pytest.fail("O modo local tentou construir um componente Gemini.")

    monkeypatch.setattr(bootstrap, "GeminiProvider", unexpected_gemini)
    monkeypatch.setattr(bootstrap, "build_gemini_embedding_provider", unexpected_gemini)
    monkeypatch.setattr(
        bootstrap,
        "build_local_embedding_provider",
        lambda settings: FakeEmbeddingProvider(),
    )

    container = build_container(Settings(_env_file=None, provider="local", data_root=tmp_path))

    assert isinstance(container.llm, LocalLLMProvider)
    assert container.status.network_required is False


@pytest.mark.asyncio
async def test_assistant_answers_offline_with_local_components() -> None:
    class LocalKnowledgeBase:
        ready = True

        async def retrieve(self, query: str, *, limit: int | None = None) -> list[Any]:
            del query, limit
            from navi.domain.models import KnowledgeChunk

            return [KnowledgeChunk("Trecho tributário", "Tributos.pdf", page=2)]

    service = AssistantService(
        llm=FakeLocalProvider(),
        knowledge_base=LocalKnowledgeBase(),
        conversations=InMemoryConversationStore(),
        human_contact="Fale com o NAF.",
    )

    response = await service.answer(session_id="desktop:local", question="Pergunta")

    assert response.answer == "Resposta local simulada."
    assert response.provider == "local"
    assert response.sources[0].label == "Tributos.pdf, p. 2"


@pytest.mark.asyncio
async def test_desktop_controller_uses_and_clears_its_own_session(tmp_path: Path) -> None:
    class FakeAssistant:
        def __init__(self) -> None:
            self.answered_session: str | None = None
            self.cleared_session: str | None = None

        async def answer(self, *, session_id: str, question: str) -> AssistantResponse:
            self.answered_session = session_id
            return AssistantResponse(answer=question)

        async def clear_history(self, session_id: str) -> None:
            self.cleared_session = session_id

    assistant = FakeAssistant()
    status = RuntimeStatus(
        provider="local",
        model="fake.gguf",
        embeddings_provider="local",
        embeddings_model="fake",
        network_required=False,
        index_ready=True,
        model_ready=True,
        index_dir=tmp_path / "index",
        data_root=tmp_path,
    )
    controller = DesktopController(SimpleNamespace(assistant=assistant, status=status))

    response = await controller.answer("Pergunta")
    await controller.new_conversation()

    assert response.answer == "Pergunta"
    assert assistant.answered_session == "desktop:local"
    assert assistant.cleared_session == "desktop:local"
