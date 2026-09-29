from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage

from navi.domain.models import KnowledgeBaseNotReadyError, KnowledgeChunk


class LlamaIndexKnowledgeBase:
    """Adaptador de recuperacao vetorial. Nao possui acesso ao LLM generativo."""

    def __init__(
        self,
        *,
        storage_dir: Path,
        embed_model: Any,
        top_k: int = 5,
        min_score: float = 0.20,
        expected_embedding_provider: str | None = None,
        expected_embedding_model: str | None = None,
    ) -> None:
        self._storage_dir = storage_dir
        self._embed_model = embed_model
        self._top_k = top_k
        self._min_score = min_score
        self._expected_embedding_provider = expected_embedding_provider
        self._expected_embedding_model = expected_embedding_model
        self._index: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._index_files_exist() and not self._manifest_compatibility_error()

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def _index_files_exist(self) -> bool:
        required = ("docstore.json", "index_store.json")
        return all((self._storage_dir / name).is_file() for name in required)

    def _manifest_compatibility_error(self) -> str | None:
        if self._expected_embedding_provider is None and self._expected_embedding_model is None:
            return None
        manifest_path = self._storage_dir / "manifest.json"
        if not manifest_path.is_file():
            return "O índice não possui manifesto de compatibilidade. Recrie a base."
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return "O manifesto do índice é inválido. Recrie a base."

        provider = manifest.get("embedding_provider", "gemini")
        model = manifest.get("embedding_model")
        if (
            self._expected_embedding_provider is not None
            and provider != self._expected_embedding_provider
        ):
            return "O índice foi criado com outro provider de embeddings. Recrie a base."
        if self._expected_embedding_model is not None and model != self._expected_embedding_model:
            return "O índice foi criado com outro modelo de embeddings. Recrie a base."
        return None

    async def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        if not self._index_files_exist():
            command = (
                "navi ingest-local"
                if self._expected_embedding_provider == "local"
                else "navi ingest"
            )
            raise KnowledgeBaseNotReadyError(
                f"O indice do RAG ainda nao existe. Execute `{command}`."
            )
        compatibility_error = self._manifest_compatibility_error()
        if compatibility_error is not None:
            raise KnowledgeBaseNotReadyError(compatibility_error)
        async with self._load_lock:
            if self._index is not None:
                return

            def load() -> Any:
                storage = StorageContext.from_defaults(persist_dir=str(self._storage_dir))
                return load_index_from_storage(storage, embed_model=self._embed_model)

            self._index = await asyncio.to_thread(load)

    async def retrieve(self, query: str, *, limit: int | None = None) -> list[KnowledgeChunk]:
        await self._ensure_loaded()
        retriever = self._index.as_retriever(similarity_top_k=limit or self._top_k)
        nodes = await retriever.aretrieve(query)

        chunks: list[KnowledgeChunk] = []
        for result in nodes:
            score = float(result.score) if result.score is not None else None
            if score is not None and score < self._min_score:
                continue
            metadata = result.node.metadata or {}
            page = metadata.get("page_number")
            chunks.append(
                KnowledgeChunk(
                    text=result.node.get_content().strip(),
                    source_name=str(metadata.get("source_name", "Documento do NAF")),
                    page=int(page) if page is not None else None,
                    score=score,
                )
            )
        return chunks
