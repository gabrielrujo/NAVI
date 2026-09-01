from __future__ import annotations

import asyncio
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
    ) -> None:
        self._storage_dir = storage_dir
        self._embed_model = embed_model
        self._top_k = top_k
        self._min_score = min_score
        self._index: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        required = ("docstore.json", "index_store.json")
        return all((self._storage_dir / name).is_file() for name in required)

    async def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        if not self.ready:
            raise KnowledgeBaseNotReadyError(
                "O indice do RAG ainda nao existe. Execute `navi ingest`."
            )
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

