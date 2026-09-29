from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.genai.types import EmbedContentConfig
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from pydantic import PrivateAttr

from navi.domain.models import EmbeddingError


@dataclass(frozen=True, slots=True)
class GeminiEmbeddingProvider:
    api_key: str = field(repr=False)
    model: str
    batch_size: int

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def network_required(self) -> bool:
        return True

    def build_model(self, *, retries: int = 3) -> Any:
        return GoogleGenAIEmbedding(
            model_name=self.model,
            api_key=self.api_key,
            embed_batch_size=self.batch_size,
            retries=retries,
            embedding_config=EmbedContentConfig(output_dimensionality=768),
        )


class FastEmbedEmbedding(BaseEmbedding):
    """Adaptador pequeno entre FastEmbed e a interface de embeddings do LlamaIndex."""

    _cache_dir: Path = PrivateAttr()
    _threads: int | None = PrivateAttr()
    _local_files_only: bool = PrivateAttr()
    _backend_factory: Callable[..., Any] | None = PrivateAttr()
    _backend: Any | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: Path,
        batch_size: int,
        threads: int | None = None,
        local_files_only: bool = True,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(model_name=model_name, embed_batch_size=batch_size)
        self._cache_dir = cache_dir
        self._threads = threads
        self._local_files_only = local_files_only
        self._backend_factory = backend_factory

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedEmbedding"

    def _get_backend(self) -> Any:
        if self._backend is not None:
            return self._backend

        # O build oficial do ONNX Runtime pode habilitar telemetria em macOS/Linux.
        # O modo local da NAVI deve permanecer offline e sem telemetria.
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        factory = self._backend_factory
        if factory is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbeddingError(
                    "FastEmbed não está instalado. Instale as dependências do modo local."
                ) from exc
            factory = TextEmbedding

        try:
            self._backend = factory(
                model_name=self.model_name,
                cache_dir=str(self._cache_dir),
                threads=self._threads,
                cuda=False,
                lazy_load=True,
                local_files_only=self._local_files_only,
            )
        except Exception as exc:
            mode = "no armazenamento local" if self._local_files_only else ""
            raise EmbeddingError(
                f"Não foi possível carregar o modelo de embeddings {mode}.".strip()
            ) from exc
        return self._backend

    @staticmethod
    def _as_list(values: Any) -> list[float]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [float(value) for value in values]

    def _get_query_embedding(self, query: str) -> list[float]:
        try:
            result = next(iter(self._get_backend().query_embed(query)))
            return self._as_list(result)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Falha ao gerar o embedding local da pergunta.") from exc

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self._get_backend().embed(texts, batch_size=self.embed_batch_size)
            return [self._as_list(embedding) for embedding in embeddings]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Falha ao gerar embeddings locais dos documentos.") from exc


@dataclass(frozen=True, slots=True)
class LocalEmbeddingProvider:
    model: str
    cache_dir: Path
    batch_size: int = 32
    threads: int | None = None
    local_files_only: bool = True
    backend_factory: Callable[..., Any] | None = field(default=None, repr=False)

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def network_required(self) -> bool:
        return False

    def build_model(self, *, retries: int = 3) -> FastEmbedEmbedding:
        del retries
        return FastEmbedEmbedding(
            model_name=self.model,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
            threads=self.threads,
            local_files_only=self.local_files_only,
            backend_factory=self.backend_factory,
        )
