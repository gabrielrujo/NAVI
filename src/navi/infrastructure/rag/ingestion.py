from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1
_RATE_WINDOW_SECONDS = 60.0
_RATE_LIMIT_SAFETY_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_SECONDS_PATTERN = re.compile(r"(?P<seconds>\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IngestionStats:
    files: int
    pages: int
    chunks: int
    storage_dir: str


@dataclass(frozen=True, slots=True)
class EmbeddingIngestionPolicy:
    """Limites aplicados por texto, inclusive quando a API usa batchEmbedContents."""

    batch_size: int = 20
    texts_per_minute: int = 100
    max_attempts: int = 5
    retry_base_seconds: float = 2.0
    retry_max_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("O tamanho do lote de embeddings deve ser positivo.")
        if self.texts_per_minute < 1:
            raise ValueError("O limite de textos por minuto deve ser positivo.")
        if self.max_attempts < 1:
            raise ValueError("O numero maximo de tentativas deve ser positivo.")
        if self.retry_base_seconds <= 0 or self.retry_max_seconds <= 0:
            raise ValueError("Os intervalos de retry devem ser positivos.")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("O intervalo maximo de retry deve ser maior ou igual ao inicial.")


class _TextRateLimiter:
    """Janela deslizante que contabiliza cada texto, nao somente chamadas HTTP."""

    def __init__(
        self,
        *,
        texts_per_minute: int,
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        self._limit = texts_per_minute
        self._sleep = sleep
        self._monotonic = monotonic
        self._completed_batches: deque[tuple[float, int]] = deque()

    def wait_for_capacity(self, texts: int) -> None:
        if texts > self._limit:
            raise ValueError(
                f"Um lote com {texts} textos excede o limite de {self._limit} por minuto."
            )

        while True:
            now = self._monotonic()
            cutoff = now - _RATE_WINDOW_SECONDS
            while self._completed_batches and self._completed_batches[0][0] <= cutoff:
                self._completed_batches.popleft()

            used = sum(count for _, count in self._completed_batches)
            if used + texts <= self._limit:
                return

            wait_seconds = max(
                0.0,
                self._completed_batches[0][0]
                + _RATE_WINDOW_SECONDS
                + _RATE_LIMIT_SAFETY_SECONDS
                - now,
            )
            logger.info(
                "Cota local de embeddings atingida (%d/%d textos); aguardando %.1f s.",
                used,
                self._limit,
                wait_seconds,
            )
            self._sleep(wait_seconds)

    def record_success(self, texts: int) -> None:
        self._completed_batches.append((self._monotonic(), texts))


def load_pdf_documents(documents_dir: Path) -> tuple[list[Document], list[dict[str, Any]]]:
    paths = sorted(documents_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {documents_dir}.")

    documents: list[Document] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        reader = PdfReader(path)
        extracted_pages = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            extracted_pages += 1
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_name": path.name,
                        "page_number": page_number,
                    },
                    excluded_llm_metadata_keys=["page_number"],
                )
            )
        sources.append(
            {
                "name": path.name,
                "pages_total": len(reader.pages),
                "pages_with_text": extracted_pages,
            }
        )
    return documents, sources


def _node_embedding_text(node: BaseNode) -> str:
    return node.get_content(metadata_mode=MetadataMode.EMBED)


def _embedding_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checkpoint_path(storage_dir: Path) -> Path:
    return storage_dir.with_name(f"{storage_dir.name}.ingest") / "embeddings.json"


def _load_embedding_checkpoint(path: Path, *, model_name: str) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _CHECKPOINT_VERSION:
            raise ValueError("versao incompativel")
        if payload.get("embedding_model") != model_name:
            logger.info(
                "Checkpoint de embeddings pertence ao modelo %s; iniciando cache para %s.",
                payload.get("embedding_model", "desconhecido"),
                model_name,
            )
            return {}
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_embeddings, dict):
            raise ValueError("campo embeddings invalido")

        embeddings: dict[str, list[float]] = {}
        for key, values in raw_embeddings.items():
            if not isinstance(key, str) or not isinstance(values, list) or not values:
                continue
            try:
                embeddings[key] = [float(value) for value in values]
            except (TypeError, ValueError):
                continue
        return embeddings
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Checkpoint de embeddings invalido em %s: %s", path, exc)
        return {}


def _save_embedding_checkpoint(
    path: Path,
    *,
    model_name: str,
    embeddings: dict[str, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    payload = {
        "version": _CHECKPOINT_VERSION,
        "embedding_model": model_name,
        "updated_at": datetime.now(UTC).isoformat(),
        "completed": len(embeddings),
        "embeddings": embeddings,
    }
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _duration_seconds(value: object) -> float | None:
    if isinstance(value, int | float):
        return max(0.0, float(value))
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    try:
        return max(0.0, float(stripped))
    except ValueError:
        pass
    match = _SECONDS_PATTERN.fullmatch(stripped)
    return float(match.group("seconds")) if match else None


def _find_retry_delays(value: object) -> list[float]:
    delays: list[float] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"retryDelay", "retry_delay"}:
                delay = _duration_seconds(nested)
                if delay is not None:
                    delays.append(delay)
            delays.extend(_find_retry_delays(nested))
    elif isinstance(value, list):
        for nested in value:
            delays.extend(_find_retry_delays(nested))
    return delays


def _retry_delay_seconds(exc: BaseException) -> float | None:
    delays = _find_retry_delays(getattr(exc, "details", None))

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_after = headers.get("retry-after")
        delay = _duration_seconds(retry_after)
        if delay is not None:
            delays.append(delay)

    message = str(exc)
    retry_match = re.search(
        r"(?:retry(?:ing)?\s+(?:in|after)|retryDelay[^\d]*)\s*(\d+(?:\.\d+)?)\s*s",
        message,
        re.IGNORECASE,
    )
    if retry_match:
        delays.append(float(retry_match.group(1)))
    return max(delays) if delays else None


def _is_retryable_embedding_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    return code in _RETRYABLE_STATUS_CODES or isinstance(
        exc, (ConnectionError, TimeoutError, httpx.HTTPError)
    )


def _embed_batch_with_retry(
    *,
    embed_model: Any,
    texts: list[str],
    policy: EmbeddingIngestionPolicy,
    limiter: _TextRateLimiter,
    sleep: Callable[[float], None],
    jitter: Callable[[float, float], float],
) -> list[list[float]]:
    for attempt in range(1, policy.max_attempts + 1):
        limiter.wait_for_capacity(len(texts))
        try:
            result = embed_model.get_text_embedding_batch(texts, show_progress=False)
        except Exception as exc:
            if not _is_retryable_embedding_error(exc) or attempt == policy.max_attempts:
                logger.error(
                    "Falha ao gerar lote de %d embeddings na tentativa %d/%d.",
                    len(texts),
                    attempt,
                    policy.max_attempts,
                )
                raise

            server_delay = _retry_delay_seconds(exc)
            exponential_delay = min(
                policy.retry_max_seconds,
                policy.retry_base_seconds * (2 ** (attempt - 1)),
            )
            delay = max(server_delay or 0.0, exponential_delay)
            delay += jitter(0.0, min(1.0, delay * 0.1))
            logger.warning(
                "API de embeddings retornou %s; aguardando %.1f s antes da tentativa %d/%d%s.",
                getattr(exc, "code", exc.__class__.__name__),
                delay,
                attempt + 1,
                policy.max_attempts,
                " (RetryInfo respeitado)" if server_delay is not None else "",
            )
            sleep(delay)
            continue

        if len(result) != len(texts):
            raise RuntimeError(
                f"O provider retornou {len(result)} embeddings para {len(texts)} textos."
            )
        limiter.record_success(len(texts))
        return [[float(value) for value in embedding] for embedding in result]

    raise RuntimeError("O lote de embeddings excedeu o limite de tentativas.")


def _embed_nodes_incrementally(
    *,
    nodes: Sequence[BaseNode],
    embed_model: Any,
    embedding_model_name: str,
    checkpoint_path: Path,
    policy: EmbeddingIngestionPolicy,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    jitter: Callable[[float, float], float] = random.uniform,
) -> None:
    cached_embeddings = _load_embedding_checkpoint(
        checkpoint_path,
        model_name=embedding_model_name,
    )
    pending_by_key: dict[str, tuple[str, list[BaseNode]]] = {}
    resumed_nodes = 0

    for node in nodes:
        text = _node_embedding_text(node)
        key = _embedding_cache_key(text)
        cached = cached_embeddings.get(key)
        if cached is not None:
            node.embedding = cached
            resumed_nodes += 1
            continue
        if key not in pending_by_key:
            pending_by_key[key] = (text, [])
        pending_by_key[key][1].append(node)

    pending = list(pending_by_key.items())
    logger.info(
        "Embeddings: %d nodes totais, %d recuperados do checkpoint, %d textos pendentes.",
        len(nodes),
        resumed_nodes,
        len(pending),
    )
    if not pending:
        return

    limiter = _TextRateLimiter(
        texts_per_minute=policy.texts_per_minute,
        sleep=sleep,
        monotonic=monotonic,
    )
    batch_size = min(policy.batch_size, policy.texts_per_minute)
    total_batches = (len(pending) + batch_size - 1) // batch_size
    completed = resumed_nodes

    for batch_number, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start : start + batch_size]
        texts = [item[1][0] for item in batch]
        logger.info(
            "Embeddings: processando lote %d/%d (%d textos; %d/%d nodes concluidos).",
            batch_number,
            total_batches,
            len(texts),
            completed,
            len(nodes),
        )
        embeddings = _embed_batch_with_retry(
            embed_model=embed_model,
            texts=texts,
            policy=policy,
            limiter=limiter,
            sleep=sleep,
            jitter=jitter,
        )
        for (key, (_, matching_nodes)), embedding in zip(batch, embeddings, strict=True):
            cached_embeddings[key] = embedding
            for node in matching_nodes:
                node.embedding = embedding
                completed += 1
        _save_embedding_checkpoint(
            checkpoint_path,
            model_name=embedding_model_name,
            embeddings=cached_embeddings,
        )
        logger.info("Embeddings: checkpoint salvo; %d/%d nodes concluidos.", completed, len(nodes))


def ingest_documents(
    *,
    documents_dir: Path,
    storage_dir: Path,
    embed_model: Any,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
    embedding_policy: EmbeddingIngestionPolicy | None = None,
) -> IngestionStats:
    if chunk_overlap >= chunk_size:
        raise ValueError("NAVI_CHUNK_OVERLAP deve ser menor que NAVI_CHUNK_SIZE.")
    if (storage_dir / "docstore.json").exists() and not rebuild:
        raise FileExistsError(
            f"Ja existe um indice em {storage_dir}. Use `navi ingest --rebuild` para recriar."
        )

    documents, sources = load_pdf_documents(documents_dir)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    _embed_nodes_incrementally(
        nodes=nodes,
        embed_model=embed_model,
        embedding_model_name=embedding_model_name,
        checkpoint_path=_checkpoint_path(storage_dir),
        policy=embedding_policy or EmbeddingIngestionPolicy(),
    )
    logger.info("Embeddings concluidos; construindo o indice vetorial local.")
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model, show_progress=False)

    storage_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="navi-index-", dir=storage_dir.parent))
    try:
        index.storage_context.persist(persist_dir=str(temp_dir))
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "embedding_model": embedding_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks": len(nodes),
            "sources": sources,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        moved_existing_index = False
        if storage_dir.exists():
            backup_dir = storage_dir.with_name(f"{storage_dir.name}.backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            storage_dir.replace(backup_dir)
            moved_existing_index = True
        try:
            temp_dir.replace(storage_dir)
        except Exception:
            if moved_existing_index and not storage_dir.exists():
                backup_dir.replace(storage_dir)
            raise
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return IngestionStats(
        files=len(sources),
        pages=len(documents),
        chunks=len(nodes),
        storage_dir=str(storage_dir),
    )


def stats_as_json(stats: IngestionStats) -> str:
    return json.dumps(asdict(stats), ensure_ascii=False, indent=2)
