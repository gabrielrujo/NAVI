from __future__ import annotations

import json
from pathlib import Path

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from navi.infrastructure.rag import ingestion
from navi.infrastructure.rag.ingestion import (
    EmbeddingIngestionPolicy,
    _embed_nodes_incrementally,
    ingest_documents,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeRateLimitError(Exception):
    code = 429

    def __init__(self, retry_delay: str = "49s") -> None:
        self.details = {
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay,
                    }
                ]
            }
        }
        super().__init__(f"429 RESOURCE_EXHAUSTED. Please retry in {retry_delay}.")


class FakeTransientError(Exception):
    code = 503


class FakeEmbeddingModel:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[list[str]] = []

    def get_text_embedding_batch(
        self, texts: list[str], *, show_progress: bool = False
    ) -> list[list[float]]:
        assert show_progress is False
        self.calls.append(list(texts))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]


def _nodes(*texts: str) -> list[TextNode]:
    return [TextNode(text=text) for text in texts]


def _run_embedding(
    *,
    nodes: list[TextNode],
    model: FakeEmbeddingModel,
    checkpoint_path: Path,
    policy: EmbeddingIngestionPolicy,
    clock: FakeClock,
) -> None:
    _embed_nodes_incrementally(
        nodes=nodes,
        embed_model=model,
        embedding_model_name="gemini-embedding-2",
        checkpoint_path=checkpoint_path,
        policy=policy,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        jitter=lambda _start, _end: 0.0,
    )


def test_rate_limit_uses_retry_info_then_succeeds(tmp_path: Path) -> None:
    clock = FakeClock()
    model = FakeEmbeddingModel([FakeRateLimitError("49s"), None])
    nodes = _nodes("primeiro", "segundo")

    _run_embedding(
        nodes=nodes,
        model=model,
        checkpoint_path=tmp_path / "checkpoint.json",
        policy=EmbeddingIngestionPolicy(batch_size=2, max_attempts=3),
        clock=clock,
    )

    assert len(model.calls) == 2
    assert clock.sleeps == [49.0]
    assert all(node.embedding is not None for node in nodes)


def test_rate_limiter_counts_every_text_inside_batches(tmp_path: Path) -> None:
    clock = FakeClock()
    model = FakeEmbeddingModel()

    _run_embedding(
        nodes=_nodes("a", "b", "c", "d", "e", "f"),
        model=model,
        checkpoint_path=tmp_path / "checkpoint.json",
        policy=EmbeddingIngestionPolicy(batch_size=2, texts_per_minute=4),
        clock=clock,
    )

    assert [len(call) for call in model.calls] == [2, 2, 2]
    assert clock.sleeps == [61.0]


def test_checkpoint_resumes_without_reembedding_completed_nodes(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "index.ingest" / "embeddings.json"
    first_nodes = _nodes("a", "b", "c")
    first_model = FakeEmbeddingModel([None, RuntimeError("interrompido")])

    with pytest.raises(RuntimeError, match="interrompido"):
        _run_embedding(
            nodes=first_nodes,
            model=first_model,
            checkpoint_path=checkpoint_path,
            policy=EmbeddingIngestionPolicy(batch_size=2),
            clock=FakeClock(),
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed"] == 2

    resumed_nodes = _nodes("a", "b", "c")
    resumed_model = FakeEmbeddingModel()
    _run_embedding(
        nodes=resumed_nodes,
        model=resumed_model,
        checkpoint_path=checkpoint_path,
        policy=EmbeddingIngestionPolicy(batch_size=2),
        clock=FakeClock(),
    )

    assert len(resumed_model.calls) == 1
    assert len(resumed_model.calls[0]) == 1
    assert all(node.embedding is not None for node in resumed_nodes)


def test_retry_stops_at_configured_attempt_limit(tmp_path: Path) -> None:
    clock = FakeClock()
    model = FakeEmbeddingModel([FakeRateLimitError("3s"), FakeRateLimitError("4s")])

    with pytest.raises(FakeRateLimitError):
        _run_embedding(
            nodes=_nodes("a"),
            model=model,
            checkpoint_path=tmp_path / "checkpoint.json",
            policy=EmbeddingIngestionPolicy(batch_size=1, max_attempts=2),
            clock=clock,
        )

    assert len(model.calls) == 2
    assert clock.sleeps == [3.0]
    assert not (tmp_path / "checkpoint.json").exists()


def test_transient_error_uses_exponential_backoff(tmp_path: Path) -> None:
    clock = FakeClock()
    model = FakeEmbeddingModel(
        [FakeTransientError("indisponivel"), FakeTransientError("indisponivel"), None]
    )

    _run_embedding(
        nodes=_nodes("a"),
        model=model,
        checkpoint_path=tmp_path / "checkpoint.json",
        policy=EmbeddingIngestionPolicy(
            batch_size=1,
            max_attempts=3,
            retry_base_seconds=2.0,
            retry_max_seconds=10.0,
        ),
        clock=clock,
    )

    assert len(model.calls) == 3
    assert clock.sleeps == [2.0, 4.0]


def test_failed_rebuild_keeps_existing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage_dir = tmp_path / "index"
    storage_dir.mkdir()
    docstore = storage_dir / "docstore.json"
    docstore.write_text("indice-valido", encoding="utf-8")
    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        lambda _documents_dir: ([Document(text="conteudo tributario")], []),
    )
    model = FakeEmbeddingModel([RuntimeError("falha antes da publicacao")])

    with pytest.raises(RuntimeError, match="falha antes da publicacao"):
        ingest_documents(
            documents_dir=tmp_path / "documents",
            storage_dir=storage_dir,
            embed_model=model,
            embedding_model_name="gemini-embedding-2",
            chunk_size=200,
            chunk_overlap=20,
            rebuild=True,
            embedding_policy=EmbeddingIngestionPolicy(batch_size=1),
        )

    assert docstore.read_text(encoding="utf-8") == "indice-valido"
    assert not storage_dir.with_name("index.backup").exists()
