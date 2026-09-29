from __future__ import annotations

from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Cria o modelo de embeddings usado pela ingestão e pela recuperação."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def network_required(self) -> bool: ...

    def build_model(self, *, retries: int = 3) -> Any: ...
