from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from navi.config import Settings


@dataclass(frozen=True, slots=True)
class PortableLayout:
    root: Path
    models: Path
    documents: Path
    index: Path
    embeddings: Path
    config: Path

    def ensure(self) -> None:
        for path in (
            self.root,
            self.models,
            self.documents,
            self.index,
            self.embeddings,
            self.config,
        ):
            path.mkdir(parents=True, exist_ok=True)


def portable_layout(settings: Settings) -> PortableLayout:
    return PortableLayout(
        root=settings.data_root,
        models=settings.data_root / "models",
        documents=settings.resolved_local_documents_dir,
        index=settings.resolved_local_index_dir,
        embeddings=settings.resolved_local_embedding_cache_dir,
        config=settings.resolved_local_config_dir,
    )
