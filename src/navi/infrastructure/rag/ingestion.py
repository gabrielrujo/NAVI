from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class IngestionStats:
    files: int
    pages: int
    chunks: int
    storage_dir: str


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


def ingest_documents(
    *,
    documents_dir: Path,
    storage_dir: Path,
    embed_model: Any,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool = False,
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
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model, show_progress=True)

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

        if storage_dir.exists():
            backup_dir = storage_dir.with_name(f"{storage_dir.name}.backup")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            storage_dir.replace(backup_dir)
        temp_dir.replace(storage_dir)
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

