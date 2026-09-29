from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence

import uvicorn

from navi.api.app import create_app
from navi.bootstrap import (
    build_container,
    build_gemini_embedding_provider,
    build_local_embedding_provider,
)
from navi.channels.telegram.bot import run_telegram
from navi.config import Settings
from navi.domain.models import ConfigurationError, EmbeddingError
from navi.infrastructure.portable import portable_layout
from navi.infrastructure.rag.ingestion import (
    EmbeddingIngestionPolicy,
    ingest_documents,
    stats_as_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navi", description="NAVI - assistente do NAF")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Cria o indice RAG a partir dos PDFs")
    ingest.add_argument(
        "--rebuild",
        action="store_true",
        help="Recria um indice existente e conserva o anterior em data/index.backup",
    )
    ingest_local = subcommands.add_parser(
        "ingest-local", help="Cria o indice portátil usando embeddings locais"
    )
    ingest_local.add_argument(
        "--rebuild",
        action="store_true",
        help="Recria o indice local e conserva o anterior em runtime/index.backup",
    )
    subcommands.add_parser("api", help="Inicia o canal HTTP/FastAPI")
    subcommands.add_parser("telegram", help="Inicia o bot do Telegram via long polling")
    desktop = subcommands.add_parser("app", help="Abre o aplicativo gráfico local")
    desktop.add_argument(
        "--fullscreen",
        action="store_true",
        help="Abre a janela em tela cheia para simular o futuro modo totem",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)

    try:
        if args.command == "ingest":
            embeddings = build_gemini_embedding_provider(settings)
            stats = ingest_documents(
                documents_dir=settings.documents_dir,
                storage_dir=settings.index_dir,
                # A ingestao controla o retry para poder respeitar RetryInfo e salvar checkpoints.
                embed_model=embeddings.build_model(retries=1),
                embedding_model_name=embeddings.model_name,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                rebuild=args.rebuild,
                embedding_policy=EmbeddingIngestionPolicy(
                    batch_size=settings.embedding_batch_size,
                    texts_per_minute=settings.embedding_texts_per_minute,
                    max_attempts=settings.embedding_max_attempts,
                    retry_base_seconds=settings.embedding_retry_base_seconds,
                    retry_max_seconds=settings.embedding_retry_max_seconds,
                ),
                embedding_provider_name=embeddings.provider_name,
            )
            print(stats_as_json(stats))
            return 0

        if args.command == "ingest-local":
            layout = portable_layout(settings)
            layout.ensure()
            embeddings = build_local_embedding_provider(settings)
            stats = ingest_documents(
                documents_dir=layout.documents,
                storage_dir=layout.index,
                embed_model=embeddings.build_model(retries=1),
                embedding_model_name=embeddings.model_name,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                rebuild=args.rebuild,
                embedding_policy=EmbeddingIngestionPolicy(
                    batch_size=settings.local_embedding_batch_size,
                    texts_per_minute=1_000_000,
                    max_attempts=1,
                ),
                embedding_provider_name=embeddings.provider_name,
                checkpoint_path=settings.local_ingestion_checkpoint_path,
            )
            print(stats_as_json(stats))
            return 0

        if args.command == "app":
            from navi.channels.desktop.app import run_desktop

            return run_desktop(settings=settings, fullscreen=args.fullscreen)

        container = build_container(settings)
        if args.command == "api":
            uvicorn.run(
                create_app(container),
                host=settings.api_host,
                port=settings.api_port,
                log_level=settings.log_level.lower(),
            )
            return 0

        if args.command == "telegram":
            asyncio.run(
                run_telegram(
                    token=settings.require_telegram_token(),
                    container=container,
                )
            )
            return 0
    except (
        ConfigurationError,
        EmbeddingError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
