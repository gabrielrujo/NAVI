from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence

import uvicorn

from navi.api.app import create_app
from navi.bootstrap import build_container, build_embedding_model
from navi.channels.telegram.bot import run_telegram
from navi.config import Settings
from navi.domain.models import ConfigurationError
from navi.infrastructure.rag.ingestion import ingest_documents, stats_as_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navi", description="NAVI - assistente do NAF")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Cria o indice RAG a partir dos PDFs")
    ingest.add_argument(
        "--rebuild",
        action="store_true",
        help="Recria um indice existente e conserva o anterior em data/index.backup",
    )
    subcommands.add_parser("api", help="Inicia o canal HTTP/FastAPI")
    subcommands.add_parser("telegram", help="Inicia o bot do Telegram via long polling")
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
            stats = ingest_documents(
                documents_dir=settings.documents_dir,
                storage_dir=settings.index_dir,
                embed_model=build_embedding_model(settings),
                embedding_model_name=settings.embedding_model,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                rebuild=args.rebuild,
            )
            print(stats_as_json(stats))
            return 0

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
    except (ConfigurationError, FileNotFoundError, FileExistsError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

