from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from navi.domain.models import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal["gemini"] = Field(default="gemini", validation_alias="NAVI_PROVIDER")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash-lite", validation_alias="GEMINI_MODEL")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GEMINI_BASE_URL",
    )
    gemini_timeout_seconds: float = Field(
        default=30.0, gt=0, validation_alias="GEMINI_TIMEOUT_SECONDS"
    )
    gemini_max_output_tokens: int = Field(
        default=1024, ge=128, le=8192, validation_alias="GEMINI_MAX_OUTPUT_TOKENS"
    )

    documents_dir: Path = Field(
        default=Path("data/documents"), validation_alias="NAVI_DOCUMENTS_DIR"
    )
    index_dir: Path = Field(default=Path("data/index"), validation_alias="NAVI_INDEX_DIR")
    embedding_model: str = Field(
        default="gemini-embedding-2", validation_alias="NAVI_EMBEDDING_MODEL"
    )
    embedding_batch_size: int = Field(
        default=20, ge=1, le=100, validation_alias="NAVI_EMBEDDING_BATCH_SIZE"
    )
    embedding_texts_per_minute: int = Field(
        default=100, ge=1, validation_alias="NAVI_EMBEDDING_TEXTS_PER_MINUTE"
    )
    embedding_max_attempts: int = Field(
        default=5, ge=1, le=10, validation_alias="NAVI_EMBEDDING_MAX_ATTEMPTS"
    )
    embedding_retry_base_seconds: float = Field(
        default=2.0, gt=0, validation_alias="NAVI_EMBEDDING_RETRY_BASE_SECONDS"
    )
    embedding_retry_max_seconds: float = Field(
        default=120.0, gt=0, validation_alias="NAVI_EMBEDDING_RETRY_MAX_SECONDS"
    )
    rag_top_k: int = Field(default=5, ge=1, le=20, validation_alias="NAVI_RAG_TOP_K")
    rag_min_score: float = Field(default=0.20, ge=-1, le=1, validation_alias="NAVI_RAG_MIN_SCORE")
    chunk_size: int = Field(default=900, ge=200, validation_alias="NAVI_CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, ge=0, validation_alias="NAVI_CHUNK_OVERLAP")

    telegram_bot_token: SecretStr | None = Field(
        default=None, validation_alias="TELEGRAM_BOT_TOKEN"
    )
    human_contact: str = Field(
        default="Procure a equipe do NAF para atendimento individualizado.",
        validation_alias="NAVI_HUMAN_CONTACT",
    )
    max_history_messages: int = Field(
        default=8, ge=0, le=30, validation_alias="NAVI_MAX_HISTORY_MESSAGES"
    )
    max_question_chars: int = Field(
        default=3000, ge=100, le=10000, validation_alias="NAVI_MAX_QUESTION_CHARS"
    )
    api_host: str = Field(default="0.0.0.0", validation_alias="NAVI_API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, validation_alias="NAVI_API_PORT")
    log_level: str = Field(default="INFO", validation_alias="NAVI_LOG_LEVEL")

    def require_gemini_api_key(self) -> str:
        if self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip():
            raise ConfigurationError("Defina GEMINI_API_KEY no arquivo .env.")
        return self.gemini_api_key.get_secret_value()

    def require_telegram_token(self) -> str:
        if self.telegram_bot_token is None:
            raise ConfigurationError("Defina TELEGRAM_BOT_TOKEN no arquivo .env.")
        token = self.telegram_bot_token.get_secret_value().strip()
        if not token:
            raise ConfigurationError("Defina TELEGRAM_BOT_TOKEN no arquivo .env.")
        return token
