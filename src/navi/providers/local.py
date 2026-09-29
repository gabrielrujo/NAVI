from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from navi.domain.models import ChatMessage, MessageRole, ProviderError


class LocalLLMProvider:
    """Executa um modelo GGUF local por meio de llama-cpp-python."""

    def __init__(
        self,
        *,
        model_path: Path | None,
        context_size: int = 4096,
        max_output_tokens: int = 512,
        threads: int | None = None,
        chat_format: str | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._context_size = context_size
        self._max_output_tokens = max_output_tokens
        self._threads = threads
        self._chat_format = chat_format
        self._backend_factory = backend_factory
        self._backend: Any | None = None
        self._lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self._model_path.name if self._model_path is not None else "não configurado"

    @property
    def network_required(self) -> bool:
        return False

    @property
    def model_available(self) -> bool:
        return self._model_path is not None and self._model_path.is_file()

    def _load_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        if self._model_path is None:
            raise ProviderError("Configure NAVI_LOCAL_MODEL_PATH para um arquivo GGUF.")
        if not self._model_path.is_file():
            raise ProviderError("Modelo local não encontrado no caminho configurado.")

        factory = self._backend_factory
        if factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise ProviderError(
                    "llama-cpp-python não está instalado. Instale as dependências do modo local."
                ) from exc
            factory = Llama

        options: dict[str, Any] = {
            "model_path": str(self._model_path),
            "n_ctx": self._context_size,
            "n_gpu_layers": 0,
            "verbose": False,
        }
        if self._threads is not None:
            options["n_threads"] = self._threads
        if self._chat_format:
            options["chat_format"] = self._chat_format
        try:
            self._backend = factory(**options)
        except Exception as exc:
            raise ProviderError("Não foi possível carregar o modelo GGUF local.") from exc
        return self._backend

    def _generate_sync(self, *, system_prompt: str, messages: Sequence[ChatMessage]) -> str:
        with self._lock:
            backend = self._load_backend()
            chat_messages = [{"role": "system", "content": system_prompt}]
            chat_messages.extend(
                {
                    "role": "assistant" if message.role is MessageRole.ASSISTANT else "user",
                    "content": message.content,
                }
                for message in messages
            )
            try:
                response = backend.create_chat_completion(
                    messages=chat_messages,
                    temperature=0.2,
                    max_tokens=self._max_output_tokens,
                )
                text = response["choices"][0]["message"]["content"].strip()
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError("O modelo local não conseguiu gerar uma resposta.") from exc
            if not text:
                raise ProviderError("O modelo local retornou uma resposta sem texto.")
            return text

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: Sequence[ChatMessage],
    ) -> str:
        if not messages:
            raise ProviderError("A conversa enviada ao modelo local está vazia.")
        return await asyncio.to_thread(
            self._generate_sync,
            system_prompt=system_prompt,
            messages=messages,
        )

    async def aclose(self) -> None:
        if self._backend is None:
            return
        close = getattr(self._backend, "close", None)
        if close is not None:
            await asyncio.to_thread(close)
