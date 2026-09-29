from __future__ import annotations

import asyncio
from collections.abc import Sequence
from urllib.parse import quote

import httpx

from navi.domain.models import ChatMessage, MessageRole, ProviderError


class GeminiProvider:
    """Provider Gemini via API REST, sem dependencias no Telegram ou no RAG."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 30.0,
        max_output_tokens: int = 1024,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_output_tokens = max_output_tokens
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def network_required(self) -> bool:
        return True

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: Sequence[ChatMessage],
    ) -> str:
        if not messages:
            raise ProviderError("A conversa enviada ao Gemini esta vazia.")

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "model" if message.role is MessageRole.ASSISTANT else "user",
                    "parts": [{"text": message.content}],
                }
                for message in messages
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": self._max_output_tokens,
            },
        }
        url = f"{self._base_url}/models/{quote(self._model, safe='')}:generateContent"

        response: httpx.Response | None = None
        for attempt in range(3):
            try:
                response = await self._client.post(
                    url,
                    headers={"x-goog-api-key": self._api_key},
                    json=payload,
                )
            except httpx.HTTPError as exc:
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise ProviderError("Nao foi possivel conectar ao Gemini.") from exc

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
            break

        if response is None:
            raise ProviderError("O Gemini nao retornou uma resposta.")
        if response.is_error:
            detail = self._safe_error_detail(response)
            raise ProviderError(f"Gemini retornou HTTP {response.status_code}: {detail}")

        try:
            body = response.json()
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Resposta inesperada ou bloqueada pelo Gemini.") from exc

        if not text:
            raise ProviderError("O Gemini retornou uma resposta sem texto.")
        return text

    @staticmethod
    def _safe_error_detail(response: httpx.Response) -> str:
        try:
            detail = response.json().get("error", {}).get("message", "erro nao detalhado")
        except (TypeError, ValueError):
            detail = "erro nao detalhado"
        return str(detail)[:300]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
