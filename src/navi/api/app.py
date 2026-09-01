from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from navi.bootstrap import ApplicationContainer
from navi.domain.models import (
    InvalidQuestionError,
    KnowledgeBaseNotReadyError,
    ProviderError,
)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1)


class SourceResponse(BaseModel):
    name: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    provider: str
    model: str


def create_app(container: ApplicationContainer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await container.aclose()

    app = FastAPI(
        title="NAVI - Assistente do NAF",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "rag_ready": container.knowledge_base.ready,
            "provider": container.llm.provider_name,
            "model": container.llm.model_name,
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        try:
            response = await container.assistant.answer(
                session_id=f"api:{payload.session_id}",
                question=payload.message,
            )
        except InvalidQuestionError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        except KnowledgeBaseNotReadyError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "O provedor de IA esta temporariamente indisponivel.",
            ) from exc

        return ChatResponse(
            answer=response.answer,
            sources=[SourceResponse(name=item.name, page=item.page) for item in response.sources],
            provider=response.provider,
            model=response.model,
        )

    @app.delete("/v1/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def clear_chat(session_id: str) -> None:
        await container.assistant.clear_history(f"api:{session_id}")

    return app

