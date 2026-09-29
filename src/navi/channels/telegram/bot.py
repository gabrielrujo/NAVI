from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from navi.bootstrap import ApplicationContainer
from navi.channels.telegram.formatting import format_response, split_telegram_text
from navi.domain.models import (
    EmbeddingError,
    InvalidQuestionError,
    KnowledgeBaseNotReadyError,
    ProviderError,
)
from navi.services.assistant import AssistantService

logger = logging.getLogger(__name__)


def _session_id(message: Message) -> str:
    user_id = message.from_user.id if message.from_user else 0
    return f"telegram:{message.chat.id}:{user_id}"


def build_router() -> Router:
    router = Router(name="navi-telegram")

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(
            "Olá! Eu sou o NAVI, assistente virtual do NAF. 👋\n\n"
            "Posso explicar temas tributários com base nos materiais disponibilizados pelo NAF. "
            "Envie sua dúvida em texto. Para apagar o histórico, use /limpar."
        )

    @router.message(Command("ajuda"))
    async def help_message(message: Message) -> None:
        await message.answer(
            "Faça uma pergunta tributária em texto. Minhas respostas são informativas e não "
            "substituem uma análise contábil, fiscal ou jurídica. Não envie CPF, senhas ou "
            "dados bancários."
        )

    @router.message(Command("limpar"))
    async def clear(message: Message, assistant: AssistantService) -> None:
        await assistant.clear_history(_session_id(message))
        await message.answer("Histórico desta conversa apagado.")

    @router.message(F.text)
    async def answer(message: Message, assistant: AssistantService) -> None:
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        try:
            response = await assistant.answer(
                session_id=_session_id(message),
                question=message.text or "",
            )
        except InvalidQuestionError as exc:
            await message.answer(str(exc))
            return
        except KnowledgeBaseNotReadyError:
            await message.answer(
                "Minha base de conhecimento ainda está sendo preparada. Tente novamente mais tarde."
            )
            return
        except EmbeddingError:
            logger.exception("Falha nos embeddings ao responder mensagem do Telegram")
            await message.answer(
                "A base de conhecimento está temporariamente indisponível. "
                "Tente novamente em instantes."
            )
            return
        except ProviderError:
            logger.exception("Falha no provider ao responder mensagem do Telegram")
            await message.answer(
                "O serviço de IA está temporariamente indisponível. Tente novamente em instantes."
            )
            return

        for part in split_telegram_text(format_response(response)):
            await message.answer(part)

    @router.message()
    async def unsupported(message: Message) -> None:
        await message.answer("Neste protótipo, envie sua pergunta como mensagem de texto.")

    return router


async def run_telegram(*, token: str, container: ApplicationContainer) -> None:
    bot = Bot(token=token)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router())
    try:
        await dispatcher.start_polling(
            bot,
            assistant=container.assistant,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        await bot.session.close()
        await container.aclose()
