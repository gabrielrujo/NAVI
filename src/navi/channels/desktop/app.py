from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from navi.bootstrap import build_container
from navi.channels.desktop.controller import DesktopController
from navi.config import Settings
from navi.domain.models import (
    ConfigurationError,
    EmbeddingError,
    InvalidQuestionError,
    KnowledgeBaseNotReadyError,
    ProviderError,
)
from navi.infrastructure.portable import portable_layout

logger = logging.getLogger(__name__)


def _friendly_error(exc: BaseException) -> str:
    if isinstance(exc, InvalidQuestionError | ProviderError | EmbeddingError):
        return str(exc)
    if isinstance(exc, KnowledgeBaseNotReadyError):
        return "Base documental ainda não foi preparada. Execute a ingestão correspondente."
    logger.exception("Falha inesperada no aplicativo desktop", exc_info=exc)
    return "Ocorreu um erro inesperado. Consulte o log técnico."


def run_desktop(*, settings: Settings, fullscreen: bool = False) -> int:
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtGui import QCloseEvent, QFont
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise ConfigurationError(
            "PySide6 não está instalado. Instale as dependências do aplicativo local."
        ) from exc

    portable_layout(settings).ensure()
    container = build_container(settings)
    controller = DesktopController(container)

    class AsyncCall(QThread):
        succeeded = Signal(object)
        failed = Signal(str)

        def __init__(
            self,
            operation: Callable[[], Coroutine[Any, Any, object]],
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._operation = operation

        def run(self) -> None:
            try:
                self.succeeded.emit(asyncio.run(self._operation()))
            except Exception as exc:
                logger.exception("Operação do desktop falhou")
                self.failed.emit(_friendly_error(exc))

    class NaviWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._worker: AsyncCall | None = None
            self.setWindowTitle("NAVI — Assistente Virtual NAF")
            self.resize(980, 720)

            central = QWidget(self)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(24, 24, 24, 24)
            layout.setSpacing(14)

            title = QLabel("NAVI — Assistente Virtual NAF")
            title.setFont(QFont("", 22, QFont.Weight.Bold))
            layout.addWidget(title)

            self.status_label = QLabel()
            self.status_label.setFont(QFont("", 12))
            layout.addWidget(self.status_label)

            self.transcript = QTextEdit()
            self.transcript.setReadOnly(True)
            self.transcript.setFont(QFont("", 14))
            self.transcript.setHtml(
                "<p><b>NAVI:</b> Olá! Como posso ajudar?</p>"
                "<p>Faça uma pergunta tributária sem informar dados pessoais ou credenciais.</p>"
            )
            layout.addWidget(self.transcript, 1)

            input_row = QHBoxLayout()
            self.question = QLineEdit()
            self.question.setPlaceholderText("Digite sua pergunta...")
            self.question.setMinimumHeight(52)
            self.question.setFont(QFont("", 14))
            self.question.returnPressed.connect(self._send)
            input_row.addWidget(self.question, 1)
            self.send_button = QPushButton("Enviar")
            self.send_button.setMinimumSize(120, 52)
            self.send_button.clicked.connect(self._send)
            input_row.addWidget(self.send_button)
            layout.addLayout(input_row)

            actions = QHBoxLayout()
            self.new_button = QPushButton("Nova conversa")
            self.new_button.setMinimumHeight(48)
            self.new_button.clicked.connect(self._new_conversation)
            actions.addWidget(self.new_button)
            actions.addStretch(1)
            status_button = QPushButton("Status")
            status_button.setMinimumHeight(48)
            status_button.clicked.connect(self._show_status)
            actions.addWidget(status_button)
            layout.addLayout(actions)

            self.setCentralWidget(central)
            self._refresh_status()

        def _refresh_status(self) -> None:
            status = controller.status
            if not status.index_ready:
                state = "Base documental não preparada"
            elif not status.model_ready:
                state = "Modelo local não encontrado"
            else:
                state = "Pronta"
            mode = "Local" if status.provider == "local" else "Gemini"
            self.status_label.setText(f"● {state}   |   Modo: {mode}")

        def _set_busy(self, busy: bool) -> None:
            self.question.setEnabled(not busy)
            self.send_button.setEnabled(not busy)
            self.new_button.setEnabled(not busy)
            if busy:
                self.status_label.setText("● NAVI está processando...")
                return
            self.question.setFocus()

        def _start(
            self,
            operation: Callable[[], Coroutine[Any, Any, object]],
            on_success: Callable[[object], None],
        ) -> None:
            self._set_busy(True)
            self._worker = AsyncCall(operation, self)
            self._worker.succeeded.connect(on_success)
            self._worker.failed.connect(self._operation_failed)
            self._worker.finished.connect(lambda: self._set_busy(False))
            self._worker.start()

        def _send(self) -> None:
            question = self.question.text().strip()
            if not question or self._worker is not None and self._worker.isRunning():
                return
            self.question.clear()
            self.transcript.append(f"<p><b>Usuário:</b> {html.escape(question)}</p>")
            self._start(lambda: controller.answer(question), self._answer_ready)

        def _answer_ready(self, response: object) -> None:
            from navi.domain.models import AssistantResponse

            if not isinstance(response, AssistantResponse):
                self._operation_failed("A NAVI retornou uma resposta inesperada.")
                return
            self.transcript.append(f"<p><b>NAVI:</b> {html.escape(response.answer)}</p>")
            if response.sources:
                items = "".join(
                    f"<li>{html.escape(source.label)}</li>" for source in response.sources
                )
                self.transcript.append(f"<p><b>Materiais consultados:</b></p><ul>{items}</ul>")
            self._refresh_status()

        def _operation_failed(self, message: str) -> None:
            self.transcript.append(f"<p><b>NAVI:</b> {html.escape(message)}</p>")
            self._refresh_status()

        def _new_conversation(self) -> None:
            if self._worker is not None and self._worker.isRunning():
                return
            self._start(controller.new_conversation, self._conversation_cleared)

        def _conversation_cleared(self, _: object) -> None:
            self.transcript.setHtml(
                "<p><b>NAVI:</b> Nova conversa iniciada. Como posso ajudar?</p>"
            )
            self._refresh_status()

        def _show_status(self) -> None:
            status = controller.status
            QMessageBox.information(
                self,
                "Status da NAVI",
                "\n".join(
                    (
                        f"PROVIDER: {status.provider}",
                        f"MODELO: {status.model}",
                        f"EMBEDDINGS: {status.embeddings_provider}",
                        f"MODELO DE EMBEDDINGS: {status.embeddings_model}",
                        f"NETWORK REQUIRED: {str(status.network_required).lower()}",
                        f"ÍNDICE PRONTO: {str(status.index_ready).lower()}",
                        f"MODELO LOCAL PRONTO: {str(status.model_ready).lower()}",
                        f"NAVI_DATA_ROOT: {status.data_root}",
                    )
                ),
            )

        def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
            if self._worker is not None and self._worker.isRunning():
                QMessageBox.information(
                    self,
                    "NAVI em uso",
                    "Aguarde a resposta atual antes de fechar o aplicativo.",
                )
                event.ignore()
                return
            event.accept()

    # O Qt não deve interpretar opções do CLI da NAVI, como ``--fullscreen``.
    qt_app = QApplication.instance() or QApplication(["navi"])
    window = NaviWindow()
    if fullscreen:
        window.showFullScreen()
    else:
        window.show()
    try:
        return qt_app.exec()
    finally:
        asyncio.run(container.aclose())
