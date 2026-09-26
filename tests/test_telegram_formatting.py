from navi.channels.telegram.formatting import format_response, split_telegram_text
from navi.domain.models import AssistantResponse, SourceReference


def test_format_response_shows_consulted_materials_and_pages() -> None:
    response = AssistantResponse(
        answer="Resposta objetiva.",
        sources=(
            SourceReference(name="Tributos.pdf", page=5),
            SourceReference(name="Cidadania Fiscal.pdf"),
        ),
    )

    assert format_response(response) == (
        "Resposta objetiva.\n\n"
        "Materiais consultados:\n"
        "• Tributos.pdf, p. 5\n"
        "• Cidadania Fiscal.pdf"
    )


def test_format_response_without_sources_preserves_answer() -> None:
    response = AssistantResponse(answer="Não encontrei essa informação.")

    assert format_response(response) == "Não encontrei essa informação."


def test_split_telegram_text_respects_limit_and_preserves_content() -> None:
    text = " ".join(["tributo"] * 100)
    parts = split_telegram_text(text, limit=80)

    assert all(len(part) <= 80 for part in parts)
    assert " ".join(parts) == text
