from navi.channels.telegram.formatting import format_response, split_telegram_text
from navi.domain.models import AssistantResponse, SourceReference


def test_format_response_hides_sources_from_telegram_message() -> None:
    response = AssistantResponse(
        answer="Resposta objetiva.",
        sources=(SourceReference(name="Tributos.pdf", page=5),),
    )

    assert format_response(response) == "Resposta objetiva."


def test_split_telegram_text_respects_limit_and_preserves_content() -> None:
    text = " ".join(["tributo"] * 100)
    parts = split_telegram_text(text, limit=80)

    assert all(len(part) <= 80 for part in parts)
    assert " ".join(parts) == text
