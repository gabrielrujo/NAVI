from navi.channels.telegram.formatting import split_telegram_text


def test_split_telegram_text_respects_limit_and_preserves_content() -> None:
    text = " ".join(["tributo"] * 100)
    parts = split_telegram_text(text, limit=80)

    assert all(len(part) <= 80 for part in parts)
    assert " ".join(parts) == text

