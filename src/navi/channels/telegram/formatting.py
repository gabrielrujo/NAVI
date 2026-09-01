from __future__ import annotations

from navi.domain.models import AssistantResponse


def format_response(response: AssistantResponse) -> str:
    if not response.sources:
        return response.answer
    source_lines = "\n".join(f"• {source.label}" for source in response.sources)
    return f"{response.answer}\n\nFontes consultadas:\n{source_lines}"


def split_telegram_text(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return parts

