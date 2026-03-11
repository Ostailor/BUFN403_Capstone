from __future__ import annotations

from .config import THEME_KEYWORDS


def tag_themes(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for tag, keywords in THEME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.append(tag)
    return tags
