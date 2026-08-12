"""Canvas dimensions derived from the selected avatar image."""

from __future__ import annotations


def dimensions_for_avatar(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Avatar dimensions must be positive")
    ratio = width / height
    if ratio <= 0.70:
        return (1080, 1920)
    if ratio >= 1.55:
        return (1920, 1080)
    target_height = 1080
    target_width = int(round(target_height * ratio / 2) * 2)
    return (target_width, target_height)
