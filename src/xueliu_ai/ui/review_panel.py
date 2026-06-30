from __future__ import annotations


def format_review_summary(summary: dict[str, int]) -> str:
    return "\n".join(f"{event}: {count}" for event, count in sorted(summary.items()))
