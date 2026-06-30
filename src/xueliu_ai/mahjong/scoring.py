from __future__ import annotations


def fan_to_base_score(fan: int) -> int:
    if fan <= 0:
        return 0
    return 2 ** max(0, fan - 1)
