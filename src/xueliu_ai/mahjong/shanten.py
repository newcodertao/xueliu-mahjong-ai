from __future__ import annotations

from functools import lru_cache

from xueliu_ai.mahjong.tiles import tiles_to_counts


def normal_shanten(tiles: list[str]) -> int:
    counts = tuple(tiles_to_counts(tiles))
    return _normal_shanten_counts(counts)


def seven_pairs_shanten(tiles: list[str]) -> int:
    counts = tiles_to_counts(tiles)
    pairs = sum(1 for count in counts if count >= 2)
    unique = sum(1 for count in counts if count > 0)
    need_unique = max(0, 7 - unique)
    return max(0, 6 - pairs + need_unique)


def best_shanten(tiles: list[str]) -> int:
    return min(normal_shanten(tiles), seven_pairs_shanten(tiles))


@lru_cache(maxsize=200_000)
def _normal_shanten_counts(counts: tuple[int, ...]) -> int:
    best = 8

    def walk(state: list[int], start: int, melds: int, taatsu: int, pair: int) -> None:
        nonlocal best
        while start < 27 and state[start] == 0:
            start += 1
        if start >= 27:
            usable_taatsu = min(taatsu, 4 - melds)
            value = 8 - melds * 2 - usable_taatsu - pair
            best = min(best, value)
            return

        if state[start] >= 3:
            state[start] -= 3
            walk(state, start, melds + 1, taatsu, pair)
            state[start] += 3

        suit_offset = (start // 9) * 9
        rank = start % 9
        if rank <= 6 and state[start + 1] > 0 and state[start + 2] > 0:
            state[start] -= 1
            state[start + 1] -= 1
            state[start + 2] -= 1
            walk(state, start, melds + 1, taatsu, pair)
            state[start] += 1
            state[start + 1] += 1
            state[start + 2] += 1

        if pair == 0 and state[start] >= 2:
            state[start] -= 2
            walk(state, start, melds, taatsu, 1)
            state[start] += 2

        if state[start] >= 2:
            state[start] -= 2
            walk(state, start, melds, taatsu + 1, pair)
            state[start] += 2

        if rank <= 7 and start + 1 < suit_offset + 9 and state[start + 1] > 0:
            state[start] -= 1
            state[start + 1] -= 1
            walk(state, start, melds, taatsu + 1, pair)
            state[start] += 1
            state[start + 1] += 1

        if rank <= 6 and start + 2 < suit_offset + 9 and state[start + 2] > 0:
            state[start] -= 1
            state[start + 2] -= 1
            walk(state, start, melds, taatsu + 1, pair)
            state[start] += 1
            state[start + 2] += 1

        state[start] -= 1
        walk(state, start, melds, taatsu, pair)
        state[start] += 1

    walk(list(counts), 0, 0, 0, 0)
    return best


def is_complete_hand(tiles: list[str]) -> bool:
    if len(tiles) % 3 != 2:
        return False
    return best_shanten(tiles) == -1 or _is_standard_complete(tuple(tiles_to_counts(tiles))) or _is_seven_pairs_complete(tiles)


def _is_seven_pairs_complete(tiles: list[str]) -> bool:
    counts = tiles_to_counts(tiles)
    return len(tiles) == 14 and sum(1 for count in counts if count == 2) == 7


@lru_cache(maxsize=100_000)
def _is_standard_complete(counts: tuple[int, ...]) -> bool:
    total = sum(counts)
    if total == 0:
        return True
    if total % 3 == 2:
        for i, count in enumerate(counts):
            if count >= 2:
                reduced = list(counts)
                reduced[i] -= 2
                if _can_form_melds(tuple(reduced)):
                    return True
        return False
    return _can_form_melds(counts)


@lru_cache(maxsize=100_000)
def _can_form_melds(counts: tuple[int, ...]) -> bool:
    try:
        i = next(index for index, count in enumerate(counts) if count)
    except StopIteration:
        return True

    state = list(counts)
    if state[i] >= 3:
        state[i] -= 3
        if _can_form_melds(tuple(state)):
            return True
        state[i] += 3

    rank = i % 9
    if rank <= 6 and i // 9 == (i + 2) // 9 and state[i + 1] and state[i + 2]:
        state[i] -= 1
        state[i + 1] -= 1
        state[i + 2] -= 1
        if _can_form_melds(tuple(state)):
            return True
    return False
