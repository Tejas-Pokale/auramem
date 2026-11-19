"""Reciprocal rank fusion (PLAN.md section 6, step 4).

Each signal contributes 1 / (k + rank) for every candidate it returned (rank is 1-based).
Raw scores from different indexes are never compared, only ranks.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class Fused[T]:
    key: str
    item: T
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())


def reciprocal_rank_fusion[T](
    rankings: Mapping[str, Sequence[tuple[str, T]]], *, k: int
) -> list[Fused[T]]:
    """Fuse ranked ``(key, item)`` lists by signal name. Ties break on best rank, then key."""
    fused: dict[str, Fused[T]] = {}
    for signal, ranked in rankings.items():
        rank = 0
        for key, item in ranked:
            entry = fused.setdefault(key, Fused(key=key, item=item))
            if signal in entry.ranks:
                continue
            rank += 1
            entry.ranks[signal] = rank
            entry.score += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda f: (-f.score, f.best_rank, f.key))
