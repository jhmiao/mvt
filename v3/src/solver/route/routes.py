from dataclasses import dataclass
from typing import Tuple, Set

EventCopy = Tuple[int, int, int]  # (i, d, tau)

@dataclass(frozen=True)
class Route:
    w: int
    d: int
    visits: Tuple[EventCopy, ...]      # ordered
    cost: float
    work: int
    depot_ok: int
    # optional extension:
    travel: int | float = 0
    waiting: int = 0
    # fairness tags, etc.

    # Do NOT store a dense A-matrix here.
    @property
    def covered(self) -> frozenset[EventCopy]:
        return frozenset(self.visits)

