# src/models/solution.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Iterable, Optional
import datetime as dt
import copy

@dataclass
class Route:
    """One nurse's tour on one day."""
    day_idx: int                # 0..D-1
    nurse: int                  # 0..n-1
    nodes: List[int]            # ordered node indices, e.g., [HOME, ..., HOME]

    arrival: List[int] = field(default_factory=list)
    start:   List[int] = field(default_factory=list)
    depart:  List[int] = field(default_factory=list)

    # first index that needs recompute; None = clean
    dirty_from: Optional[int] = None

    def copy(self) -> "Route":
        return Route(
            self.day_idx, self.nurse, list(self.nodes),
            list(self.arrival), list(self.start), list(self.depart),
            self.dirty_from
        )

    def __len__(self) -> int:
        return len(self.nodes)

    def segment(self, i: int, j: int) -> List[int]:
        """Return nodes[i:j] (Python slicing semantics)."""
        return self.nodes[i:j]

@dataclass
class Solution:
    """All routes across the horizon."""
    # key = (day_idx, nurse_id)
    day_routes: Dict[Tuple[int, int], "Route"] = field(default_factory=dict)
    # optional mapping if you also carry calendar dates
    day_list: Optional[List[dt.date]] = None

    # -------- core mutators --------
    def set_nodes(self, d: int, w: int, nodes: List[int]) -> None:
        self.day_routes[(d, w)] = Route(day_idx=d, nurse=w, nodes=list(nodes))
        # if Route has mark_dirty, call it (safe even if not present)
        if hasattr(self.day_routes[(d, w)], "mark_dirty"):
            self.day_routes[(d, w)].mark_dirty(0)

    def replace_subpath(self, d: int, w: int, start: int, end: int, new_subpath: List[int]) -> None:
        r = self.day_routes[(d, w)]
        r.nodes = r.nodes[:start] + list(new_subpath) + r.nodes[end:]
        if hasattr(r, "mark_dirty"):
            r.mark_dirty(start)

    def swap_positions(self, d: int, w: int, i: int, j: int) -> None:
        r = self.day_routes[(d, w)]
        r.nodes[i], r.nodes[j] = r.nodes[j], r.nodes[i]
        if hasattr(r, "mark_dirty"):
            r.mark_dirty(min(i, j))

    # -------- accessors --------
    def get_route(self, d: int, w: int) -> Optional["Route"]:
        return self.day_routes.get((d, w))

    def get_nodes(self, d: int, w: int) -> List[int]:
        r = self.day_routes.get((d, w))
        return r.nodes if r else []

    def routes_for_day(self, d: int) -> Iterable["Route"]:
        """Iterate routes for a specific day."""
        for (day_idx, _w), r in self.day_routes.items():
            if day_idx == d:
                yield r

    def iter_routes(self, day: Optional[int] = None, nurse: Optional[int] = None) -> Iterable["Route"]:
        """
        Iterate all routes, optionally filtered by day and/or nurse.
        Usage:
            for r in sol.iter_routes(): ...
            for r in sol.iter_routes(day=0): ...
            for r in sol.iter_routes(nurse=3): ...
            for r in sol.iter_routes(day=1, nurse=2): ...
        """
        for (d, w), r in self.day_routes.items():
            if day is not None and d != day:
                continue
            if nurse is not None and w != nurse:
                continue
            yield r

