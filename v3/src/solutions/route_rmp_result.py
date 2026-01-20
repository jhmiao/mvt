from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

EventCopy = Tuple[int, int, int]  # (event, day, time_slot)


@dataclass(frozen=True)
class EventSchedule:
    event: int
    day: int
    time_slot: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "event": int(self.event),
            "day": int(self.day),
            "time_slot": int(self.time_slot),
        }


@dataclass(frozen=True)
class RouteSelection:
    nurse: int
    day: int
    route_index: int
    visits: Tuple[EventCopy, ...]
    cost: float
    work: int
    depot_ok: int
    travel: Optional[float] = None
    waiting: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nurse": int(self.nurse),
            "day": int(self.day),
            "route_index": int(self.route_index),
            "visits": [
                {"event": int(i), "day": int(d), "time_slot": int(tau)}
                for i, d, tau in self.visits
            ],
            "cost": float(self.cost),
            "work": int(self.work),
            "depot_ok": int(self.depot_ok),
            "travel": None if self.travel is None else float(self.travel),
            "waiting": None if self.waiting is None else int(self.waiting),
        }


@dataclass(frozen=True)
class RouteRmpResult:
    status: int
    runtime: float
    obj_val: Optional[float]
    obj_bound: Optional[float]
    mip_gap: Optional[float]
    node_count: Optional[int]
    event_schedule: List[EventSchedule] = field(default_factory=list)
    selected_routes: List[RouteSelection] = field(default_factory=list)
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": int(self.status),
            "runtime": float(self.runtime),
            "obj_val": self.obj_val,
            "obj_bound": self.obj_bound,
            "mip_gap": self.mip_gap,
            "node_count": self.node_count,
            "event_schedule": [entry.to_dict() for entry in self.event_schedule],
            "selected_routes": [entry.to_dict() for entry in self.selected_routes],
            "extra": self.extra,
        }
