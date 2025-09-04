from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import datetime as dt

@dataclass(frozen=True)
class EventSpec:
    id: int
    loc: Optional[Union[Tuple[float, float], int]] = None    # coordinates or loc_id
    req: Dict[str, int]                            # {"RN": k_rn, "LVN": k_lvn}
    duration: int                                  # in minutes
    time_window: Dict[int, Tuple[int, int]]         # day -> (start_minute, end_minute)

@dataclass
class EventPlan:
    day: Optional[int] = None  # day index, e.g., 0 for first day
    start_time: Optional[int] = None  # minutes from reference (e.g., midnight)
    end_time: Optional[int] = None    # minutes from reference (e.g., midnight)
    nurses: Dict[str, List[int]] = field(default_factory=lambda: {"RN": [], "LVN": []})
    earliest_feasible: Optional[int] = None  # minutes from reference
    latest_feasible: Optional[int] = None    # minutes from reference

@dataclass
class Event:
    spec: EventSpec
    plan: EventPlan = field(default_factory=EventPlan)
