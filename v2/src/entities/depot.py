# src/entities/depot.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
import datetime as dt

@dataclass(frozen=True)
class DepotSpec:
    id: int                           # unique id, you’ll map this to a node index
    loc: Optional[Union[Tuple[float, float], int]] = None    # coordinates or loc_id (if using postal codes)
    name: str
    time_window: Optional[Dict[int, Tuple[int, int]]] = None  # day -> (start, end)

@dataclass
class DepotState:
    """Optional runtime bookkeeping: who/when visited (per day)."""
    # For quick stats (not required for solving):
    # visits[(day, nurse_id)] = list of visit (arrival_dt, depart_dt)
    visits: dict = None

    def __post_init__(self):
        if self.visits is None:
            self.visits = {}
