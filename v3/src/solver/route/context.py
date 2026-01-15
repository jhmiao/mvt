from dataclasses import dataclass
from typing import Dict, List, Tuple
import gurobipy

EventCopy = Tuple[int, int, int]  # (i, d, tau)

@dataclass
class RouteMasterContext:
    event_copies: List[EventCopy]
    y: Dict[EventCopy, "gurobipy.Var"]
    z: Dict[Tuple[int, int, int], "gurobipy.Var"]   # (w,d,k)
    pool: Dict[Tuple[int, int], List["Route"]]      # (w,d)->list of routes
