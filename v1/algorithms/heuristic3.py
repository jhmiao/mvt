from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import datetime as dt
import pickle

class Event:
    def __init__(self, 
                 id: int,
                 location: tuple[float, float],
                 duration: dt.timedelta,
                 time_window: tuple[dt.datetime, dt.datetime],
                 nurse_ids: List[int] = None,
                 start_time: dt.datetime = None,
                 end_time: dt.datetime = None,
                 day: dt.date = None):
        self.id = id
        self.location = location
        self.duration = duration
        self.time_window = time_window
        self.nurse_ids = nurse_ids or []
        self.start_time = start_time
        self.end_time = end_time
        self.day = day


class Assignment:
    event_id: int
    nurse_id: int
    day: dt.date
    start_time: dt.datetime
    end_time: dt.datetime

    def __init__(self, event_id: int, nurse_id: int, 
                 day: dt.date, start_time: dt.datetime, end_time: dt.datetime):
        self.event_id = event_id
        self.nurse_id = nurse_id
        self.day = day
        self.start_time = start_time
        self.end_time = end_time


class Nurse:
    def __init__(self, id: int, name: str = None):
        self.id = id
        self.name = name
        self.events: List[Event] = []

@dataclass
class Schedule:
    def __init__(self):
        self.events: Dict[int, Event] = {}                         # event_id -> Event
        self.nurses: Dict[int, Nurse] = {}                         # nurse_id -> Nurse
    
    @classmethod
    def from_pickle_summary(cls, filename: str) -> "Schedule":
        """
        Load a schedule summary from pickle file and rebuild
        Nurses and Events.
        """
        with open(filename, "rb") as f:
            summary = pickle.load(f)

        # Expecting summary to look like:
        # {
        #   "nurses": {id: {"name": ..., "events": [event_ids]}},
        #   "events": {id: {"location": ..., "duration": ..., "time_window": ..., "nurse_ids": [...], "start_time": ..., "end_time": ..., "day": ...}}
        # }
        
        sched = cls()

        # Rebuild events
        for eid, edata in summary.get("events", {}).items():
            event = Event(
                id=eid,
                location=edata["location"],
                duration=edata["duration"],
                time_window=edata["time_window"],
                nurse_ids=edata.get("nurse_ids", []),
                start_time=edata.get("start_time"),
                end_time=edata.get("end_time"),
                day=edata.get("day"),
            )
            sched.events[eid] = event

        # Rebuild nurses
        for nid, ndata in summary.get("nurses", {}).items():
            nurse = Nurse(id=nid, name=ndata.get("name"))
            for eid in ndata.get("events", []):
                if eid in sched.events:
                    nurse.events.append(sched.events[eid])
            sched.nurses[nid] = nurse

        return sched

    # -------- convenience API --------
    def get_route(self, day: dt.date, nurse_id: int) -> List[int]:
        return self.day_routes.get((day, nurse_id), [])

    def set_route(self, day: dt.date, nurse_id: int, route: List[int]) -> None:
        self.day_routes[(day, nurse_id)] = route

    def route_travel_time(self, day: dt.date, nurse_id: int,
                          with_home: bool = True) -> float:
        """Sum travel for the route order (add home/depot legs if modeled)."""
        r = self.get_route(day, nurse_id)
        if not r: return 0.0
        legs = []
        # If you model nurse-specific homes, call a dist_fn(nurse_id, from_node, to_node)
        for a, b in zip(r[:-1], r[1:]):
            legs.append(self.travel[(a, b)])
        return float(sum(legs))

    def recompute_times_forward(self, day: dt.date, nurse_id: int) -> bool:
        """Earliest-feasible pass for start/end times; return feasibility."""
        # Implement: propagate travel + service, enforce time windows & daily cap
        return True

    def two_opt_improve(self, day: dt.date, nurse_id: int) -> bool:
        """Run 2-opt on a single route, accept improving feasible swaps."""
        # Implement: local delta calc + recompute_times_forward from swap point
        return True

    def best_position_reinsert(self, day: dt.date, nurse_id: int) -> bool:
        """Try removing one event and reinserting at best position (feasible)."""
        return True

    def total_travel(self) -> float:
        return sum(self.route_travel_time(day, nid)
                   for (day, nid) in self.day_routes)

    # -------- I/O bridges to your pickle summary --------
    @classmethod
    def from_pickle_summary(cls, summary_dict, events, nurses, travel) -> "Schedule":
        sch = cls(events=events, nurses=nurses, travel=travel)
        # Reconstruct day_routes from summary["active_x"] (x[i,j,d,w]=1 arcs)
        # Build ordered routes per (day, nurse) by following successors
        return sch

    def to_pickle_summary(self) -> dict:
        # Convert day_routes back to active_x arcs for saving
        return {"active_x": ...}


