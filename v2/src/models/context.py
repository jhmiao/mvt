# src/models/context.py
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Iterable, Set

@dataclass
class Violation:
    kind: str              # "start_unique", "time_window", "travel", "staffing", "depot_cover"
    day: int
    event: Optional[int] = None
    nurse: Optional[int] = None
    msg: str = ""

@dataclass
class FeasReport:
    ok: bool
    violations: List[Violation] = field(default_factory=list)

class Context:
    def __init__(self, pd):
        """
        Expected on `pd`:
          - m (int): number of events, indexed 0..m-1
          - day (int): number of days D
          - dist_matrix[u, v] (int)
          - C_dur[j] or service_times[j] (int)   # service duration at event j (0 for depot/home)
          - time_window[j] -> (earliest, latest)
          - nurse_type[w] in {"RN","LVN"} (or ints you map to those)
          - req_RN[j], req_LVN[j]  # required per event (optional; default 0 if missing)
        Node IDs:
          - events: 0..m-1
          - HOME:   m   (raw; you map to 100+w in routes)
          - DEPOT_AM: m+1
          - DEPOT_PM: m+2
        """
        self.pd = pd
        self.m = pd.m
        self.D = pd.day
        self.dist = pd.C_event
        self.C_dur = getattr(pd, "C_dur", None)
        self.time_window = pd.time_window  # shape (m, day, 2)
        self.nurse_type = pd.nurse_type  # list/array length n or dict {w: "RN"/"LVN"}
        # Set req_RN and req_LVN from min_nurse if available and has two columns
        if hasattr(pd, "min_nurse") and pd.min_nurse is not None and len(pd.min_nurse.shape) == 2 and pd.min_nurse.shape[1] == 2:
            self.req_RN = pd.min_nurse[:, 0]
            self.req_LVN = pd.min_nurse[:, 1]
        else:
            self.req_RN = getattr(pd, "req_RN", None)
            self.req_LVN = getattr(pd, "req_LVN", None)
        self.DEPOT_AM = pd.m + 1
        self.DEPOT_PM = pd.m + 2

    # ---- helper(s) ----
    def _service_dur(self, node: int) -> int:
        # service is only for event nodes; 0 for depots/homes
        if 0 <= node < self.m and self.C_dur is not None:
            return self.C_dur[node]
        return 0

    def check_solution(self, sol) -> FeasReport:
        vios: List[Violation] = []

        # Maps for cross-route/event checks per day:
        # (d, j) -> canonical start time (int)
        start_time_by_event: Dict[Tuple[int, int], int] = {}
        # (d, j) -> set of nurses attending
        attendees_by_event: Dict[Tuple[int, int], Set[int]] = {}
        # per day: which nurses’ routes contain AM/PM depots
        routes_have_AM: Dict[int, Set[int]] = {d: set() for d in range(self.D)}
        routes_have_PM: Dict[int, Set[int]] = {d: set() for d in range(self.D)}

        # Pass 1: per-route local checks, collect global info
        for r in sol.iter_routes():
            nodes = r.nodes
            starts = getattr(r, "start", None)
            departs = getattr(r, "depart", None)

            if starts is None or departs is None or len(starts) != len(nodes) or len(departs) != len(nodes):
                vios.append(Violation(
                    kind="data",
                    day=r.day_idx,
                    nurse=r.nurse,
                    msg="Route missing aligned start/depart arrays"
                ))
                continue

            d = r.day_idx
            w = r.nurse

            # Mark depot presence for this route
            if self.DEPOT_AM in nodes:
                routes_have_AM[d].add(w)
            if self.DEPOT_PM in nodes:
                routes_have_PM[d].add(w)

            # check time windows & travel feasibility along this route
            for k in range(len(nodes)):
                node = nodes[k]
                st = starts[k]

                # 1) Time windows for event nodes only
                if 0 <= node < self.m:
                    # d is the day index for this route
                    e, L = self.time_window[node, d, 0], self.time_window[node, d, 1]
                    if st < e or st > L:
                        vios.append(Violation(
                            kind="time_window", day=d, event=node, nurse=w,
                            msg=f"Start {st} outside TW [{e},{L}] at pos {k}"
                        ))

                # 2) Travel feasibility between consecutive event nodes:
                if k < len(nodes) - 1 and 0 <= node < self.m:
                    u, v = node, nodes[k+1]
                    if 0 <= v < self.m:
                        travel = self.dist[u, v]
                        dep_u = departs[k]
                        st_v = starts[k+1]
                        if dep_u + travel > st_v:
                            vios.append(Violation(
                                kind="travel", day=d, nurse=w,
                                msg=f"Dep({u})={dep_u} + dist={travel} > Start({v})={st_v} at edge ({u}->{v})"
                            ))

            # Collect start times and staffing from this route
            for k, node in enumerate(nodes):
                if 0 <= node < self.m:
                    key = (d, node)
                    st = starts[k]
                    # (1) unique/consistent start time across all attendees
                    if key in start_time_by_event and start_time_by_event[key] != st:
                        vios.append(Violation(
                            kind="start_unique", day=d, event=node, nurse=w,
                            msg=f"Inconsistent start times for event {node}: seen {start_time_by_event[key]} vs {st}"
                        ))
                    else:
                        start_time_by_event.setdefault(key, st)

                    # track attendees
                    attendees_by_event.setdefault(key, set()).add(w)

        # Pass 2: staffing counts and AM/PM coverage per event
        for (d, j), nurses in attendees_by_event.items():
            # (4) staffing requirements
            if self.req_RN is not None:
                need_RN = self.req_RN[j]
            else:
                need_RN = 0
            if self.req_LVN is not None:
                need_LVN = self.req_LVN[j]
            else:
                need_LVN = 0

            have_RN = sum(1 for w in nurses if self.nurse_type[w] == "RN")
            have_LVN = sum(1 for w in nurses if self.nurse_type[w] == "LVN")

            if have_RN < need_RN or have_LVN < need_LVN:
                vios.append(Violation(
                    kind="staffing", day=d, event=j,
                    msg=f"Need RN>= {need_RN}, LVN>= {need_LVN}; have RN={have_RN}, LVN={have_LVN}"
                ))

            # AM/PM depot coverage among attendees of this event (same day)
            # among nurses going to event j on day d,
            # at least one has DEPOT_AM in their route and at least one has DEPOT_PM.
            has_AM = any(w in routes_have_AM[d] for w in nurses)
            has_PM = any(w in routes_have_PM[d] for w in nurses)
            if not has_AM or not has_PM:
                vios.append(Violation(
                    kind="depot_cover", day=d, event=j,
                    msg=f"Event {j}: requires attendees to include AM and PM depot visitors; has_AM={has_AM}, has_PM={has_PM}"
                ))

        return FeasReport(ok=(len(vios) == 0), violations=vios)
