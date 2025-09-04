# solution_io.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Union, Any
import numpy as np
import pickle

# ---------------------------
# 1) Problem parameters
# ---------------------------

@dataclass
class ProblemData:
    # Travel
    C_event: np.ndarray         # (m, m)    event -> event
    C_home: np.ndarray          # (n, m)    home(w) -> event j
    C_depot_e: np.ndarray       # (m,)      event i -> depot
    C_depot_h: np.ndarray       # (n,)      depot -> home(w)  (if used in your model)
    # Service and constraints
    C_dur: np.ndarray           # (m,)      service duration of event i (minutes)
    time_window: np.ndarray     # (m, day, 2) minutes from day anchor
    min_nurse: np.ndarray       # (m, day)  min nurses required (if role-agnostic; otherwise split)
    # Cardinalities
    nr: int
    nl: int
    n: int
    m: int
    day: int

    @property
    def RN_range(self) -> range:
        return range(0, self.nr)

    @property
    def LVN_range(self) -> range:
        return range(self.nr, self.nr + self.nl)

    def role_of(self, w: int) -> str:
        return "RN" if w < self.nr else "LVN"


# ---------------------------
# 2) Solution container
# ---------------------------

@dataclass
class Route:
    day: int
    nurse: int                 # global id in [0, n-1]
    nodes: List[int]           # ordered, includes start 'm' and end 'm'

@dataclass
class Solution:
    # key = (d, w) ; value = ordered node list (must start with m and end with m)
    day_routes: Dict[Tuple[int, int], Route] = field(default_factory=dict)

    def get_route_nodes(self, d: int, w: int) -> List[int]:
        r = self.day_routes.get((d, w))
        return r.nodes if r else []

    def all_routes(self) -> Iterable[Route]:
        return self.day_routes.values()


# ---------------------------
# 3) Travel lookup using ProblemData
# ---------------------------

def arc_cost(pd: ProblemData, w: int, i: int, j: int) -> float:
    """
    Cost of traversing arc (i -> j) for nurse w using the indexing:
      events: 0..m-1 ; home(w): m ; depot: m+1
    """
    m = pd.m
    depot = m + 1

    # Home -> Event or Event -> Home
    if i == m and 0 <= j < m:
        return float(pd.C_home[w, j])
    if 0 <= i < m and j == m:
        return float(pd.C_home[w, i])

    # Event -> Event
    if 0 <= i < m and 0 <= j < m:
        return float(pd.C_event[i, j])

    # Event -> Depot or Depot -> Event
    if 0 <= i < m and j == depot:
        return float(pd.C_depot_e[i])
    if i == depot and 0 <= j < m:
        return float(pd.C_depot_e[j])

    # Home -> Depot or Depot -> Home
    if i == m and j == depot:
        return float(pd.C_depot_h[w])
    if i == depot and j == m:
        return float(pd.C_depot_h[w])

    # Any other arc, return 0 for safety
    return 0.0


def route_travel(pd: ProblemData, w: int, nodes: List[int]) -> float:
    if not nodes or len(nodes) <= 2:
        return 0.0
    total = 0.0
    for a, b in zip(nodes[:-1], nodes[1:]):
        cost = arc_cost(pd, w, a, b)
        print(f"Nurse {w}: {a} -> {b}, cost = {cost}")
        total += cost
    return total


# ---------------------------
# 4) Read feasible solution from pickle and rebuild routes
# ---------------------------

def _extract_active_arcs(active_x: Any) -> Iterable[Tuple[int, int, int, int]]:
    """
    Normalize active arcs to a stream of tuples (i, j, d, w) with value==1.
    Handles a few common shapes:
      - dict with tuple keys (i,j,d,w) -> 0/1
      - list/array of 4-tuples (i,j,d,w)
      - dense numpy array x with shape (m+2, m+2, day, n)
    """
    # Case A: dict-like {(i,j,d,w): 1}
    if isinstance(active_x, dict):
        for key, val in active_x.items():
            if not val:
                continue
            if len(key) != 4:
                raise ValueError("active_x dict keys must be (i,j,d,w)")
            i, j, d, w = key
            yield (int(i), int(j), int(d), int(w))
        return

    # Case B: list/tuple of tuples
    if isinstance(active_x, (list, tuple)) and active_x and isinstance(active_x[0], (list, tuple)):
        for t in active_x:
            if len(t) < 4:
                raise ValueError("active_x tuples must be at least (i,j,d,w)")
            i, j, d, w = t[:4]
            yield (int(i), int(j), int(d), int(w))
        return

    # Case C: dense numpy array
    if isinstance(active_x, np.ndarray):
        if active_x.ndim != 4:
            raise ValueError("active_x ndarray must have shape (m+2, m+2, day, n)")
        idxs = np.argwhere(active_x == 1)
        # idxs rows are [i,j,d,w]
        for i, j, d, w in idxs:
            yield (int(i), int(j), int(d), int(w))
        return

    raise TypeError("Unrecognized active_x format")


def _build_routes_from_arcs(arcs: Iterable[Tuple[int, int, int, int]],
                            m: int, day: int, n: int) -> Dict[Tuple[int, int], Route]:
    """
    Build ordered routes per (d, w) from successor maps.
    Each route must start at node 'm' (home of w) and end at node 'm' (home of w).
    If no events are scheduled, the route is [m, m].
    """
    # successor[(d,w)][i] = j
    succ: Dict[Tuple[int, int], Dict[int, int]] = {}
    for i, j, d, w in arcs:
        succ.setdefault((d, w), {})[i] = j

    routes: Dict[Tuple[int, int], Route] = {}

    for d in range(day):
        for w in range(n):
            key = (d, w)
            s_map = succ.get(key, {})
            nodes = [m]
            current = m
            guard = 0
            max_hops = len(s_map) + 2  # basic cycle guard

            # If there are no scheduled events, route is [m, m]
            if not s_map:
                nodes.append(m)
                routes[key] = Route(day=d, nurse=w, nodes=nodes)
                continue

            # Otherwise, follow the chain until you return to home (m)
            while guard <= max_hops:
                if current not in s_map:
                    # If the chain is broken, return home
                    nodes.append(m)
                    break
                nxt = s_map[current]
                nodes.append(nxt)
                current = nxt
                guard += 1
                if current == m:
                    break

            # Ensure the route ends at home (m)
            if nodes[-1] != m:
                nodes.append(m)

            routes[key] = Route(day=d, nurse=w, nodes=nodes)

    return routes


def load_solution_from_pickle(pickle_path: str, pd: ProblemData) -> Solution:
    """
    Reads a feasible solution pickle (with summary['active_x']) and returns a Solution with routes.
    Expects x(i,j,d,w) using indices:
      events: 0..m-1 ; home(w): m ; depot: m+1
    """
    with open(pickle_path, "rb") as f:
        summary = pickle.load(f)

    if "active_x" not in summary:
        raise KeyError("Pickle missing key 'active_x'")

    arcs = list(_extract_active_arcs(summary["active_x"]))
    routes = _build_routes_from_arcs(arcs, m=pd.m, day=pd.day, n=pd.n)
    return Solution(day_routes=routes)


# ---------------------------
# 5) (Optional) Example utility to compute total travel of the solution
# ---------------------------

def total_travel_minutes(pd: ProblemData, sol: Solution) -> float:
    tot = 0.0
    for (d, w), route in sol.day_routes.items():
        tot += route_travel(pd, w, route.nodes)
    return tot
