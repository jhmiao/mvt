# src/solver/extract.py
from typing import Dict, Tuple, Iterable, List, Any
import numpy as np
from ..models.solution import Solution, Route

# def _normalize_active_x(active_x: Any) -> Iterable[Tuple[int,int,int,int]]:
#     """
#     Yield (i, j, d, w) for each active arc.
#     Supports:
#       - dict {(i,j,d,w): 0/1}
#       - list/tuple of (i,j,d,w[, ...])
#       - ndarray shape (m+K, m+K, day, n) with 0/1
#     """
#     if isinstance(active_x, dict):
#         for (i,j,d,w), val in active_x.items():
#             if val: yield int(i), int(j), int(d), int(w)
#         return
#     if isinstance(active_x, (list, tuple)) and active_x and isinstance(active_x[0], (list, tuple)):
#         for t in active_x:
#             i, j, d, w = t[:4]
#             yield int(i), int(j), int(d), int(w)
#         return
#     if isinstance(active_x, np.ndarray):
#         idxs = np.argwhere(active_x == 1)
#         for i,j,d,w in idxs:
#             yield int(i), int(j), int(d), int(w)
#         return
#     raise TypeError("Unsupported active_x format")


def routes_from_active_x_t(active_x: Any, active_t: Any, pd) -> Solution:
    """
    Build ordered routes per (day, nurse) from active arcs and attach per-node
    start, arrival, and depart arrays.

    Node indexing in pd:
      events: 0..m-1
      HOME:   m            # shared home in raw arcs (will map to unique home 100+w)
      DEPOTs: m+1, m+2     # if you use them
    """
    m, n, D = pd.m, pd.n, pd.day
    HOME_SHARED = m
    DEPOT_AM, DEPOT_PM = m + 1, m + 2

    # pick the service duration source (C_dur)
    if hasattr(pd, "C_dur"):
        def service_dur(node: int) -> int:
            # 0 for homes/depots (defensive)
            if node >= m: return 0
            return pd.C_dur[node]
    else:
        def service_dur(node: int) -> int:
            if node >= m: return 0
            return pd.service_times[node]

    # helper to read t[i,d] regardless of container type
    def get_t(i: int, d: int, default: int = 0):
        if isinstance(active_t, dict):
            return active_t.get((i, d), default)
        try:
            return active_t[i, d]
        except Exception:
            return default

    # Build successor maps per (d,w)
    succ_maps: Dict[Tuple[int, int], Dict[int, int]] = {}
    for i, j, d, w in active_x:
        succ_maps.setdefault((d, w), {})[i] = j

    sol = Solution(day_routes={})
    for d in range(D):
        for w in range(n):
            s_map = succ_maps.get((d, w), {})

            nodes: List[int] = [HOME_SHARED]
            starts: List[int] = [0]  # first home start will be set to 0 anyway
            current = HOME_SHARED
            guard = 0
            max_hops = len(s_map) + 2

            while True:
                nxt = s_map.get(current)

                if nxt is None:
                    # Chain broken: close the route by returning home
                    if nodes[-1] != HOME_SHARED:
                        nodes.append(HOME_SHARED)
                        starts.append(0)  # will be overwritten to 1440 below
                    break

                nodes.append(nxt)
                if 0 <= nxt < m:
                    # event node: use t[i,d] for start
                    starts.append(get_t(nxt, d, default=0))
                else:
                    # home/depot: start placeholder (0 now; will set last home to 1440)
                    starts.append(0)

                guard += 1

                if nxt == HOME_SHARED:
                    break

                current = nxt
                if guard > max_hops:
                    if nodes[-1] != HOME_SHARED:
                        nodes.append(HOME_SHARED)
                        starts.append(0)
                    break

            # Convert shared HOME to unique home per nurse
            unique_home = 100 + w
            nodes = [unique_home if v == HOME_SHARED else v for v in nodes]

            # --- arrival & depart ---
            # arrival := start (per your instruction "equal to service" -> service start)
            arrival = list(starts)

            # depart := start + C_dur[k] for event nodes; == start for home/depots
            depart: List[int] = []
            for node, st in zip(nodes, starts):
                if 0 <= node < m:            # event node
                    depart.append(st + service_dur(node))
                else:                         # home or depot
                    depart.append(st)

            # Enforce day endpoints:
            # first home -> 0; last home -> 1440 for arrival/start/depart
            if len(nodes) == 1:
                arrival[0] = 0
                starts[0]  = 0
                depart[0]  = 0

            elif nodes:
                # first is home by construction
                arrival[0] = 0
                starts[0]  = 0
                depart[0]  = 0

                arrival[-1] = 1440
                starts[-1]  = 1440
                depart[-1]  = 1440

            # if second-to-last is depot_PM, set last start/depart to 1440
            if len(nodes) > 2 and nodes[-2] == DEPOT_PM:
                arrival[-2] = 1440
                starts[-2] = 1440
                depart[-2] = 1440

            # Build route and attach arrays
            r = Route(day_idx=d, nurse=w, nodes=nodes)
            setattr(r, "start", starts)
            setattr(r, "arrival", arrival)
            setattr(r, "depart", depart)

            sol.day_routes[(d, w)] = r

    return sol



def routes_from_active_x(active_x: Any, pd) -> Solution:
    """
    Build ordered routes per (day, nurse) from active arcs.
    Assumes node indexing in pd:
      events: 0..m-1
      HOME:   m
      DEPOTs: e.g., m+1 (AM), m+2 (PM)  — pick your policy terminus
    """
    m, n, D = pd.m, pd.n, pd.day
    HOME = m
    # end route at home

    # Build successor maps per (d,w)
    succ_maps: Dict[Tuple[int,int], Dict[int,int]] = {}
    for i,j,d,w in active_x:
        succ_maps.setdefault((d,w), {})[i] = j

    sol = Solution(day_routes={})
    for d in range(D):
        for w in range(n):
            s_map = succ_maps.get((d, w), {})

            nodes: List[int] = [HOME]   # start at home
            current = HOME
            guard = 0
            max_hops = len(s_map) + 2   # simple cycle guard

            while True:
                nxt = s_map.get(current)

                if nxt is None:
                    # Chain broken: close the route by returning home
                    if nodes[-1] != HOME:
                        nodes.append(HOME)
                    break

                nodes.append(nxt)
                guard += 1

                # Route ends when we get back home
                if nxt == HOME:
                    break

                # Follow the chain
                current = nxt

                # Safety: if something’s off (cycle not returning to home), close route
                if guard > max_hops:
                    if nodes[-1] != HOME:
                        nodes.append(HOME)
                    break
            # replace any node value = m with 100 + w
            nodes = [n if n != m  else 100 + w for n in nodes]
            sol.day_routes[(d, w)] = Route(day_idx=d, nurse=w, nodes=nodes)


    return sol
