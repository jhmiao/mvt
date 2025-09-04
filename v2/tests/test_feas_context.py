# tests/test_feas_context.py
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import sys, pathlib

# add v2 to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


# from src.models.solution import Route, Solution
from src.solver.extract import routes_from_active_x_t
from src.models.context import Context

# For this demo, we’ll assume they are in scope.
# If not, adjust the imports above.

# ---------- Minimal ProblemData for testing ----------
class DistWrapper:
    """
    Wraps a base distance matrix indexed on [0..m+2] (events + HOME_SHARED + depots),
    but also supports unique homes 100+w by mapping them to HOME_SHARED (= m).
    """
    def __init__(self, base: np.ndarray, m: int):
        self.base = base
        self.m = m                # HOME_SHARED index
        self.DEPOT_AM = m + 1
        self.DEPOT_PM = m + 2

    def __getitem__(self, key: Tuple[int, int]) -> int:
        u, v = key
        u = self._map(u)
        v = self._map(v)
        return int(self.base[u, v])

    def _map(self, node: int) -> int:
        # unique home -> shared home
        if node >= 100:
            return self.m
        return node

@dataclass
class PD:
    # core sizes
    m: int     # number of events
    n: int     # nurses
    day: int   # days

    # core data
    C_event: any
    C_dur: List[int]
    time_windows: Dict[int, Tuple[int, int]]
    nurse_type: Dict[int, str]
    req_RN: List[int]
    req_LVN: List[int]

# ---------- Build a tiny feasible instance ----------
def make_problem():
    m = 2              # events: 0, 1
    n = 2              # nurses: 0(RN), 1(LVN)
    D = 1              # one day
    HOME_SHARED = m
    DEPOT_AM = m + 1
    DEPOT_PM = m + 2
    N = m + 3          # events + home + 2 depots

    # Base distances among events/home/depots
    base = np.zeros((N, N), dtype=int)
    # simple symmetric distances for testing
    for i in range(N):
        for j in range(N):
            base[i, j] = 10 if i != j else 0

    dist = DistWrapper(base, m=HOME_SHARED)

    # service durations for events (0..m-1); depots/homes implicitly 0
    C_dur = [30, 40]

    # time windows (wide around 8:00)
    # shape (event, day, 2)
    tw = np.zeros((m, D, 2), dtype=int)
    for j in range(m):
        for d in range(D):
            tw[j, d, 0] = 8*60
            tw[j, d, 1] = 12*60

    nurse_type = {0: "RN", 1: "LVN"}
    # each event needs one RN and one LVN
    req_RN  = [1, 1]
    req_LVN = [1, 1]

    pd = PD(
        m=m, n=n, day=D,
        C_event=dist,
        C_dur=C_dur,
        time_windows=tw,
        nurse_type=nurse_type,
        req_RN=req_RN,
        req_LVN=req_LVN,
    )
    return pd

# ---------- Build active arcs and times ----------
def make_feasible_active(pd: PD):
    m = pd.m
    d = 0
    # Nodes: m=HOME_SHARED, m+1=DEPOT_AM, m+2=DEPOT_PM
    HOME = m
    AM = m + 1
    PM = m + 2

    # Nurse 0 route: HOME -> AM -> event 0 -> PM -> HOME
    # Nurse 1 route: HOME -> AM -> event 1 -> PM -> HOME
    active_x = [
        (HOME, AM, d, 0), (AM, 0, d, 0), (0, PM, d, 0), (PM, HOME, d, 0),
        (HOME, AM, d, 1), (AM, 1, d, 1), (1, PM, d, 1), (PM, HOME, d, 1),
    ]

    # Start times per (event, day)
    active_t = {
        (0, d): 8*60 + 10,   # 08:10
        (1, d): 8*60 + 20,   # 08:20
    }
    return active_x, active_t

# ---------- Pretty printer ----------
def print_report(report):
    if report.ok:
        print("FEASIBLE ✅")
        return
    print("INFEASIBLE ❌")
    for v in report.violations:
        print(f" - [{v.kind}] day={v.day}"
              f"{'' if v.event is None else f', event={v.event}'}"
              f"{'' if v.nurse is None else f', nurse={v.nurse}'}"
              f" :: {v.msg}")

import numpy as np

def make_three_nurses_two_events():
    """
    Build a toy problem:
      - 2 events (0, 1)
      - 3 nurses (0=RN, 1=LVN, 2=RN)
      - Day horizon = 1
      - Nurse 0 attends both events
      - Nurse 1 attends event 0
      - Nurse 2 attends event 1
    """
    m = 2   # events: 0,1
    n = 3   # nurses: 0,1,2
    D = 1
    HOME = m
    AM = m + 1
    PM = m + 2
    N = m + 3  # total nodes in base dist

    # distance matrix (10 everywhere except 0 diag)
    base = np.full((N, N), 10, dtype=int)
    np.fill_diagonal(base, 0)

    # simple wrapper so unique home IDs map back to shared HOME
    class DistWrapper:
        def __getitem__(self, key):
            u, v = key
            if u >= 100: u = HOME
            if v >= 100: v = HOME
            return int(base[u, v])
    dist = DistWrapper()

    # service durations
    C_dur = [30, 20]   # event 0 needs 30min, event 1 needs 20min

    # time windows (8am–12pm)
    # shape (event, day, 2)
    tw = np.zeros((m, D, 2), dtype=int)
    for j in range(m):
        for d in range(D):
            tw[j, d, 0] = 8*60
            tw[j, d, 1] = 12*60


    nurse_type = {0: "RN", 1: "LVN", 2: "RN"}
    req_RN = [1, 2]   
    req_LVN = [1, 0]

    class PD:
        pass
    pd = PD()
    pd.m, pd.n, pd.day = m, n, D
    pd.C_event = dist
    pd.C_dur = C_dur
    pd.time_windows = tw
    pd.nurse_type = nurse_type
    pd.req_RN = req_RN
    pd.req_LVN = req_LVN

    # -------- active arcs and times --------
    d = 0
    active_x = []
    active_t = {}

    # Nurse 0: HOME -> AM -> event0 -> event1 -> PM -> HOME
    active_x += [
        (HOME, AM, d, 0),
        (AM, 0, d, 0),
        (0, 1, d, 0),
        (1, PM, d, 0),
        (PM, HOME, d, 0),
    ]
    active_t[(0, d)] = 8*60 + 10   # 08:10
    active_t[(1, d)] = 9*60 + 00   # 09:00

    # Nurse 1: HOME -> AM -> event0 -> PM -> HOME
    active_x += [
        (HOME, AM, d, 1),
        (AM, 0, d, 1),
        (0, PM, d, 1),
        (PM, HOME, d, 1),
    ]
    active_t[(0, d)] = 8*60 + 10   # must match nurse 0’s start for event 0

    # Nurse 2: HOME -> AM -> event1 -> PM -> HOME
    active_x += [
        (HOME, AM, d, 2),
        (AM, 1, d, 2),
        (1, PM, d, 2),
        (PM, HOME, d, 2),
    ]
    active_t[(1, d)] = 9*60 + 00   # must match nurse 0’s start for event 1

    return pd, active_x, active_t

# ---------- Runner ----------
def main():
    pd, active_x, active_t = make_three_nurses_two_events()
    sol = routes_from_active_x_t(active_x, active_t, pd)

    # check route and its times
    # for r in sol.iter_routes():
    #     print(f"Route (day={r.day_idx}, nurse={r.nurse}) nodes={r.nodes}")
    #     print("  start  :", getattr(r, 'start', None))
    #     print("  arrival:", getattr(r, 'arrival', None))
    #     print("  depart :", getattr(r, 'depart', None))
    
    ctx = Context(pd)
    report = ctx.check_solution(sol)

    # --- after building `pd`, `sol`, `ctx` ---
    d = 0
    event_j = 1                 # choose which event to test
    DEPOT_PM = pd.m + 2         # shared PM depot id

    # find all nurses attending event_j on day d
    attendees = []
    for (day_idx, w), r in sol.day_routes.items():
        if day_idx != d:
            continue
        if event_j in r.nodes:
            attendees.append(w)

    # helper: remove all occurrences of a node from a route and keep time arrays aligned
    def remove_node_everywhere(route, node_id: int):
        # collect indices to remove (in reverse so pops are safe)
        idxs = [i for i, n in enumerate(route.nodes) if n == node_id]
        for idx in reversed(idxs):
            route.nodes.pop(idx)
            # keep per-node arrays aligned if they exist
            for attr in ("start", "arrival", "depart"):
                arr = getattr(route, attr, None)
                if arr is not None and len(arr) > idx:
                    arr.pop(idx)

    # remove PM depot from all attendees' routes
    for w in attendees:
        r = sol.get_route(d, w)
        remove_node_everywhere(r, DEPOT_PM)

    for r in sol.iter_routes():
        print(f"Route (day={r.day_idx}, nurse={r.nurse}) nodes={r.nodes}")
        print("  start  :", getattr(r, 'start', None))
        print("  arrival:", getattr(r, 'arrival', None))
        print("  depart :", getattr(r, 'depart', None))
    # (optional) sanity: every attendee still has AM depot so the violation is *only* missing PM
    # AM_ok = all((pd.m + 1) in sol.get_route(d, w).nodes for w in attendees)

    # run feasibility check; should trigger kind="depot_cover" for event_j
    report = ctx.check_solution(sol)
    if report.ok:
        print("Unexpected: still FEASIBLE")
    else:
        for v in report.violations:
            if v.kind == "depot_cover" and v.day == d and v.event == event_j:
                print("Depot coverage violation triggered as expected ✅")
            else:
                print(f"Other violation: {v.kind} :: {v.msg}")

    if report.ok:
        print("FEASIBLE ✅")
    else:
        print("INFEASIBLE ❌")
        for v in report.violations:
            print(v)
    # pd = make_problem()

    # # from yourmodule import routes_from_active_x_t  # <-- replace with your real import

    # active_x, active_t = make_feasible_active(pd)
    # sol = routes_from_active_x_t(active_x, active_t, pd)

    # # sanity: show a route and its times
    # for r in sol.iter_routes():
    #     print(f"Route (day={r.day_idx}, nurse={r.nurse}) nodes={r.nodes}")
    #     print("  start  :", getattr(r, 'start', None))
    #     print("  arrival:", getattr(r, 'arrival', None))
    #     print("  depart :", getattr(r, 'depart', None))

    # # You: import your Context with checker
    # # from yourmodule import Context  # <-- replace with your real import

    # ctx = Context(pd)
    # # report = ctx.check_solution(sol)
    # # print_report(report)

    # # ---- Quick negative tests you can toggle ----

    # # (A) Make travel infeasible: shrink an event start time so travel can't fit
    # # r0 = sol.get_route(0, 0)
    # # idx_evt0 = r0.nodes.index(0)
    # # r0.start[idx_evt0] = 0  # way too early relative to AM -> event travel
    # # r0.depart[idx_evt0] = r0.start[idx_evt0] + pd.C_dur[0]
    # # report = ctx.check_solution(sol)
    # # print("\nAfter forcing travel violation:")
    # # print_report(report)

    # # (B) Break unique start: change nurse 1's start for event 1
    # r1 = sol.get_route(0, 1)
    # k = r1.nodes.index(1)
    # r1.start[k] = 9*60  # 09:00, conflicts with canonical 08:20
    # r1.depart[k] = r1.start[k] + pd.C_dur[1]
    # report = ctx.check_solution(sol)
    # print("\nAfter breaking unique/consistent start:")
    # print_report(report)

    # # (C) Fail staffing: flip nurse 1 to RN so we have 2 RNs and 0 LVNs at event 1
    # # pd.nurse_type[1] = "RN"
    # # report = ctx.check_solution(sol)
    # # print("\nAfter breaking staffing requirements:")
    # # print_report(report)

    # # (D) Fail depot coverage: remove PM from nurse 0’s route
    # r0 = sol.get_route(0, 0)
    # r0.nodes.remove(pd.m + 2)  # remove DEPOT_PM
    # # keep starts/arrival/depart lengths aligned or rebuild routes
    # report = ctx.check_solution(sol)
    # print("\nAfter breaking depot AM/PM coverage:")
    # print_report(report)

if __name__ == "__main__":
    main()
