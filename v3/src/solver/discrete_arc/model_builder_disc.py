import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.structures.problem_data import ProblemData
from src.solver.config import SolverConfig
# from .constraints_disc import add_base_constraints


def build_model(problem_data: ProblemData, config: SolverConfig) -> gp.Model:
    """
    Build and return a Gurobi optimization model based on the provided problem data and configuration.

    Parameters:
    problem_data (ProblemData): Travel costs, event durations, time windows, nurse requirements, etc.
    config (SolverConfig): Configuration settings for the solver.

    Returns:
    gp.Model: A Gurobi optimization model.
    """
    model = gp.Model("Nurse_Scheduling_Routing_Problem_Discrete_Arc")

    C_event = problem_data.event_event_costs
    C_home = problem_data.home_event_costs
    C_depot_e = problem_data.event_depot_costs
    C_depot_h = problem_data.home_depot_costs
    # C_depot = np.concatenate([C_depot_e, C_depot_h])
    C_dur = problem_data.event_durations
    time_windows = problem_data.time_windows
    min_nurses = problem_data.min_nurses
    nr = problem_data.total_rn
    nl = problem_data.total_lvn
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    # build discrete time blocks
    blocks_per_day, days, B_d, B_jd, B_j = build_discrete_blocks(time_windows)



    W = range(n)   # nurses: 0, 1, ..., n-1
    I = range(m)   # events: 0, 1, ..., m-1
    D = range(days)  # days: 0, 1, ..., days-1

    RN = range(nr)          # RNs: 0, 1, ..., nr-1
    LVN = range(nr, n)      # LVNs: nr, nr+

    # Variables

    HOME, DEPOT = "H", "D"
    NODES_ORIG = list(I) + [HOME, DEPOT]
    HOME_DEPOT = [HOME, DEPOT]

    # # debugging
    # print("Blocks per day:")
    # print(B_d)
    # print("Feasible blocks per event:")
    # for j in list(I) + ["H", "D"]:
    #     print(f"Event {j}: {B_j[j]}")

    # # also print B_jd
    # print("Feasible blocks per event-day:")
    # for j in list(I) + ["H", "D"]:
    #     for d in D:
    #         print(f"Event {j}, Day {d}: {B_jd.get((j,d), [])}")

    # Build allowed keys
    x_keys = gp.tuplelist(
        (w, i, j, b)
        for w in W
        for i in NODES_ORIG
        for j in NODES_ORIG
        for b in B_j[j]          # <-- only feasible blocks for that destination event / home / depot
        if i != j                # drop self-loop
    )

    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")

    s_keys = gp.tuplelist(
        (i, b)
        for i in I
        for b in B_j[i]
    )
    
    s = model.addVars(s_keys, vtype=GRB.BINARY, name="s")

    # 4) alpha_{id}^w : nurse w picks up at depot for event i on day d
    # alpha_keys: only allow alpha[w,i,d] if B_jd[(i,d)] is nonempty (i.e. event i could be scheduled on day d)
    alpha_beta_keys = gp.tuplelist(
        (w, i, d)
        for w in W
        for i in I
        for d in D
        if (i, d) in B_jd and B_jd[(i, d)]
    )
    
    alpha = model.addVars(alpha_beta_keys, vtype=GRB.BINARY, name="alpha")

    # 5) beta_{id}^w : nurse w drops off at depot for event i on day d
    beta = model.addVars(alpha_beta_keys, vtype=GRB.BINARY, name="beta")

    # Objective
    def is_event(node):
        return isinstance(node, int)  # since your events are indexed 0..m-1

    obj_x = gp.quicksum(
        (
            C_event[i, j] if is_event(i) and is_event(j) else
            C_home[w, j]  if i == HOME and is_event(j) else
            C_home[w, i]  if is_event(i) and j == HOME else
            C_depot_e[j]  if i == DEPOT and is_event(j) else
            C_depot_e[i]  if is_event(i) and j == DEPOT else
            C_depot_h[w]  if (i == HOME and j == DEPOT) or (i == DEPOT and j == HOME) else
            0  # default cost for any arcs we didn't explicitly define (e.g. home to home)
        ) * x[w, i, j, b]
        for (w, i, j, b) in x.keys()   # safe when x is sparse
    )



    model.setObjective(obj_x, GRB.MINIMIZE)


    # Constraints
    # Each event happens once:
    model.addConstrs(
        (gp.quicksum(s[i, b] for b in B_j[i]) == 1 for i in I),
        name="event_once"
    )

    # arrive only if scheduled
    model.addConstrs(
        (gp.quicksum(x[w,i,j,b] for i in NODES_ORIG if (w,i,j,b) in x) <= s[j,b] for w in W for j in I for b in B_j[j]),
        name="arrive_implies_scheduled"
    )

    # Staffing constraints: for each event j, enough RNs and LVNs arrive at j (once each).
    for j in I:
        model.addConstr(
            gp.quicksum(
                gp.quicksum(x[w, i, j, b] for i in NODES_ORIG for b in B_j[j] if (w, i, j, b) in x)
                for w in RN
            ) >= min_nurses[j][0],
            name=f"min_RN_j{j}"
        )

        model.addConstr(
            gp.quicksum(
                gp.quicksum(x[w, i, j, b] for i in NODES_ORIG for b in B_j[j] if (w, i, j, b) in x)
                for w in LVN
            ) >= min_nurses[j][1],
            name=f"min_LVN_j{j}"
        )

    # # each event is visited at most once by each nurse
    # for w in W:
    #     for j in I:
    #         model.addConstr(
    #             gp.quicksum(
    #                 x[w,i,j,b]
    #                 for i in NODES_ORIG
    #                 for b in B_j[j]
    #                 if (w,i,j,b) in x
    #             ) <= 1,
    #             name=f"visit_once_w{w}_j{j}"
    #         )

    # # prune self-loops: no x[i,i,b] -- handled via x_keys construction
    # for w in W:
    #     for j in I:
    #         for b in B_j[j]:
    #             if (w, j, j, b) in x:
    #                 model.addConstr(
    #                     x[w, j, j, b] == 0,
    #                     name=f"no_self_loop_w{w}_j{j}_b{b}"
    #                 )

    # # prune self-loops: no y[loc,loc,d] -- handled via y_keys construction
    # for w in W:
    #     for loc in HOME_DEPOT:
    #         for d in D:
    #             if (w, loc, loc, d) in y:
    #                 model.addConstr(
    #                     y[w, loc, loc, d] == 0,
    #                     name=f"no_self_loop_y_w{w}_loc{loc}_d{d}"
    #                 )

    # Depot
    model.addConstrs((gp.quicksum(alpha[w, j, d] for w in W for d in D if (w, j, d) in alpha) == 1 for j in I), name="pick_up_leader")

    model.addConstrs((gp.quicksum(beta[w, j, d] for w in W for d in D if (w, j, d) in beta) == 1 for j in I), name="drop_off_leader")

    # alpha/beta implies visit: if nurse w picks up/drops off at depot for event j on day d, then they must have a flow into j on day d (i.e. they must visit j that day)
    for w in W:
        for j in I:
            for d in D:
                inflow_wjd = gp.quicksum(
                    x[w, i, j, b]
                    for i in NODES_ORIG
                    for b in B_d[d]
                    if (w, i, j, b) in x   # important if x is sparse (only feasible keys)
                )

                if (w, j, d) in alpha:
                    model.addConstr(alpha[w, j, d] <= inflow_wjd, name=f"alpha_implies_visit_w{w}_j{j}_d{d}")
                if (w, j, d) in beta:
                    model.addConstr(beta[w, j, d]  <= inflow_wjd, name=f"beta_implies_visit_w{w}_j{j}_d{d}")

    # alpha/beta implies depot trips
    # model.addConstrs(
    #     (alpha[w, i, d] <= gp.quicksum(
    #             x[w, DEPOT, j, b]   # morning depot->event flow
    #             for j in I
    #             for b in B_jd.get((j, d), [])
    #             if (w, DEPOT, j, b) in x   # needed if x is sparse / only feasible keys
    #         )
    #     for (w, i, d) in alpha.keys()),
    #     name="alpha_implies_HD"
    # )
    model.addConstrs(
        (
            alpha[w, j, d]
            <= (
                x[w, HOME, DEPOT, min(B_jd[(DEPOT, d)])]
                if B_jd[(DEPOT, d)] and (w, HOME, DEPOT, min(B_jd[(DEPOT, d)])) in x
                else 0
            )
            for (w, j, d) in alpha.keys()
        ),
        name="alpha_implies_morning_HD"
    )

    model.addConstrs(
        (
            beta[w, i, d] 
            <= (
                x[w, DEPOT, HOME, max(B_jd[("H", d)])]
                if B_jd[("H", d)] and (w, DEPOT, HOME, max(B_jd[("H", d)])) in x
                else 0
            )
        for (w, i, d) in beta.keys()),
        name="beta_implies_DH"
    )

    # (1) home flow at most 1
    model.addConstrs(
        (
            gp.quicksum(
                x[w, HOME, j, b]
                for j in NODES_ORIG
                for b in B_d[d]
                if (w, HOME, j, b) in x   # important if x is sparse (only feasible keys)
            )
            <= 1
            for w in W for d in D
        ),
        name="home_flow_at_most_one"
    )

    # event inflow = outflow, and time feasibility: for each event j, day d, nurse w, if nurse arrives at j on day d, they must either leave to home/depot that day or go to another event on the same day that they can feasibly reach after performing j's service.

    def time_offset_blocks(j, k, w):
        """Blocks needed for service at j plus travel j->k (minutes -> blocks, rounded up)."""
        if k in I:
            return tau(C_dur[j] + C_event[j][k])
        elif k == HOME:
            return tau(C_dur[j] + C_home[w][j]) 
        elif k == DEPOT:
            return tau(C_dur[j] + C_depot_e[j])
        else:
            return 0 

    def service_time(i):
        return C_dur[i] if i in I else 0

    def travel_time(i, j, w):
        if i in I and j in I:
            return C_event[i][j]
        elif i == HOME and j in I:
            return C_home[w][j]
        elif i == DEPOT and j in I:
            return C_depot_e[j]
        elif i in I and j == HOME:
            return C_home[w][i]
        elif i in I and j == DEPOT:
            return C_depot_e[i]
        elif (i == HOME and j == DEPOT) or (i == DEPOT and j == HOME):
            return C_depot_h[w]
        else:
            return 0

    # (4) outflow exists only when time-feasible inflow exists
    for w in W:
        for j in I:
            for k in NODES_ORIG:
                for b in B_j[k]:
                    if (w, j, k, b) not in x:
                        continue
                    d = delta(b)

                    inflow = gp.quicksum(
                        x[w, i, j, bp]
                        for i in NODES_ORIG
                        for bp in B_jd.get((j, d), [])
                        if (w, i, j, bp) in x
                        if bp <= b - tau(service_time(j) + travel_time(j, k, w))
                    )
                    model.addConstr(
                        x[w, j, k, b] <= inflow,
                        name=f"outflow_requires_inflow_w{w}_j{j}_k{k}_b{b}"
                    )

    # (5) event inflow = outflow (per day)
    for w in W:
        for j in I:   # apply to events and depot (but not home)
            # if j == HOME:
            #     continue
            for d in D:
                inflow = gp.quicksum(
                    x[w, i, j, b]
                    for i in NODES_ORIG
                    for b in B_jd.get((j, d), [])
                    if (w, i, j, b) in x
                )
                outflow = gp.quicksum(
                    x[w, j, k, bp]
                    for k in NODES_ORIG
                    for bp in B_jd.get((k, d), [])
                    if (w, j, k, bp) in x
                )
                model.addConstr(
                    inflow == outflow,
                    name=f"event_flow_balance_w{w}_j{j}_d{d}"
                )
    # depot inflow = outflow (per day)
    for w in W:
        for d in D:
            bd = B_jd.get((DEPOT, d), [])
            if not bd:
                continue  # no depot blocks for this day

            b_early = min(bd)

            # Only add if the morning home->depot arc exists; otherwise you'd force outflow=0
            if (w, HOME, DEPOT, b_early) not in x:
                continue

            inflow = x[w, HOME, DEPOT, b_early]

            outflow = gp.quicksum(
                x[w, DEPOT, j, bp]
                for j in I
                for bp in B_jd.get((j, d), [])
                if (w, DEPOT, j, bp) in x
            )

            model.addConstr(
                inflow == outflow,
                name=f"morning_depot_flow_balance_w{w}_d{d}"
            )

    for w in W:
        for d in D:
            bd = B_jd.get((HOME, d), [])
            if not bd:
                continue  # no home blocks for this day

            b_home_late = max(bd)

            # Only add if the evening depot->home arc exists; otherwise you'd force inflow=0
            if (w, DEPOT, HOME, b_home_late) not in x:
                continue
            
            outflow = x[w, DEPOT, HOME, b_home_late]

            b_depot_late = max(B_jd.get((DEPOT, d), []))

            inflow = gp.quicksum(
                x[w, j, DEPOT, b_depot_late]
                for j in I
                if (w, j, DEPOT, b_depot_late) in x
            )


            model.addConstr(
                inflow == outflow,
                name=f"evening_depot_flow_balance_w{w}_d{d}"
            )

    # for w in W:
    #     for j in I:
    #         # only enforce for b where x could represent arrival to j
    #         for b in B_j[j]:
    #             d = delta(b)

    #             inflow = gp.quicksum(
    #                 x[w, i, j, b]
    #                 for i in NODES_ORIG
    #                 if (w, i, j, b) in x
    #             )

    #             # feasible next-event departures from j within the same day
    #             outflow = gp.quicksum(
    #                 x[w, j, k, bp]
    #                 for k in NODES_ORIG
    #                 for bp in B_jd[(k,d)]
    #                 if bp >= b + time_offset_blocks(j, k, w)
    #                 if (w, j, k, bp) in x
    #             )

    #             model.addConstr(
    #                 inflow <= outflow,
    #                 name=f"flow_time_w{w}_j{j}_b{b}"
    #             )
    
    # link x and y: for all w, d, j in HOME/DEPOT: if nurse goes from i to loc and enters loc in block b (x), then y[w,i,loc,d] = 1 for that day
    # model.addConstrs(
    #     (y[w, i, loc, d] ==
    #     gp.quicksum(x[w, i, loc, b] for b in B_d[d] if (w, i, loc, b) in x))
    #     for w in W for i in NODES_ORIG for loc in HOME_DEPOT for d in D if any((w, i, loc, b) in x for b in B_d[d])
    # )

    if config.enforce_hour_balance:
        # Same averaging logic as your continuous version (minutes of service required)
        total_RN_minutes = float(np.sum(C_dur * min_nurses[:, 0]))
        total_LVN_minutes = float(np.sum(C_dur * min_nurses[:, 1]))
        avg_RN_minutes = total_RN_minutes / nr if nr > 0 else 0.0
        avg_LVN_minutes = total_LVN_minutes / nl if nl > 0 else 0.0

        def visit_expr(w, j):
            return gp.quicksum(
                x[w, i, j, b]
                for i in NODES_ORIG
                for b in B_j[j]
                if (w, i, j, b) in x  # needed if x is sparse
            )
        # RN nurses: w = 0..nr-1
        for w in RN:
            total_minutes_w = gp.quicksum(C_dur[j] * visit_expr(w, j) for j in range(m))
            model.addConstr(total_minutes_w >= 0.8 * avg_RN_minutes, name=f"min_balanced_hours_RN_w{w}")
            model.addConstr(total_minutes_w <= 1.2 * avg_RN_minutes, name=f"max_balanced_hours_RN_w{w}")

        # LVN nurses: w = nr..n-1
        for w in LVN:
            total_minutes_w = gp.quicksum(C_dur[j] * visit_expr(w, j) for j in range(m))
            model.addConstr(total_minutes_w >= 0.8 * avg_LVN_minutes, name=f"min_balanced_hours_LVN_w{w}")
            model.addConstr(total_minutes_w <= 1.2 * avg_LVN_minutes, name=f"max_balanced_hours_LVN_w{w}")


    # Set solver parameters from config
    if config.work_limit is not None:
        model.Params.WorkLimit = config.work_limit
    if config.time_limit is not None:
        model.Params.TimeLimit = config.time_limit
    model.Params.Seed = config.seed
    model.Params.OutputFlag = config.gurobi_outputflag

    return model

import math

BLOCK_LEN = 30          # minutes
BLOCKS_PER_DAY = 48     # 24*60 / 30

def delta(b):
    """Return day index (0-based) of global block b."""
    return b // BLOCKS_PER_DAY

def tau(t):
    """Return number of 30-min blocks needed for time t (minutes), rounded up."""
    return math.ceil(t / BLOCK_LEN)

def build_discrete_blocks(time_windows):
    """
    time_windows[i][d] = (lb, ub) in minutes from 0..1440 within that day.
    We assume event start times must lie on block boundaries (k * block_len_min).

    Returns:
      blocks_per_day: int
      days: int
      B_d: dict d -> list of global block indices in that day
      B_jd: dict (j,d) -> list of feasible global blocks for event j on day d
      B_j: dict j -> sorted list of feasible global blocks across all days
    """
    m = len(time_windows)
    days = len(time_windows[0])
    # blocks_per_day = 1440 // block_len_min  # 48 for 30-min blocks
    blocks_per_day = BLOCKS_PER_DAY  # use constant for consistency
    block_len_min = BLOCK_LEN

    # Day blocks: global indexing b = d*blocks_per_day + k
    B_d = {
        d: list(range(d * blocks_per_day, (d + 1) * blocks_per_day))
        for d in range(days)
    }

    B_jd = {}
    B_j = {j: [] for j in range(m)} 
    # add "H" and "D" as special events with their own feasible blocks
    B_j["H"] = []  # home
    B_j["D"] = []  # depot

    for j in range(m):
        for d in range(days):
            lb, ub = time_windows[j][d]

            # Guard against None / missing windows if you have those
            if lb is None or ub is None:
                B_jd[(j, d)] = []
                continue

            if ub == 0:
                B_jd[(j, d)] = []
                continue

            # Clamp to [0, 1440]
            lb = max(0, min(1440, lb))
            ub = max(0, min(1440, ub))

            # If window is empty, no feasible blocks
            if ub < lb:
                B_jd[(j, d)] = []
                continue

            # Convert minute window to feasible within-day block starts.
            # Start time = k*L must satisfy lb <= k*L <= ub.
            k_min = math.ceil(lb / block_len_min)
            k_max = math.floor(ub / block_len_min)

            # Keep k within [0, blocks_per_day-1]
            k_min = max(0, k_min)
            k_max = min(blocks_per_day - 1, k_max)

            if k_max < k_min:
                B_jd[(j, d)] = []
                continue

            # Global block indices
            feasible_global = [d * blocks_per_day + k for k in range(k_min, k_max + 1)]
            B_jd[(j, d)] = feasible_global
            B_j[j].extend(feasible_global)
    
    for j in ["H", "D"]:
        # For the 2 special "events" (morning / evening home, morning / evening depot)
        # for each day, allow 0,46 for home, 6,40 for depot (these are just proxies that guarantees feasibility)
        # these are just proxies that guarantees feasibility
        for d in range(days):
            if j == "H":  # home
                B_jd[("H", d)] = [d * blocks_per_day + 0, d * blocks_per_day + 46]
            elif j == "D":  # depot
                B_jd[("D", d)] = [d * blocks_per_day + 6, d * blocks_per_day + 40]
            # elif j == m+2:  # evening depot
            #     B_jd[(j, d)] = [d * blocks_per_day + 40]
            # elif j == m+3:  # evening home
            #     B_jd[(j, d)] = [d * blocks_per_day + 46]
            B_j[j].extend(B_jd[(j, d)])

    # Sort + unique (in case windows overlap oddly)
    for j in list(range(m)) + ["H", "D"]:
        B_j[j] = sorted(set(B_j[j]))

    return blocks_per_day, days, B_d, B_jd, B_j
