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
    blocks_per_day, days, B_d, B_jd, B_j = build_discrete_blocks(time_windows, block_len_min=30)



    W = range(n)   # nurses: 0, 1, ..., n-1
    I = range(m)   # events: 0, 1, ..., m-1
    D = range(days)  # days: 0, 1, ..., days-1

    RN = range(nr)          # RNs: 0, 1, ..., nr-1
    LVN = range(nr, n)      # LVNs: nr, nr+

    # Variables

    HOME, DEPOT = "H", "D"
    NODES_ORIG = list(I) + [HOME, DEPOT]
    HOME_DEPOT = [HOME, DEPOT]

    # debugging
    # print("Blocks per day:")
    # print(B_d)
    # print("Feasible blocks per event:")
    # for j in I:
    #     print(f"Event {j}: {B_j[j]}")

    # Build allowed keys
    x_keys = gp.tuplelist(
        (w, i, j, b)
        for w in W
        for i in NODES_ORIG
        for j in I
        for b in B_j[j]          # <-- only feasible blocks for that destination event
        if i != j                # optional: drop self-loop if both are events
    )

    x = model.addVars(x_keys, vtype=GRB.BINARY, name="x")
    y = model.addVars(
        W, NODES_ORIG, HOME_DEPOT, range(days),
        vtype=GRB.BINARY,
        name="y"
    )
    # new termination-by-block variable (only meaningful for j in I)
    y_end_keys = gp.tuplelist(
        (w, j, loc, b)
        for w in W
        for j in I
        for loc in HOME_DEPOT
        for b in B_j[j]
    )
    y_end = model.addVars(y_end_keys, vtype=GRB.BINARY, name="y_end")

    # link to day-level y: y[w,j,loc,d] = sum_{b in day d feasible for j} y_end[w,j,loc,b]
    model.addConstrs(
        (y[w, j, loc, d] ==
        gp.quicksum(y_end[w, j, loc, b] for b in B_jd[(j, d)] if (w, j, loc, b) in y_end)
        for w in W for j in I for loc in HOME_DEPOT for d in range(days)),
        name="link_yend_to_y"
    )

    s = {}
    for i in I:
        for b in B_j[i]:
            s[i, b] = model.addVar(vtype=GRB.BINARY, name=f"s[{i},{b}]")
    s = gp.tupledict(s)

    # 4) alpha_{id}^w : nurse w picks up at depot for event i on day d
    alpha = model.addVars(
        W, I, D,
        vtype=GRB.BINARY,
        name="alpha"
    )

    # 5) beta_{id}^w : nurse w drops off at depot for event i on day d
    beta = model.addVars(
        W, I, D,
        vtype=GRB.BINARY,
        name="beta"
    )

    # Objective
    def is_event(node):
        return isinstance(node, int)  # since your events are indexed 0..m-1

    # ----- x-part (arrive at event j) -----
    obj_x = gp.quicksum(
        (
            C_event[i, j] if is_event(i) else
            C_home[w, j]  if i == HOME else
            C_depot_e[j]  # i == DEPOT
        ) * x[w, i, j, b]
        for (w, i, j, b) in x.keys()   # safe when x is sparse
    )

    # ----- y-part (arrive at home/depot) -----
    terms_y = []
    for (w, i, loc, d) in y.keys():
        # Skip meaningless arcs if they exist (recommended to not create them in the first place)
        if i == loc:
            continue

        if is_event(i) and loc == HOME:
            cij = C_home[w, i]        # event -> home (assumed symmetric)
        elif is_event(i) and loc == DEPOT:
            cij = C_depot_e[i]        # event -> depot
        elif (i == HOME and loc == DEPOT) or (i == DEPOT and loc == HOME):
            cij = C_depot_h[w]        # home <-> depot
        else:
            continue

        terms_y.append(cij * y[w, i, loc, d])

    obj_y = gp.quicksum(terms_y)

    model.setObjective(obj_x + obj_y, GRB.MINIMIZE)


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

    # each event is visited at most once by each nurse
    for w in W:
        for j in I:
            model.addConstr(
                gp.quicksum(
                    x[w,i,j,b]
                    for i in NODES_ORIG
                    for b in B_j[j]
                    if (w,i,j,b) in x
                ) <= 1,
                name=f"visit_once_w{w}_j{j}"
            )
    # prune self-loops: no x[i,i,b]
    for w in W:
        for j in I:
            for b in B_j[j]:
                if (w, j, j, b) in x:
                    model.addConstr(
                        x[w, j, j, b] == 0,
                        name=f"no_self_loop_w{w}_j{j}_b{b}"
                    )
    # prune self-loops: no y[loc,loc,d]
    for w in W:
        for loc in HOME_DEPOT:
            for d in D:
                if (w, loc, loc, d) in y:
                    model.addConstr(
                        y[w, loc, loc, d] == 0,
                        name=f"no_self_loop_y_w{w}_loc{loc}_d{d}"
                    )
    # Depot
    model.addConstrs(
        (gp.quicksum(alpha[w, j, d] for w in W for d in D) == 1 for j in I),
        name="pick_up_leader"
    )

    model.addConstrs(
        (gp.quicksum(beta[w, j, d] for w in W for d in D) == 1 for j in I),
        name="drop_off_leader"
    )



    # Assumes:
    #   alpha[w,j,d], beta[w,j,d]   (as created earlier: model.addVars(W, I, D, ...))
    #   x[w,i,j,b] exists only for feasible (w,i,j,b)
    #   B_d[d] = list of global blocks belonging to day d
    #   NODES_ORIG = list(I) + ["H","D"]

    for w in W:
        for j in I:
            for d in D:
                inflow_wjd = gp.quicksum(
                    x[w, i, j, b]
                    for i in NODES_ORIG
                    for b in B_d[d]
                    if (w, i, j, b) in x   # important if x is sparse (only feasible keys)
                )

                model.addConstr(alpha[w, j, d] <= inflow_wjd, name=f"alpha_implies_visit_w{w}_j{j}_d{d}")
                model.addConstr(beta[w, j, d]  <= inflow_wjd, name=f"beta_implies_visit_w{w}_j{j}_d{d}")

    # alpha/beta implies depot trips
    model.addConstrs(
        (alpha[w, i, d] <= y[w, HOME, DEPOT, d]
        for w in W for i in I for d in D),
        name="alpha_implies_HD"
    )

    model.addConstrs(
        (beta[w, i, d] <= y[w, DEPOT, HOME, d]
        for w in W for i in I for d in D),
        name="beta_implies_DH"
    )

    # home flow at most 1
    model.addConstrs(
        (
            y[w, HOME, DEPOT, d]
            + gp.quicksum(
                x[w, HOME, j, b]
                for j in I
                for b in B_d[d]
                if (w, HOME, j, b) in x   # important if x is sparse (only feasible keys)
            )
            <= 1
            for w in W for d in D
        ),
        name="home_flow_at_most_one"
    )

    # (1) depot pick-up linking: y_HD equals exactly one depot->event departure that day
    model.addConstrs(
        (
            y[w, HOME, DEPOT, d]
            == gp.quicksum(
                x[w, DEPOT, j, b]
                for j in I
                for b in B_d[d]
                if (w, DEPOT, j, b) in x   # needed if x is sparse / only feasible keys
            )
            for w in W for d in D
        ),
        name="depot_pickup_link"
    )

    # (2) depot drop-off linking: y_DH equals exactly one event->depot arrival that day
    # Here the arc into DEPOT is a y-variable: y[w, i, DEPOT, d]
    model.addConstrs(
        (
            y[w, DEPOT, HOME, d]
            == gp.quicksum(
                y[w, i, DEPOT, d]
                for i in I
            )
            for w in W for d in D
        ),
        name="depot_dropoff_link"
    )
    def time_offset_blocks(j, l):
        """Blocks needed for service at j plus travel j->l (minutes -> blocks, rounded up)."""
        return tau(C_dur[j] + C_event[j][l])  # adjust indexing if c_travel is dict[(j,l)]

    for w in W:
        for j in I:
            # only enforce for b where x could represent arrival to j
            for b in B_j[j]:
                d = delta(b)

                inflow = gp.quicksum(
                    x[w, i, j, b]
                    for i in NODES_ORIG
                    if (w, i, j, b) in x
                )

                # termination to home/depot on that day (time suppressed)
                term = y_end[w, j, HOME, b] + y_end[w, j, DEPOT, b]

                # feasible next-event departures from j within the same day
                out_to_events = gp.quicksum(
                    x[w, j, l, bp]
                    for l in I
                    for bp in B_jd[(l,d)]
                    if bp >= b + time_offset_blocks(j, l)
                    if (w, j, l, bp) in x
                )

                model.addConstr(
                    inflow == term + out_to_events,
                    name=f"flow_time_w{w}_j{j}_b{b}"
                )
    model.addConstrs(
        (gp.quicksum(y_end[w,j,loc,b] for loc in HOME_DEPOT for b in B_j[j] if (w,j,loc,b) in y_end) <= 1
        for w in W for j in I),
        name="end_at_most_once_per_event"
    )


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

def build_discrete_blocks(time_windows, block_len_min=30):
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
    blocks_per_day = 1440 // block_len_min  # 48 for 30-min blocks

    # Day blocks: global indexing b = d*blocks_per_day + k
    B_d = {
        d: list(range(d * blocks_per_day, (d + 1) * blocks_per_day))
        for d in range(days)
    }

    B_jd = {}
    B_j = {j: [] for j in range(m)}

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

    # Sort + unique (in case windows overlap oddly)
    for j in range(m):
        B_j[j] = sorted(set(B_j[j]))

    return blocks_per_day, days, B_d, B_jd, B_j

BLOCK_LEN = 30          # minutes
BLOCKS_PER_DAY = 48     # 24*60 / 30

def delta(b):
    """Return day index (0-based) of global block b."""
    return b // BLOCKS_PER_DAY

def tau(t):
    """Return number of 30-min blocks needed for time t (minutes), rounded up."""
    return math.ceil(t / BLOCK_LEN)