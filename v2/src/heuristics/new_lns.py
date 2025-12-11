from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import random
from typing import Dict, Tuple, List, Set, TYPE_CHECKING
from xml.parsers.expat import model
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.solver.extract import routes_from_active_x_t

def build_full_model(data, *, min_hour=None, max_hour=None):
    m, n, day = data.m, data.n, data.day
    C_event, C_home, C_depot_e, C_depot_h, C_dur, time_window, min_nurse = data.C_event, data.C_home, data.C_depot_e, data.C_depot_h, data.C_dur, data.time_window, data.min_nurse
    C_depot = np.concatenate([C_depot_e, C_depot_h])

    nr = data.nr
    nl = data.nl
    model = gp.Model("MVT_scheduling_continuous")

    # --- variables (same shapes as you have now) ---
    x = model.addVars(m+3, m+3, day, n, vtype=GRB.BINARY, name="x")   # x[i,j,d,w]
    s = model.addVars(m, day, vtype=GRB.BINARY, name="s")             # s[i,d]
    t = model.addVars(m, day, vtype=GRB.INTEGER, name="t")            # t[i,d]
    alpha = model.addVars(m, day, n, vtype=GRB.BINARY, name="alpha")  # alpha[i,d,w]
    beta  = model.addVars(m, day, n, vtype=GRB.BINARY, name="beta")   # beta[i,d,w]

    # --- objective & constraints ---
    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(day) for w in range(n)) for i in range(m) for j in range(m)
    )
    home_cost = gp.quicksum(
        C_home[w,i] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(day)) for w in range (n) for i in range(m)
    )
    depot_event_cost = gp.quicksum(
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+2, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )
    depot_home_cost = gp.quicksum(
        C_depot[m+w] * gp.quicksum((x[m+2, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )
    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    model.setObjective(objective, GRB.MINIMIZE)

    # Pruning
    model.addConstrs(x[i,i,d,w] == 0 for d in range(day) for w in range(n) for i in range(m+2))

    # infeasible depot travels
    # event -> depot_am, depot_pm -> event
    model.addConstrs(x[i,m+1,d,w] == 0 for d in range(day) for w in range(n) for i in range(m))
    model.addConstrs(x[m+2,i,d,w] == 0 for d in range(day) for w in range(n) for i in range(m))
    # depot_am -> home, home -> depot_pm
    model.addConstrs(x[m+1,m,d,w] == 0 for d in range(day) for w in range(n))
    model.addConstrs(x[m,m+2,d,w] == 0 for d in range(day) for w in range(n))

    # Each event happens on one day
    model.addConstrs(gp.quicksum(s[i,d] for d in range(day)) == 1 for i in range(m))

    # Each event is scheduled exactly once during its feasible time window
    model.addConstrs(gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m))
    model.addConstrs(t[i,d] >= time_window[i][d][0] * s[i,d] for d in range(day) for i in range(m))
    model.addConstrs(t[i,d] <= time_window[i][d][1] * s[i,d] for d in range(day) for i in range(m))
    model.addConstrs(sum(x[i,j,d,w] for j in range(m+3)) <= s[i,d] for i in range(m) for d in range(day) for w in range(n))

    M = 1440  # large constant for time constraints
    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                model.addConstrs(t[j,d] >= t[i,d] + C_dur[i] + C_event[i,j] - M * (1 - x[i,j,d,w]) for w in range(n) for d in range(day))
 
    # Minimum working hours
    if min_hour is not None:
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day)) for j in range(m)) >= min_hour[w] * 60 for w in range(n)),
            name="min_working_hours"
        )

    # Maximum working hours
    if max_hour is not None:
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day)) for j in range(m)) <= max_hour[w] * 60 for w in range(n)),
            name="max_working_hours"
        )
    
    # staffing
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day) for w in range(nr)) >= min_nurse[j][0] for j in range(m)),
        name="min_RN"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day) for w in range(nr, nr+nl)) >= min_nurse[j][1] for j in range(m)),
        name="min_LVN"
    )        
  
    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+3)) == gp.quicksum(x[j, i, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="event_network_flow"
    )

    # outflow from home is at most 1
    model.addConstrs(
        (gp.quicksum(x[m, i, d, w] for i in range(m+3)) <= 1 for d in range(day) for w in range(n)),
        name="home_outflow"
    )

    # depot inflow = outflow
    model.addConstrs(
        (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="morning_depot_flow"
    )

    model.addConstrs(
        (x[m+2, m, d, w] == gp.quicksum(x[j, m+2, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="evening_depot_flow"
    )
    
    # team leader: exactly one pick up and one drop off leader for each event
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for w in range(n) for d in range(day)) == 1 for j in range(m)),
        name="pick_up_leader"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for w in range(n) for d in range(day)) == 1 for j in range(m)),
        name="drop_off_leader"
    )

    # team leader goes to the event
    model.addConstrs(
        (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="pick_up_leader_event"
    )

    model.addConstrs(
        (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="drop_off_leader_event"
    )

    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_home_depot"
    )
    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+2, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_depot_home"
    )

    model.update()
    return model, x, t, s, alpha, beta

def _extract_active_var(model, x, s, t, alpha, beta):
    """
    Extracts active variable values from a solved Gurobi model.

    Args:
        model: solved gurobipy.Model
        x, s, t, alpha, beta: dicts of Gurobi variables

    Returns:
        active_x: dict of (i,j,d,w):1 for selected arcs
        active_t: dict of (i,d): start times
        active_s, active_alpha, active_beta: (empty or filled if needed)
    """
    active_x = {}
    active_t = {}
    active_s = {}
    active_alpha = {}
    active_beta = {}

    for key, var in x.items():
        if var.X > 0.5:  # binary route indicator
            active_x[key] = 1

    for key, var in t.items():
        active_t[key] = var.X

    # These might not be used in your heuristic, but we include them for completeness
    for key, var in s.items():
        active_s[key] = var.X
    for key, var in alpha.items():
        active_alpha[key] = var.X
    for key, var in beta.items():
        active_beta[key] = var.X

    return active_x, active_t, active_s, active_alpha, active_beta


def lns_day(inc_sol, active_x, active_t, pd, ctx, day_idx, worklimit=1000, threads=8, verbose=False):
    """
    Local Neighborhood Search (LNS) for one or multiple days.

    Args:
        inc_sol: current incumbent Solution object
        active_x: dict of active arcs from incumbent solution
        active_t: dict of active start times from incumbent solution
        pd: problem data
        ctx: solver context (used for feasibility checks)
        day_idx: int or list[int], day(s) to optimize
        worklimit: Gurobi WorkLimit parameter (default=1000)
        threads: number of threads (default=8)
        verbose: if True, prints solver output

    Returns:
        active_x, active_t, new_sol: updated structures
    """

    # Normalize input: always treat day_idx as a list
    if isinstance(day_idx, int):
        day_idx = [day_idx]

    # Build full model
    model, x, t, s, alpha, beta = build_full_model(pd, min_hour=None, max_hour=25 * np.ones(pd.n))
    m = pd.m
    n = pd.n

    # ---- Fix all arcs not on selected day(s) ----
    for (i, j, d, w), var in x.items():
        if d not in day_idx:
            val = 1.0 if (i, j, d, w) in active_x else 0.0
            var.LB, var.UB = val, val

    # ---- Fix depot-to-depot arcs (AM/PM/home loops) ----
    depot_nodes = [m, m + 1, m + 2]  # home, depot AM, depot PM
    for (i, j, d, w), var in x.items():
        if i in depot_nodes and j in depot_nodes and d not in day_idx:
            val = 1.0 if (i, j, d, w) in active_x else 0.0
            var.LB, var.UB = val, val

    # ---- Warm start ----
    for (i, j, d, w), var in x.items():
        var.Start = 1.0 if (i, j, d, w) in active_x else 0.0

    # Optionally warm-start continuous vars
    # for (i, d), val in active_t.items():
    #     if (i, d) in t:
    #         t[i, d].Start = val

    # ---- Solver settings ----
    model.Params.MIPFocus = 1
    model.Params.Threads = threads
    model.Params.WorkLimit = worklimit
    model.Params.OutputFlag = 1 if verbose else 0

    # ---- Debug: check how many arcs are actually free ----
    free_count = sum(
        1 for (i, j, d, w), var in x.items()
        if d in day_idx and (var.LB < 0.5 or var.UB > 0.5)
    )
    print(f"Free arcs on day(s) {day_idx}: {free_count}")

    # ---- Solve ----
    model.optimize()

    if model.SolCount == 0:
        if verbose:
            print(f"No feasible improvement found for day(s) {day_idx}.")
        return active_x, active_t, inc_sol

    # ---- Extract new solution ----
    active_x, active_t, _, _, _ = _extract_active_var(model, x, s, t, alpha, beta)
    new_sol = routes_from_active_x_t(active_x, active_t, pd)

    if getattr(ctx.check_solution(new_sol), "feasible", True) and model.ObjVal < getattr(inc_sol, "obj", float('inf')):
        new_sol.obj = float(model.ObjVal)
        if verbose:
            print(f"Day(s) {day_idx} improved: obj = {new_sol.obj:.2f}")
        return active_x, active_t, new_sol
    else:
        return active_x, active_t, inc_sol
