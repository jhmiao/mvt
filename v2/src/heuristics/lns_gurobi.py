# src/heuristics/lns_gurobi.py
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import random
from typing import Dict, Tuple, List, Set, TYPE_CHECKING
from xml.parsers.expat import model
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Only for type checkers; no runtime import (prevents cycles)
if TYPE_CHECKING:
    from src.models.solution import Solution, Route
    from src.models.problem_data import ProblemData
    from src.models.context import Context

# Types you already have:
# - Solution with: day_routes: Dict[(int day, int nurse), Route]
# - Route with fields: day_idx, nurse, nodes (e.g., [100+w, ..., 100+w])
# - Context(pd).check_solution(sol) -> report
# - routes_from_active_x_t(active_x, active_t, pd) -> Solution

from src.solver.extract import routes_from_active_x_t

# 1) Build the full model
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
    alpha = model.addVars(m, day, n, vtype=GRB.BINARY, name="alpha")
    beta  = model.addVars(m, day, n, vtype=GRB.BINARY, name="beta")

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

    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            for d in range(day):
                for w in range(n):
                    model.addGenConstrIndicator(
                        x[i, j, d, w],
                        True,
                        t[j, d] >= t[i, d] + C_dur[i] + C_event[i, j],
                        name=f"time_ind_i{i}_j{j}_d{d}_w{w}"
                    )
 
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

# 2) Helpers, Destroy, Hard Fix, Warm Start, etc.
def _to_mip_node(v, w, m): 
    return m if v == 100 + w else v

# collect candidate in/out sets from x’s current UBs (fast for fixing)
def _cand_sets(pd, x):
    m, n, D = pd.m, pd.n, pd.day
    cand_out, cand_in = {}, {}
    for d in range(D):
        for w in range(n):
            for i in range(m+3):
                outs = [j for j in range(m+3) if j != i and x[i,j,d,w].UB > 0]
                cand_out[(d,w,i)] = outs
            for j in range(m+3):
                ins = [i for i in range(m+3) if i != j and x[i,j,d,w].UB > 0]
                cand_in[(d,w,j)] = ins
    return cand_in, cand_out

# choose a random set of event–day pairs to free
def _choose_destroy_eventdays(inc_sol, pd, frac, rng):
    # collect all (i,d) that appear in incumbent routes
    eventdays = set()
    for (d,w), r in inc_sol.day_routes.items():
        for v in r.nodes:
            if 0 <= v < pd.m:
                eventdays.add((v, d))
    if not eventdays: return set()
    k = max(1, round(frac * len(eventdays)))
    return set(rng.sample(list(eventdays), k))

# hard-fix outside the destroyed (i,d) set
def _hard_fix_multinurse(model, x, s, t, inc_sol, active_t, U, pd, cand_in, cand_out, keep_route_anchor=True):
    m, n, D = pd.m, pd.n, pd.day

    # for each event t(i,d), get team membership
    team = { (i,d): set() for i in range(m) for d in range(D) }
    for (d,w), r in inc_sol.day_routes.items():
        for v in r.nodes:
            if 0 <= v < m:
                team.setdefault((v,d), set()).add(w)

    # Fix s and t for all non-destroyed (i,d)
    for (i,d), tau in active_t.items():
        if (i,d) not in U:
            t[i,d].LB = tau
            t[i,d].UB = tau
            # optionally for s
            s[i,d].LB = 1 if (i,d) in active_t else 0
            s[i,d].UB = 1 if (i,d) in active_t else 0

    # For each (d,w) nurse route, fix arcs for nodes not destroyed
    for (d,w), r in inc_sol.day_routes.items():
        nodes = r.nodes
        # predecessor/ successor on this nurse’s route
        succ = {nodes[k]: nodes[k+1] for k in range(len(nodes)-1)}
        pred = {nodes[k+1]: nodes[k] for k in range(len(nodes)-1)}

        for v in r.nodes:
            vi = _to_mip_node(v, w, m)
            if 0 <= v < m:        # event node
                if (v,d) in U:
                    continue      # leave free
                # lock membership: if this nurse attends v on day d in incumbent, keep it;
                # if not, forbid any incidence at (v,d) for this nurse.
                if w in team.get((v,d), set()):
                    ip = _to_mip_node(pred[v], w, m)
                    js = _to_mip_node(succ[v], w, m)
                    x[ip, vi, d, w].LB = x[ip, vi, d, w].UB = 1.0
                    x[vi, js, d, w].LB = x[vi, js, d, w].UB = 1.0
                    for k in cand_in[(d,w,vi)]:
                        if k != ip: x[k, vi, d, w].UB = 0.0
                    for k in cand_out[(d,w,vi)]:
                        if k != js: x[vi, k, d, w].UB = 0.0
                else:
                    # nurse w is NOT on team for (v,d): forbid any arcs for (v,d,w)
                    for k in cand_in [(d,w,vi)]: x[k, vi, d, w].UB = 0.0
                    for k in cand_out[(d,w,vi)]: x[vi, k, d, w].UB = 0.0
            else:
                # sentinels
                if keep_route_anchor:
                    ip = pred.get(v)
                    js = succ.get(v)
                    if ip is not None:
                        ii = _to_mip_node(ip, w, m); vv = _to_mip_node(v, w, m)
                        x[ii, vv, d, w].LB = x[ii, vv, d, w].UB = 1.0
                    if js is not None:
                        vv = _to_mip_node(v, w, m); jj = _to_mip_node(js, w, m)
                        x[vv, jj, d, w].LB = x[vv, jj, d, w].UB = 1.0

def _hard_fix_multinurse_compress(
    model, x, s, t, inc_sol, active_t, U, pd, cand_in, cand_out,
    time_band=0, fix_s=True
):
    m, D = pd.m, pd.day

    # Fix times (and s) for NON-destroyed (i,d)
    for (i,d), tau in active_t.items():
        if (i,d) in U: 
            continue
        if time_band == 0:
            t[i,d].LB = tau; t[i,d].UB = tau
        else:
            t[i,d].LB = max(t[i,d].LB, tau - time_band)
            t[i,d].UB = min(t[i,d].UB, tau + time_band)
        if fix_s:
            s[i,d].LB = 1; s[i,d].UB = 1
            for dd in range(D):
                if dd != d:
                    s[i,dd].LB = 0; s[i,dd].UB = 0

    # Compress each (d,w) route and fix skeleton arcs (skip self-arcs)
    for (d,w), r in inc_sol.day_routes.items():
        kept = []
        for v in r.nodes:
            vi = _to_mip_node(v, w, m)
            if 0 <= vi < m and (vi,d) in U:
                continue
            kept.append(vi)

        for a, b in zip(kept[:-1], kept[1:]):
            # ia, jb = _to_mip_node(a, w, m), _to_mip_node(b, w, m)
            ia, jb = a, b
            if ia == jb:  # skip self-arc like HOME->HOME
                continue
            # fix a->b
            x[ia,jb,d,w].LB = 1.0; x[ia,jb,d,w].UB = 1.0
            # prune other choices at endpoints
            for k in cand_out.get((d,w,ia), []):
                if k != jb: x[ia,k,d,w].UB = 0.0
            for k in cand_in.get((d,w,jb), []):
                if k != ia: x[k,jb,d,w].UB = 0.0


    # 3) IMPORTANT: we do NOT touch arcs incident to destroyed nodes (i,d)∈U.
    #     Gurobi is free to reinsert them anywhere (and choose attending nurses)
    #     consistent with your RN/LVN minima and other constraints.


def _warm_start_from_solution(x, inc_sol, pd):
    m = pd.m
    for (d,w), r in inc_sol.day_routes.items():
        for a,b in zip(r.nodes[:-1], r.nodes[1:]):
            ii = _to_mip_node(a, w, m); jj = _to_mip_node(b, w, m)
            try:
                x[ii, jj, d, w].Start = 1.0
            except KeyError:
                pass


def apply_safe_warm_start(model, x, s, t, alpha, beta,
                          inc_sol, active_t, U, pd,
                          start_x_skeleton=True):
    m, D = pd.m, pd.day

    # 0) Clear any stale starts
    for v in model.getVars():
        v.Start = GRB.UNDEFINED

    # 1) Start times for NON-destroyed (i,d)
    for (i,d), tau in active_t.items():
        if (i,d) in U:   # destroyed: leave free
            continue
        if t[i,d].LB <= tau <= t[i,d].UB:
            t[i,d].Start = tau

    # 2) Start s for NON-destroyed (i,d): 1 on incumbent day, 0 otherwise
    for (i,d), tau in active_t.items():
        s[i,d].Start = 1
        for dd in range(D):
            if dd != d:
                s[i,dd].Start = 0

    # 3) Optionally start skeleton arcs only (never touch arcs of destroyed nodes)
    if start_x_skeleton:
        def to_mip(v,w): return m if v == 100 + w else v
        for (d,w), r in inc_sol.day_routes.items():
            kept = [v for v in r.nodes if not (0 <= v < m and (v,d) in U)]
            for a,b in zip(kept[:-1], kept[1:]):
                ii, jj = to_mip(a,w), to_mip(b,w)
                if (ii,jj,d,w) in x.keys():
                    x[ii,jj,d,w].Start = 1.0
            # leave all other x[*] Start undefined

    # 4) Leaders (common source of “violates by 1.0”)
    # For each event i, find its incumbent day d and team; choose one w on that team.
    # Set alpha[i,d,w]=1, beta[i,d,w]=1; others 0 on that i across all days.
    team_by_id = {}  # (i,d) -> set of nurses on incumbent team
    for (d,w), r in inc_sol.day_routes.items():
        for v in r.nodes:
            if 0 <= v < m:
                team_by_id.setdefault((v,d), set()).add(w)

    for (i,d), team in team_by_id.items():
        pick = min(team)  # any attending nurse is OK
        alpha[i,d,pick].Start = 1
        beta[i,d,pick].Start  = 1
        for w in range(pd.n):
            if w != pick:
                alpha[i,d,w].Start = 0
                beta[i,d,w].Start  = 0
        # zero leaders on other days
        for dd in range(D):
            if dd != d:
                for w in range(pd.n):
                    alpha[i,dd,w].Start = 0
                    beta[i,dd,w].Start  = 0


def warm_start_fixed_only(model, tol=1e-9, use_add_mipstart=True, name="fixonly"):
    """
    Warm-start using only variables that are already fixed (LB == UB).
    If none are fixed, do nothing (avoid 'No start values specified' error).

    Call this *after* you've applied your compress-and-fix bounds.
    """
    fixed_vars, fixed_vals = [], []

    # Collect fixed vars (binary, integer, or continuous — doesn't matter)
    for v in model.getVars():
        lb, ub = v.LB, v.UB
        # treat tiny numerical gaps as fixed
        if lb > -GRB.INFINITY and ub < GRB.INFINITY and abs(ub - lb) <= tol:
            fixed_vars.append(v)
            fixed_vals.append(lb)  # equals ub

    if not fixed_vars:
        # Nothing to warm start — avoid addMIPStart([]) which triggers the warning
        print("No fixed vars to warm start")
        return 0

    if use_add_mipstart:
        # Optional: clear any old MIP starts if your code adds them elsewhere
        try:
            # Gurobi >= 10 has removeMIPStart; older versions may not
            for k in range(getattr(model, "NumStart", 0) - 1, -1, -1):
                model.removeMIPStart(k)
        except Exception:
            pass

        model.addMIPStart(fixed_vars, fixed_vals, name=name)
    else:
        # Alternative: set Start attribute directly (no MIPStart object created)
        for v, val in zip(fixed_vars, fixed_vals):
            v.Start = val

    return len(fixed_vars)



def _extract_active_x_t(model, x, t):
    # active arcs (i,j,d,w)
    active_x = [(i,j,d,w) for (i,j,d,w), var in x.items() if var.X > 0.5]
    # start times t[i,d] (you model a single start time per event-day)
    active_t = {(i,d): t[i,d].X for (i,d) in t.keys() if t[i,d].X > 0}
    return active_x, active_t


@dataclass
class LNSCfg:
    destroy_frac: float = 0.2            # 20% of scheduled events per iter
    iters: int = 100
    rng_seed: int = 0
    timelimit_s: float = 3.0             # time budget for the sub-MIP per iter
    threads: int = 8                     # 0=let Gurobi decide; else pin to cpus-per-task
    restrict_to_original_route: bool = False  # True = only reorder within same (d,w)



# lns_gurobi.py
def lns_with_gurobi(initial_sol, initial_active_t, pd, ctx, cfg):
    best_obj = None
    rng = random.Random(cfg.rng_seed)
    best_sol = deepcopy(initial_sol)
    best_active_t = dict(initial_active_t)   # <- keep times with the incumbent

    for it in range(1, cfg.iters + 1):
        print(f"Iteration {it}")
        U = _choose_destroy_eventdays(best_sol, pd, cfg.destroy_frac, rng)
        if not U:
            continue

        model, x, t, s, alpha, beta = build_full_model(pd)
        cand_in, cand_out = _cand_sets(pd, x)

        # _hard_fix_multinurse(
        #     model, x, s, t,
        #     inc_sol=best_sol,
        #     active_t=best_active_t,   # <- use the incumbent’s times
        #     U=U, pd=pd,
        #     cand_in=cand_in, cand_out=cand_out,
        #     keep_route_anchor=True
        # )
        _hard_fix_multinurse_compress(
            model, x, s, t,
            inc_sol=best_sol,
            active_t=best_active_t,
            U=U,
            pd=pd,
            cand_in=cand_in,
            cand_out=cand_out,
            time_band=15,          # start with 15; set 0 to freeze
            fix_s=True
        )
        # _warm_start_from_solution(x, best_sol, pd)
        # Only set starts for t and s; avoid x/alpha/beta starts
        # apply_safe_warm_start(model, x, s, t, alpha, beta, inc_sol=best_sol, active_t=best_active_t, U=U, pd=pd, start_x_skeleton=False)
        num_started = warm_start_fixed_only(model, tol=1e-9, use_add_mipstart=True)
        print(f"Warm-started {num_started} fixed vars")

        model.Params.MIPFocus = 1
        if cfg.threads: model.Params.Threads = cfg.threads
        model.Params.TimeLimit = cfg.timelimit_s
        model.optimize()

        if model.SolCount == 0:
            continue

        EPS = 1e-6
        if model.SolCount > 0 and model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.WORK_LIMIT, GRB.INTERRUPTED):
            active_x, active_t = _extract_active_x_t(model, x, t)
            cand_sol = routes_from_active_x_t(active_x, active_t, pd)
            if getattr(ctx.check_solution(cand_sol), "feasible", True):
                cand_obj = float(model.ObjVal)  # best feasible found
                print("Objective value:", cand_obj)
                if best_obj is None or cand_obj < best_obj - EPS:
                    best_sol = cand_sol
                    best_obj = cand_obj

    return best_sol, best_obj


def list_all_events(sol, m: int) -> List[Tuple[Tuple[int,int], int, int]]:
    """((d,w), idx_in_route, node) for nodes 0..m-1 (excludes sentinels)."""
    out = []
    for key, r in sol.day_routes.items():
        for k, node in enumerate(r.nodes):
            if 0 <= node < m:
                out.append((key, k, node))
    return out


def eval_cost_from_model(model) -> float:
    try:
        return float(model.ObjVal)
    except Exception:
        return float("inf")