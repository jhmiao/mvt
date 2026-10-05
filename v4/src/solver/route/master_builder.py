# src/solver/route/master_builder.py
"""
Restricted Master Problem (RMP) builder for route-based formulation.

Inputs:
  - problem: ProblemData (costs, durations, nurse types, staffing reqs, etc.)
  - copy_index: CopyIndex (event-time copies V and groupings)
  - pool: dict[(w,d)] -> list[Route] (pre-generated route pool)
  - config: SolverConfig (time limit, fairness flags, etc.)

Outputs:
  - model: gurobipy.Model
  - ctx: RouteMasterContext (y, z, pool, indices, etc.)

Notes:
  - This file builds the optimization model only. Solving should be done in solver_runner.py.
  - A_{v,wdr} is represented sparsely via cover indices: cover[v] = list of (w,d,k).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from .copies import CopyIndex, EventCopy, RouteVisits
from .routes import Route
from src.solver.config import SolverConfig
from src.structures.problem_data import ProblemData


# -----------------------------
# Context returned to solver_runner / solution extractor
# -----------------------------
@dataclass
class RouteMasterContext:
    copy_index: CopyIndex
    pool: Dict[Tuple[int, int], List[Route]]

    # decision vars
    y: Dict[EventCopy, gp.Var]                    # schedule copy chosen
    z: Dict[Tuple[int, int, int], gp.Var]         # (w,d,r) route selection

    # sparse A-indices
    rn_cover: Dict[EventCopy, List[Tuple[int, int, int]]]
    lvn_cover: Dict[EventCopy, List[Tuple[int, int, int]]]
    depot_cover: Optional[Dict[EventCopy, List[Tuple[int, int, int]]]] = None


# -----------------------------
# Public API
# -----------------------------
def build_master_model(
    problem: ProblemData,
    copy_index: CopyIndex,
    pool: Dict[Tuple[int, int], List[Route]],
    config: SolverConfig,
) -> Tuple[gp.Model, RouteMasterContext]:
    """
    Build the route-based restricted master problem.
    Does NOT solve.

    Assumes:
      - pool[(w,d)] exists for all w,d
      - each pool[(w,d)] includes an idle route if you want sum_r r=1
      - Route.visits is an ordered sequence of EventCopy tuples (i,d,tau),
        including the supported one-, two-, and three-event routes
      - Route.cost/work/depot_incl are precomputed
    """
    model = gp.Model("route_rmp")
    model.Params.OutputFlag = getattr(config, "gurobi_outputflag", 1)

    # -------------------------
    # 1) Variables
    # -------------------------
    y = _add_y_vars(model, copy_index)
    z = _add_z_vars(model, pool)

    # -------------------------
    # 2) Precompute cover indices (sparse A)
    # -------------------------
    rn_cover, lvn_cover, depot_cover = _build_cover_indices(problem, copy_index, pool)

    # -------------------------
    # 3) Constraints
    # -------------------------
    _add_event_scheduled_once(model, copy_index, y)
    _add_one_route_per_nurse_day(model, problem, pool, z)

    _add_staffing_constraints(
        model=model,
        problem=problem,
        copy_index=copy_index,
        y=y,
        z=z,
        rn_cover=rn_cover,
        lvn_cover=lvn_cover,
    )

    # Every scheduled event must be covered by at least one depot-using route.
    if _should_enforce_depot(config):
        _add_depot_constraints(
            model=model,
            copy_index=copy_index,
            y=y,
            z=z,
            depot_cover=depot_cover,
        )

    # -------------------------
    # 4) Objective
    # -------------------------
    _set_objective(model, problem, pool, z, config)

    ctx = RouteMasterContext(
        copy_index=copy_index,
        pool=pool,
        y=y,
        z=z,
        rn_cover=rn_cover,
        lvn_cover=lvn_cover,
        depot_cover=depot_cover,
    )
    return model, ctx


# -----------------------------
# Variable builders
# -----------------------------
def _add_y_vars(model: gp.Model, copy_index: CopyIndex) -> Dict[EventCopy, gp.Var]:
    """y[v] = 1 if event-time copy v is selected."""
    y: Dict[EventCopy, gp.Var] = {}
    for v in copy_index.copies:
        i, d, tau = v
        y[v] = model.addVar(vtype=GRB.BINARY, name=f"y[i{i},d{d},t{tau}]")

    return y


def _add_z_vars(model: gp.Model, pool: Dict[Tuple[int, int], List[Route]]) -> Dict[Tuple[int, int, int], gp.Var]:
    """z[w,d,k] = 1 if nurse w chooses route k on day d."""
    z: Dict[Tuple[int, int, int], gp.Var] = {}
    for (w, d), routes in pool.items():
        for k in range(len(routes)):
            z[(w, d, k)] = model.addVar(vtype=GRB.BINARY, name=f"z[w{w},d{d},k{k}]")
    return z


# -----------------------------
# Cover indices: sparse A_{v,wdr}
# -----------------------------
def _build_cover_indices(
    problem: ProblemData,
    copy_index: CopyIndex,
    pool: Dict[Tuple[int, int], List[Route]],
) -> Tuple[
    Dict[EventCopy, List[Tuple[int, int, int]]],
    Dict[EventCopy, List[Tuple[int, int, int]]],
    Dict[EventCopy, List[Tuple[int, int, int]]],
]:
    """
    Build sparse cover indices:
      rn_cover[v] = list of (w,d,k) routes by RN nurses covering v
      lvn_cover[v] = list of (w,d,k) routes by LVN nurses covering v
      depot_cover[v] = list of (w,d,k) depot-using routes covering v

    Use route.visits directly for speed (avoid allocating r.covered repeatedly).
    The loop is intentionally route-length agnostic, so every stop in a
    three-event route contributes one coverage coefficient.
    """
    rn_cover = defaultdict(list)
    lvn_cover = defaultdict(list)
    depot_cover = defaultdict(list)

    rn_ids = set(range(problem.total_rn))
    for (w, d), routes in pool.items():
        is_rn = w in rn_ids
        for k, r in enumerate(routes):
            visits: RouteVisits = r.visits
            for v in visits:
                if is_rn:
                    rn_cover[v].append((w, d, k))
                else:
                    lvn_cover[v].append((w, d, k))
                if r.depot_incl:
                    depot_cover[v].append((w, d, k))

    return dict(rn_cover), dict(lvn_cover), dict(depot_cover)


# -----------------------------
# Constraints
# -----------------------------
def _add_event_scheduled_once(model: gp.Model, copy_index: CopyIndex, y: Dict[EventCopy, gp.Var]) -> None:
    """Each event i must select exactly one (d,tau)."""
    for i, copies_i in copy_index.copies_by_event.items():
        model.addConstr(gp.quicksum(y[v] for v in copies_i) == 1, name=f"event_once[i{i}]")


def _add_one_route_per_nurse_day(
    model: gp.Model,
    problem: ProblemData,
    pool: Dict[Tuple[int, int], List[Route]],
    z: Dict[Tuple[int, int, int], gp.Var],
) -> None:
    """Each nurse picks exactly one route per day (idle route should be included in pool)."""
    for w in range(problem.total_nurse):
        for d in range(problem.total_day):
            routes = pool.get((w, d), [])
            if not routes:
                raise ValueError(f"Pool missing routes for nurse {w}, day {d}")
            model.addConstr(
                gp.quicksum(z[(w, d, k)] for k in range(len(routes))) == 1,
                name=f"one_route[w{w},d{d}]",
            )


def _add_staffing_constraints(
    model: gp.Model,
    problem: ProblemData,
    copy_index: CopyIndex,
    y: Dict[EventCopy, gp.Var],
    z: Dict[Tuple[int, int, int], gp.Var],
    rn_cover: Dict[EventCopy, List[Tuple[int, int, int]]],
    lvn_cover: Dict[EventCopy, List[Tuple[int, int, int]]],
) -> None:
    """
    For each copy v=(i,d,tau):
      sum_{RN routes covering v} z >= minRN[i] * y[v]
      sum_{LVN routes covering v} z >= minLVN[i] * y[v]
    """
    
    min_rn = problem.min_nurses[:, 0] 
    min_lvn = problem.min_nurses[:, 1]

    for v in copy_index.copies:
        i, d, tau = v

        rn_list = rn_cover.get(v, [])
        lvn_list = lvn_cover.get(v, [])

        model.addConstr(
            gp.quicksum(z[key] for key in rn_list) >= min_rn[i] * y[v],
            name=f"staff_rn[i{i},d{d},t{tau}]",
        )
        model.addConstr(
            gp.quicksum(z[key] for key in lvn_list) >= min_lvn[i] * y[v],
            name=f"staff_lvn[i{i},d{d},t{tau}]",
        )


def _add_depot_constraints(
    model: gp.Model,
    copy_index: CopyIndex,
    y: Dict[EventCopy, gp.Var],
    z: Dict[Tuple[int, int, int], gp.Var],
    depot_cover: Dict[EventCopy, List[Tuple[int, int, int]]],
) -> None:
    """
    For each copy v:
      sum_{depot-using routes covering v} z >= y[v]

    Thus, the selected copy of every event is served by at least one selected
    route that goes through the depot.  Direct and depot-using versions are
    separate routes, so the cost-minimizing objective selects the mode.
    """
    for v in copy_index.copies:
        i, d, tau = v
        depot_list = depot_cover.get(v, [])
        model.addConstr(
            gp.quicksum(z[key] for key in depot_list) >= y[v],
            name=f"depot_cover[i{i},d{d},t{tau}]",
        )


# -----------------------------
# Objective
# -----------------------------
def _set_objective(
    model: gp.Model,
    problem: ProblemData,
    pool: Dict[Tuple[int, int], List[Route]],
    z: Dict[Tuple[int, int, int], gp.Var],
    config: SolverConfig,
) -> None:
    """Minimize route cost plus an optional max-minus-min workload penalty."""
    obj = gp.LinExpr()
    for (w, d), routes in pool.items():
        for k, r in enumerate(routes):
            obj += float(r.cost) * z[(w, d, k)]

    if config.route_workload_fairness:
        penalty_coefficient = float(
            config.route_workload_fairness_penalty_coefficient
        )
        if penalty_coefficient < 0:
            raise ValueError("workload_fairness_penalty_coefficient must be non-negative")

        workloads = [
            gp.quicksum(
                r.work * z[(w, d, k)]
                for d in range(problem.total_day)
                for k, r in enumerate(pool[(w, d)])
            )
            for w in range(problem.total_nurse)
        ]
        workload_max = model.addVar(lb=0.0, name="workload_max")
        workload_min = model.addVar(lb=0.0, name="workload_min")
        for w, workload in enumerate(workloads):
            model.addConstr(workload <= workload_max, name=f"workload_max_ge_w{w}")
            model.addConstr(workload >= workload_min, name=f"workload_min_le_w{w}")

        obj += penalty_coefficient * (workload_max - workload_min)

    model.setObjective(obj, GRB.MINIMIZE)


def set_weekly_workload_objective(
    model: gp.Model,
    problem: ProblemData,
    pool: Dict[Tuple[int, int], List[Route]],
    z: Dict[Tuple[int, int, int], gp.Var],
    prior_travel_cost: float,
    prior_workloads_by_nurse: List[float],
    workload_penalty_coefficient: float,
) -> None:
    """Replace the RMP objective with the full weekly workload objective.

    ``prior_*`` are the incumbent contributions of all source days other than
    this RMP's day.  Thus only this day's route variables remain decision
    variables, while Gurobi evaluates exactly the same objective used by the
    weekly heuristic acceptance rule.
    """
    if len(prior_workloads_by_nurse) != problem.total_nurse:
        raise ValueError("prior_workloads_by_nurse must contain one value per nurse")
    if workload_penalty_coefficient < 0:
        raise ValueError("workload_penalty_coefficient must be non-negative")

    day_travel = gp.quicksum(
        float(route.cost) * z[(w, d, k)]
        for (w, d), routes in pool.items()
        for k, route in enumerate(routes)
    )
    weekly_workloads = [
        float(prior_workloads_by_nurse[w])
        + gp.quicksum(
            float(route.work) * z[(w, d, k)]
            for d in range(problem.total_day)
            for k, route in enumerate(pool[(w, d)])
        )
        for w in range(problem.total_nurse)
    ]
    mean_workload = gp.quicksum(weekly_workloads) / float(problem.total_nurse)
    deviations = model.addVars(problem.total_nurse, lb=0.0, name="weekly_workload_deviation")
    for w, workload in enumerate(weekly_workloads):
        model.addConstr(deviations[w] >= workload - mean_workload, name=f"weekly_dev_pos_w{w}")
        model.addConstr(deviations[w] >= mean_workload - workload, name=f"weekly_dev_neg_w{w}")

    objective = (
        float(prior_travel_cost)
        + day_travel
        + float(workload_penalty_coefficient) * gp.quicksum(deviations[w] for w in range(problem.total_nurse))
    )
    model.setObjective(objective, GRB.MINIMIZE)


def set_weekly_leader_day_objective(
    model: gp.Model,
    problem: ProblemData,
    pool: Dict[Tuple[int, int], List[Route]],
    z: Dict[Tuple[int, int, int], gp.Var],
    prior_travel_cost: float,
    prior_leader_days_by_nurse: List[float],
    leader_day_penalty_coefficient: float,
) -> None:
    """Replace the RMP objective with weekly travel plus leader-day deviation.

    A nurse has one leader day when their selected route for that day uses the
    depot.  The one-route-per-nurse-day constraint makes the depot indicator
    binary without introducing an additional variable.
    """
    if len(prior_leader_days_by_nurse) != problem.total_nurse:
        raise ValueError("prior_leader_days_by_nurse must contain one value per nurse")
    if leader_day_penalty_coefficient < 0:
        raise ValueError("leader_day_penalty_coefficient must be non-negative")

    day_travel = gp.quicksum(
        float(route.cost) * z[(w, d, k)]
        for (w, d), routes in pool.items()
        for k, route in enumerate(routes)
    )
    weekly_leader_days = [
        float(prior_leader_days_by_nurse[w])
        + gp.quicksum(
            int(bool(route.depot_incl)) * z[(w, d, k)]
            for d in range(problem.total_day)
            for k, route in enumerate(pool[(w, d)])
        )
        for w in range(problem.total_nurse)
    ]
    mean_leader_days = gp.quicksum(weekly_leader_days) / float(problem.total_nurse)
    deviations = model.addVars(problem.total_nurse, lb=0.0, name="weekly_leader_day_deviation")
    for w, leader_days in enumerate(weekly_leader_days):
        model.addConstr(deviations[w] >= leader_days - mean_leader_days, name=f"weekly_leader_day_dev_pos_w{w}")
        model.addConstr(deviations[w] >= mean_leader_days - leader_days, name=f"weekly_leader_day_dev_neg_w{w}")

    objective = (
        float(prior_travel_cost)
        + day_travel
        + float(leader_day_penalty_coefficient)
        * gp.quicksum(deviations[w] for w in range(problem.total_nurse))
    )
    model.setObjective(objective, GRB.MINIMIZE)


# -----------------------------
# Feature switches
# -----------------------------
def _should_enforce_depot(config: SolverConfig) -> bool:
    # The route RMP enables this so every selected event has depot coverage.
    return config.include_depot
