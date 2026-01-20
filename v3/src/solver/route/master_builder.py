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

from .copies import CopyIndex, EventCopy
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
      - Route.visits contains only EventCopy tuples (i,d,tau)
      - Route.cost/work/depot_ok are precomputed
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

    # depot constraint (optional; skip until you’re ready)
    if _should_enforce_depot(config):
        _add_depot_constraints(
            model=model,
            copy_index=copy_index,
            y=y,
            z=z,
            depot_cover=depot_cover,
        )

    # fairness (optional; skeleton only)
    if _should_enforce_fairness(config):
        _add_fairness_constraints(
            model=model,
            problem=problem,
            pool=pool,
            z=z,
            config=config,
        )

    # -------------------------
    # 4) Objective
    # -------------------------
    _set_objective(model, pool, z, config)

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
      depot_cover[v] = list of (w,d,k) routes with depot_ok=1 covering v

    Use route.visits directly for speed (avoid allocating r.covered repeatedly).
    """
    rn_cover = defaultdict(list)
    lvn_cover = defaultdict(list)
    depot_cover = defaultdict(list)

    rn_ids = set(range(problem.total_rn))
    for (w, d), routes in pool.items():
        is_rn = w in rn_ids
        for k, r in enumerate(routes):
            for v in r.visits:
                if is_rn:
                    rn_cover[v].append((w, d, k))
                else:
                    lvn_cover[v].append((w, d, k))
                if r.depot_ok:
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
      sum_{routes covering v with depot_ok=1} z >= y[v]
    """
    for v in copy_index.copies:
        i, d, tau = v
        depot_list = depot_cover.get(v, [])
        model.addConstr(
            gp.quicksum(z[key] for key in depot_list) >= y[v],
            name=f"depot_cover[i{i},d{d},t{tau}]",
        )


def _add_fairness_constraints(
    model: gp.Model,
    problem: ProblemData,
    pool: Dict[Tuple[int, int], List[Route]],
    z: Dict[Tuple[int, int, int], gp.Var],
    config: SolverConfig,
) -> None:
    """
    Skeleton for weekly work-hour balance constraints.

    You likely have:
      - problem.total_day
      - Route.work (minutes)
      - target weekly minutes per nurse type (avg_RN_minutes / avg_LVN_minutes)

    Implement either hard bounds (0.8-1.2) or soft deviations.
    """
    # TODO: wire to your data structures
    # Example hard bounds:
    # rn_ids = set(problem.rn_ids)
    # avg_rn = problem.avg_rn_minutes
    # for w in rn_ids:
    #     work_w = gp.quicksum(pool[(w,d)][k].work * z[(w,d,k)]
    #                          for d in range(problem.total_day)
    #                          for k in range(len(pool[(w,d)])))
    #     model.addConstr(work_w >= 0.8 * avg_rn, name=f"rn_min_work[w{w}]")
    #     model.addConstr(work_w <= 1.2 * avg_rn, name=f"rn_max_work[w{w}]")
    pass


# -----------------------------
# Objective
# -----------------------------
def _set_objective(
    model: gp.Model,
    pool: Dict[Tuple[int, int], List[Route]],
    z: Dict[Tuple[int, int, int], gp.Var],
    config: SolverConfig,
) -> None:
    """Minimize total route cost (plus optional fairness penalties if you add them)."""
    obj = gp.LinExpr()
    for (w, d), routes in pool.items():
        for k, r in enumerate(routes):
            obj += float(r.cost) * z[(w, d, k)]
    model.setObjective(obj, GRB.MINIMIZE)


# -----------------------------
# Feature switches
# -----------------------------
def _should_enforce_depot(config: SolverConfig) -> bool:
    # Add a flag to SolverConfig later, e.g., config.enforce_depot
    return hasattr(config, "enforce_depot") and bool(getattr(config, "enforce_depot"))


def _should_enforce_fairness(config: SolverConfig) -> bool:
    # You already have enforce_hour_balance / enforce_max_hours, etc.
    return bool(getattr(config, "enforce_hour_balance", False)) or bool(getattr(config, "enforce_max_hours", False))
