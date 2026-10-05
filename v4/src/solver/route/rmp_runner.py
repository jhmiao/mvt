# src/solver/route/rmp_runner.py
"""
Route-based RMP runner (NO column generation yet).

This runner:
  1) builds event-time copies
  2) builds a static route pool (idle + one-, two-, and optional three-event routes)
  3) builds the Restricted Master Problem (RMP)
  4) solves it with Gurobi
  5) returns (model, ctx) for extraction

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .copies import CopyIndex, build_event_copies
from .pool_builder import build_route_pool, RoutePoolConfig
from .routes import Route
from .master_builder import (
    RouteMasterContext,
    build_master_model,
    set_weekly_leader_day_objective,
    set_weekly_workload_objective,
)

from src.solver.config import SolverConfig
from src.structures.problem_data import ProblemData


# -----------------------------
# Result container
# -----------------------------
@dataclass
class RmpSolveResult:
    model: gp.Model
    ctx: RouteMasterContext

    status: int
    runtime: float
    obj_val: Optional[float]
    obj_bound: Optional[float]
    mip_gap: Optional[float]
    node_count: Optional[int]


@dataclass(frozen=True)
class PreparedRouteRmp:
    """Immutable route-RMP inputs that can be shared by repeated solves.

    The contained route lists must be treated as read-only so route indices stay
    stable across repairs. Each solve still creates its own Gurobi model and
    adds its own temporary repair constraints.
    """

    copy_index: CopyIndex
    pool: Dict[Tuple[int, int], list[Route]]


def prepare_route_rmp(
    problem: ProblemData,
    pool_cfg: RoutePoolConfig,
) -> PreparedRouteRmp:
    """Build the reusable event-copy and route-pool inputs for one problem."""
    copy_index = build_event_copies(problem)
    pool = build_route_pool(problem, copy_index, pool_cfg)
    return PreparedRouteRmp(copy_index=copy_index, pool=pool)


# -----------------------------
# Public API
# -----------------------------
def solve_rmp_routes(
    problem: ProblemData,
    pool_cfg: RoutePoolConfig,
    time_limit: Optional[float] = None,
    work_limit: Optional[float] = None,
    seed: int = 0,
    outputflag: int = 1,
    threads: Optional[int] = None,
    solution_limit: Optional[int] = None,
    forbidden_route_keys: Optional[Iterable[Tuple[int, int, int]]] = None,
    forbid_depot_routes_by_nurse_day: Optional[Iterable[Tuple[int, int]]] = None,
    forbid_routes_with_work_at_least_by_nurse_day: Optional[Dict[Tuple[int, int], float]] = None,
    maximum_work_by_nurse_day: Optional[Dict[Tuple[int, int], float]] = None,
    weekly_workload_objective: bool = False,
    prior_travel_cost: float = 0.0,
    prior_workloads_by_nurse: Optional[Iterable[float]] = None,
    weekly_workload_penalty_coefficient: float = 1.0,
    weekly_leader_day_objective: bool = False,
    prior_leader_days_by_nurse: Optional[Iterable[float]] = None,
    weekly_leader_day_penalty_coefficient: float = 1.0,
    stop_when_objective_below: Optional[float] = None,
    prepared: Optional[PreparedRouteRmp] = None,
) -> RmpSolveResult:
    """
    Build and solve a route-based Restricted Master Problem (RMP) once.

    Args:
      problem: ProblemData
      pool_cfg: RoutePoolConfig controlling pool size/content, including optional
        three-event skeleton generation
      time_limit: seconds
      work_limit: Gurobi work units
      seed: Gurobi Seed
      outputflag: 0/1
      threads: optional
      solution_limit: optional cap on the number of feasible MIP solutions;
        use ``1`` for a first-feasible repair.
      forbidden_route_keys: route variables to set to zero before solving.  This
        is intentionally a small hook for destroy-and-repair heuristics; keys
        use the RMP-local ``(nurse, day, route_index)`` convention.
      forbid_depot_routes_by_nurse_day: nurse-days whose depot-using route
        variables are all set to zero.
      forbid_routes_with_work_at_least_by_nurse_day: route-pool destroy
        thresholds; routes at or above the supplied workload are set to zero.
      maximum_work_by_nurse_day: optional upper bounds on a nurse-day's route
        workload, keyed by RMP-local ``(nurse, day)``.
      weekly_workload_objective: replace the normal route-cost objective with
        weekly travel plus total absolute deviation from mean workload.
      prior_travel_cost/prior_workloads_by_nurse: incumbent contributions from
        all other source days for that weekly objective.
      weekly_workload_penalty_coefficient: weekly workload-deviation weight.
      weekly_leader_day_objective: replace the normal route-cost objective with
        weekly travel plus total absolute leader-day deviation from the mean.
      stop_when_objective_below: stop at the first feasible MIP solution whose
        objective is strictly below this value.
      prepared: optional reusable event-copy and route-pool inputs. When
        omitted, they are built for this solve.

    Returns:
      RmpSolveResult containing (model, ctx) and solver metadata.
    """
    if weekly_workload_objective and weekly_leader_day_objective:
        raise ValueError("weekly workload and leader-day objectives are mutually exclusive")

    if prepared is None:
        prepared = prepare_route_rmp(problem, pool_cfg)
    copy_index = prepared.copy_index
    pool = prepared.pool

    # 3) Build the integer RMP
    solver_config = SolverConfig(
        backend="route_pool",
        seed=seed,
        gurobi_outputflag=outputflag,
        time_limit=time_limit,
        work_limit=work_limit,
        route_workload_fairness=pool_cfg.workload_fairness,
        route_workload_fairness_penalty_coefficient=pool_cfg.workload_fairness_penalty_coefficient,
        # The master requires each scheduled event copy to be covered by at
        # least one selected depot-using route.
        include_depot=True,
    )
    model, ctx = build_master_model(problem, copy_index, pool, solver_config)

    if forbidden_route_keys:
        _forbid_routes(model, ctx, forbidden_route_keys)
    if forbid_depot_routes_by_nurse_day:
        _forbid_depot_routes(model, ctx, forbid_depot_routes_by_nurse_day)
    if forbid_routes_with_work_at_least_by_nurse_day:
        _forbid_routes_with_work_at_least(
            model, ctx, forbid_routes_with_work_at_least_by_nurse_day
        )
    if maximum_work_by_nurse_day:
        _limit_nurse_day_work(model, ctx, maximum_work_by_nurse_day)
    if weekly_workload_objective:
        if prior_workloads_by_nurse is None:
            raise ValueError("prior_workloads_by_nurse is required for the weekly workload objective")
        set_weekly_workload_objective(
            model, problem, ctx.pool, ctx.z, prior_travel_cost,
            list(prior_workloads_by_nurse), weekly_workload_penalty_coefficient,
        )
    if weekly_leader_day_objective:
        if prior_leader_days_by_nurse is None:
            raise ValueError("prior_leader_days_by_nurse is required for the weekly leader-day objective")
        set_weekly_leader_day_objective(
            model, problem, ctx.pool, ctx.z, prior_travel_cost,
            list(prior_leader_days_by_nurse), weekly_leader_day_penalty_coefficient,
        )

    # 4) Set solver params and solve
    _apply_gurobi_params(model, time_limit, work_limit, seed, outputflag, threads, solution_limit)

    # -------------------------
    # maybe TODO (Column Generation):
    # -------------------------
    # - Change RMP to LP (y,z continuous in [0,1]) to obtain valid duals:
    #     * either build vars as CONTINUOUS initially, or
    #     * call model.relax() here and solve the relaxed model
    # - Read duals from key constraints (staffing, event_once, depot, etc.)
    # - Pricing: generate new routes (columns) with negative reduced cost
    # - Add new z vars and update constraints incrementally
    # - Iterate until convergence
    # - Then solve the final integer master with accumulated columns
    #
    # For now: just solve the built model once.

    if stop_when_objective_below is None:
        model.optimize()
    else:
        model._stop_when_objective_below = float(stop_when_objective_below)
        model.optimize(_stop_at_first_improving_solution)

    # 5) Collect solver metadata
    status = int(model.Status)
    runtime = float(getattr(model, "Runtime", 0.0))

    obj_val = _safe_float(getattr(model, "ObjVal", None)) if _has_primal(status) else None
    obj_bound = _safe_float(getattr(model, "ObjBound", None)) if _has_bound(status) else None
    mip_gap = _safe_float(getattr(model, "MIPGap", None)) if _has_bound(status) else None
    node_count = int(getattr(model, "NodeCount", 0)) if hasattr(model, "NodeCount") else None

    return RmpSolveResult(
        model=model,
        ctx=ctx,
        status=status,
        runtime=runtime,
        obj_val=obj_val,
        obj_bound=obj_bound,
        mip_gap=mip_gap,
        node_count=node_count,
    )


def _forbid_routes(
    model: gp.Model,
    ctx: RouteMasterContext,
    forbidden_route_keys: Iterable[Tuple[int, int, int]],
) -> None:
    """Add route-level destroy constraints, validating keys eagerly."""
    for w, d, k in forbidden_route_keys:
        key = (int(w), int(d), int(k))
        if key not in ctx.z:
            raise ValueError(f"Cannot forbid route not present in the RMP pool: {key}")
        model.addConstr(ctx.z[key] == 0, name=f"forbid_route[w{key[0]},d{key[1]},k{key[2]}]")


def _forbid_depot_routes(
    model: gp.Model,
    ctx: RouteMasterContext,
    nurse_days: Iterable[Tuple[int, int]],
) -> None:
    """Forbid every depot-using route for each requested RMP-local nurse-day."""
    for w, d in nurse_days:
        key = (int(w), int(d))
        routes = ctx.pool.get(key)
        if routes is None:
            raise ValueError(f"Cannot forbid depot routes for nurse-day not present in the RMP pool: {key}")
        for k, route in enumerate(routes):
            if route.depot_incl:
                model.addConstr(ctx.z[(key[0], key[1], k)] == 0, name=f"forbid_depot_route[w{key[0]},d{key[1]},k{k}]")


def _forbid_routes_with_work_at_least(
    model: gp.Model,
    ctx: RouteMasterContext,
    minimum_work_by_nurse_day: Dict[Tuple[int, int], float],
) -> None:
    """Temporarily remove high-workload routes from a repair neighborhood."""
    for (w, d), minimum_work in minimum_work_by_nurse_day.items():
        key = (int(w), int(d))
        routes = ctx.pool.get(key)
        if routes is None:
            raise ValueError(f"Cannot restrict routes for nurse-day not present in the RMP pool: {key}")
        for k, route in enumerate(routes):
            if float(route.work) >= float(minimum_work):
                model.addConstr(
                    ctx.z[(key[0], key[1], k)] == 0,
                    name=f"forbid_high_work_route[w{key[0]},d{key[1]},k{k}]",
                )


def _limit_nurse_day_work(
    model: gp.Model,
    ctx: RouteMasterContext,
    maximum_work_by_nurse_day: Dict[Tuple[int, int], float],
) -> None:
    """Add optional workload-reduction constraints for a repair neighborhood."""
    for (w, d), maximum_work in maximum_work_by_nurse_day.items():
        routes = ctx.pool.get((int(w), int(d)))
        if routes is None:
            raise ValueError(f"Cannot limit work for nurse-day not present in the RMP pool: {(w, d)}")
        model.addConstr(
            gp.quicksum(route.work * ctx.z[(int(w), int(d), k)] for k, route in enumerate(routes))
            <= float(maximum_work),
            name=f"max_repair_work[w{int(w)},d{int(d)}]",
        )


def _stop_at_first_improving_solution(model: gp.Model, where: int) -> None:
    """Terminate after the first incumbent that improves the supplied target."""
    if where == GRB.Callback.MIPSOL and model.cbGet(GRB.Callback.MIPSOL_OBJ) < model._stop_when_objective_below:
        model.terminate()


# -----------------------------
# Param helpers
# -----------------------------
def _apply_gurobi_params(
    model: gp.Model,
    time_limit: Optional[float],
    work_limit: Optional[float],
    seed: int,
    outputflag: int,
    threads: Optional[int],
    solution_limit: Optional[int],
) -> None:
    model.Params.OutputFlag = int(outputflag)
    model.Params.Seed = int(seed)

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if work_limit is not None:
        model.Params.WorkLimit = float(work_limit)
    if threads is not None:
        model.Params.Threads = int(threads)
    if solution_limit is not None:
        if solution_limit < 1:
            raise ValueError("solution_limit must be at least one when provided")
        model.Params.SolutionLimit = int(solution_limit)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _has_primal(status: int) -> bool:
    return status in {
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
        GRB.TIME_LIMIT,
        GRB.WORK_LIMIT,
        GRB.NODE_LIMIT,
        GRB.SOLUTION_LIMIT,
        GRB.INTERRUPTED,
        GRB.USER_OBJ_LIMIT,
    }


def _has_bound(status: int) -> bool:
    return status in {
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
        GRB.TIME_LIMIT,
        GRB.WORK_LIMIT,
        GRB.NODE_LIMIT,
        GRB.SOLUTION_LIMIT,
        GRB.INTERRUPTED,
        GRB.USER_OBJ_LIMIT,
    }
