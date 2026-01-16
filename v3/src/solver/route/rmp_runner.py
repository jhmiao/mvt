# src/solver/route/rmp_runner.py
"""
Route-based RMP runner (NO column generation yet).

This runner:
  1) builds event-time copies
  2) builds a (static) route pool (idle + single/two-event, etc.)
  3) builds the Restricted Master Problem (RMP)
  4) solves it with Gurobi
  5) returns (model, ctx) for extraction

Future (TODO):
  - Column generation loop:
      * solve RMP (LP relaxation)
      * read duals
      * pricing subproblem generates new routes (columns)
      * add columns to RMP (new z vars + update cover indices)
      * repeat until no negative reduced-cost columns
  - Then (optionally) solve final integer master (branch-and-price or fix columns).

Keep this file small and backend-specific for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from .copies import build_event_copies  # expected to return CopyIndex
from .pool_builder import build_route_pool, RoutePoolConfig
from .master_builder import build_master_model, RouteMasterContext

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
) -> RmpSolveResult:
    """
    Build and solve a route-based Restricted Master Problem (RMP) once.

    Args:
      problem: ProblemData
      pool_cfg: RoutePoolConfig controlling pool size/content
      time_limit: seconds
      work_limit: Gurobi work units
      seed: Gurobi Seed
      outputflag: 0/1
      threads: optional

    Returns:
      RmpSolveResult containing (model, ctx) and solver metadata.
    """
    # 1) Build event-time copies (V) + groupings
    copy_index = build_event_copies(problem)  # returns CopyIndex

    # 2) Build static route pool (columns)
    pool = build_route_pool(problem, copy_index, pool_cfg)

    # 3) Build RMP MILP
    # NOTE: For column generation later, you will first solve the LP relaxation.
    # For now, build the integer RMP (z,y binary).
    dummy_solver_config = _dummy_solver_config(
        backend="route_pool",
        seed=seed,
        outputflag=outputflag,
        time_limit=time_limit,
        work_limit=work_limit,
        threads=threads,
        enforce_depot=getattr(pool_cfg, "force_depot_ok_for_working_routes", False),
    )
    model, ctx = build_master_model(problem, copy_index, pool, dummy_solver_config)

    # 4) Set solver params and solve
    _apply_gurobi_params(model, time_limit, work_limit, seed, outputflag, threads)

    # -------------------------
    # TODO (Column Generation):
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

    model.optimize()

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
) -> None:
    model.Params.OutputFlag = int(outputflag)
    model.Params.Seed = int(seed)

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if work_limit is not None:
        model.Params.WorkLimit = float(work_limit)
    if threads is not None:
        model.Params.Threads = int(threads)


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


# -----------------------------
# Temporary solver config stub
# -----------------------------
@dataclass
class _DummySolverConfig:
    backend: str = "route_pool"
    seed: int = 0
    gurobi_outputflag: int = 1
    time_limit: Optional[float] = None
    work_limit: Optional[float] = None
    threads: Optional[int] = None

    # feature flags used by master_builder skeleton
    enforce_depot: bool = False
    enforce_hour_balance: bool = False
    enforce_max_hours: bool = False


def _dummy_solver_config(
    backend: str,
    seed: int,
    outputflag: int,
    time_limit: Optional[float],
    work_limit: Optional[float],
    threads: Optional[int],
    enforce_depot: bool,
) -> _DummySolverConfig:
    return _DummySolverConfig(
        backend=backend,
        seed=seed,
        gurobi_outputflag=outputflag,
        time_limit=time_limit,
        work_limit=work_limit,
        threads=threads,
        enforce_depot=enforce_depot,
    )
