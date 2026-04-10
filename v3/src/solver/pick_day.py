from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.solver.config import SolverConfig
from src.solver.model_builder import build_model
from src.solver.solver_runner import optimize_model
from src.structures.problem_data import ProblemData


@dataclass
class PickDayResult:
    """Container for LP-relaxation-based day recommendations."""

    s_lp: np.ndarray  # shape (m, days), LP values of s[i,d]
    best_day_by_event: np.ndarray  # shape (m,), argmax day index or -1 if no support
    status: int
    objective_value: Optional[float]


@dataclass
class IterativePickDayResult:
    """Result for iterative LP-based fixing of s[i,d]."""

    final_s_lp: np.ndarray
    fixed_day_by_event: np.ndarray
    fixed_reason_by_event: list[str]
    iterations: int
    status: int
    objective_value: Optional[float]
    s_lp_history: list[np.ndarray]


def _parse_s_indices(var_name: str) -> Optional[tuple[int, int]]:
    if not var_name.startswith("s[") or not var_name.endswith("]"):
        return None
    inside = var_name[2:-1]
    parts = inside.split(",")
    if len(parts) != 2:
        return None
    try:
        i = int(parts[0])
        d = int(parts[1])
    except ValueError:
        return None
    return i, d


def _extract_s_lp(
    mip_model: gp.Model,
    model: gp.Model,
) -> np.ndarray:
    """
    Read s[i,d] values from a solved LP model.

    Dimensions are inferred from the model variable names so this works for
    sampled/subset instances where total events differ from the original data file.
    """
    mip_vars = mip_model.getVars()
    lp_vars = model.getVars()

    s_positions: list[tuple[int, int, int]] = []
    max_i = -1
    max_d = -1
    for pos, mip_var in enumerate(mip_vars):
        idx = _parse_s_indices(mip_var.VarName)
        if idx is None:
            continue
        i, d = idx
        s_positions.append((pos, i, d))
        if i > max_i:
            max_i = i
        if d > max_d:
            max_d = d

    if not s_positions:
        raise RuntimeError("No s[i,d] variables found in model.")

    s_lp = np.zeros((max_i + 1, max_d + 1), dtype=float)
    target_count = len(s_positions)

    # Primary extraction: align LP vars to MIP var positions.
    if len(mip_vars) == len(lp_vars):
        for pos, i, d in s_positions:
            s_lp[i, d] = lp_vars[pos].X
        return s_lp

    # Fallback: parse names directly from solved LP vars.
    extracted = 0
    for var in lp_vars:
        idx = _parse_s_indices(var.VarName)
        if idx is None:
            continue
        i, d = idx
        if i <= max_i and d <= max_d:
            s_lp[i, d] = var.X
        extracted += 1
    if extracted >= target_count:
        return s_lp

    raise RuntimeError(
        "Unable to extract s variables from LP model. "
        f"expected_s={target_count}, mip_vars={len(mip_vars)}, lp_vars={len(lp_vars)}, extracted={extracted}."
    )


def _build_s_var_map(model: gp.Model) -> dict[tuple[int, int], gp.Var]:
    model.update()
    s_var_map: dict[tuple[int, int], gp.Var] = {}
    for var in model.getVars():
        idx = _parse_s_indices(var.VarName)
        if idx is None:
            continue
        s_var_map[idx] = var
    if not s_var_map:
        raise RuntimeError("No s[i,d] variables found in MIP model.")
    return s_var_map


def _solve_lp_relaxation(mip_model: gp.Model) -> tuple[gp.Model, np.ndarray]:
    mip_model.update()
    lp_model = mip_model.relax()
    optimize_model(lp_model)
    if lp_model.SolCount == 0 or lp_model.Status in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}:
        raise RuntimeError(f"LP relaxation did not return a feasible solution. status={lp_model.Status}")
    return lp_model, _extract_s_lp(mip_model, lp_model)


def iterative_pick_day_by_lp_relaxation(
    problem_data: ProblemData,
    solver_config: SolverConfig,
    *,
    fix_threshold: float = 0.99,
    print_s_lp_each_iteration: bool = True,
) -> IterativePickDayResult:
    """
    Iteratively fix s[i,d] from LP-relaxation only for high-confidence rows:
    fix i to argmax_d s[i,d] when max_d s[i,d] > fix_threshold.
    Stop when no additional events satisfy the threshold.
    """
    mip_model = build_model(problem_data, solver_config)
    s_var_map = _build_s_var_map(mip_model)

    max_i = max(i for i, _ in s_var_map.keys())
    total_events = max_i + 1
    fixed_day = np.full(total_events, -1, dtype=int)
    fixed_reason = [""] * total_events
    s_lp_history: list[np.ndarray] = []

    iteration = 0
    while True:
        iteration += 1
        lp_model, s_lp = _solve_lp_relaxation(mip_model)
        s_lp_history.append(s_lp.copy())

        if print_s_lp_each_iteration:
            print(f"Iteration {iteration} - s_lp:")
            print(s_lp)

        unfixed = [i for i in range(total_events) if fixed_day[i] < 0]
        to_fix: list[tuple[int, int, str]] = []
        for i in unfixed:
            row = s_lp[i]
            d = int(np.argmax(row))
            best_val = float(row[d])
            if best_val > fix_threshold:
                to_fix.append((i, d, f"threshold_gt_{fix_threshold}"))

        if not to_fix:
            break

        for i, d, reason in to_fix:
            if fixed_day[i] >= 0:
                continue
            var = s_var_map.get((i, d))
            if var is None:
                raise RuntimeError(f"Missing s variable for index ({i}, {d}).")
            mip_model.addConstr(var == 1, name=f"fix_s_i{i}_d{d}_iter{iteration}")
            fixed_day[i] = d
            fixed_reason[i] = reason

    # Final re-solve with all fixed s[i,d] constraints.
    final_lp_model, final_s_lp = _solve_lp_relaxation(mip_model)
    s_lp_history.append(final_s_lp.copy())
    if print_s_lp_each_iteration:
        print(f"Iteration {iteration + 1} (final) - s_lp:")
        print(final_s_lp)

    obj_val: Optional[float]
    try:
        obj_val = float(final_lp_model.ObjVal)
    except gp.GurobiError:
        obj_val = None

    return IterativePickDayResult(
        final_s_lp=final_s_lp,
        fixed_day_by_event=fixed_day,
        fixed_reason_by_event=fixed_reason,
        iterations=iteration,
        status=int(final_lp_model.Status),
        objective_value=obj_val,
        s_lp_history=s_lp_history,
    )


def pick_day_by_lp_relaxation(problem_data: ProblemData, solver_config: SolverConfig) -> PickDayResult:
    """
    Build the original MIP model, solve its LP relaxation, and recommend a day for each event.

    Returns:
        PickDayResult containing LP values for s[i,d] and an argmax day recommendation per event.
    """
    if problem_data.total_day <= 0:
        raise ValueError("problem_data.total_day must be positive")

    mip_model = build_model(problem_data, solver_config)
    lp_model, s_lp = _solve_lp_relaxation(mip_model)
    best_day = np.full(s_lp.shape[0], -1, dtype=int)

    for i in range(s_lp.shape[0]):
        row = s_lp[i]
        max_val = float(np.max(row))
        if max_val > 1e-9:
            best_day[i] = int(np.argmax(row))

    obj_val: Optional[float]
    try:
        obj_val = float(lp_model.ObjVal)
    except gp.GurobiError:
        obj_val = None

    return PickDayResult(
        s_lp=s_lp,
        best_day_by_event=best_day,
        status=int(lp_model.Status),
        objective_value=obj_val,
    )


def s_id_from_lp(problem_data: ProblemData, solver_config: SolverConfig) -> np.ndarray:
    """Compatibility helper returning only the recommended day index per event."""
    result = pick_day_by_lp_relaxation(problem_data, solver_config)
    return result.best_day_by_event
