from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.solutions.solutions import MergedSolution, Solution
from src.structures.problem_data import ProblemData


def _status_to_str(status_code: int) -> str:
    """Return a readable Gurobi status string."""
    status_lookup = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_lookup.get(status_code, str(status_code))


def extract_solution(model: gp.Model, problem_data: Optional[ProblemData] = None) -> MergedSolution:
    """
    Extract model variables into a MergedSolution using the Solution dataclass.
    Variable meanings (see build_model):
      - x[i,j,d,w] : nurse w goes from i to j on day d (i, j == m for home, m+1 depot_am, m+2 depot_pm)
      - s[i,d]     : event i is scheduled on day d
      - t[i,d]     : start time of event i on day d
      - alpha[i,d,w] / beta[i,d,w] : pick-up / drop-off leader flags
    """
    try:
        objective_value = float(model.ObjVal)
    except gp.GurobiError:
        objective_value = math.nan

    try:
        lower_bound = float(model.ObjBound)
    except gp.GurobiError:
        lower_bound = None

    status = _status_to_str(model.Status)

    t_vars: Dict[str, float] = {}
    x_vars: Dict[str, float] = {}
    s_vars: Dict[str, float] = {}
    alpha_vars: Dict[str, float] = {}
    beta_vars: Dict[str, float] = {}

    for var in model.getVars():
        val = var.X
        if abs(val) < 1e-6:
            continue
        name = var.VarName
        if name.startswith("t["):
            t_vars[name] = val
        elif name.startswith("x["):
            x_vars[name] = val
        elif name.startswith("s["):
            s_vars[name] = val
        elif name.startswith("alpha["):
            alpha_vars[name] = val
        elif name.startswith("beta["):
            beta_vars[name] = val

    def parse_indices(var_name: str) -> Tuple[int, ...]:
        inside = var_name[var_name.find("[") + 1 : var_name.find("]")]
        return tuple(int(p) for p in inside.split(","))

    # Infer event count from schedule variables to distinguish event nodes from home/depot.
    event_indices = [parse_indices(name)[0] for name in s_vars]
    event_count = max(event_indices) + 1 if event_indices else None
    # If problem_data provided (especially for day solves), include mapping info.
    original_event_ids = None
    day_index = None
    if problem_data is not None:
        original_event_ids = (
            [int(e) for e in problem_data.original_event_ids.tolist()]
            if problem_data.original_event_ids is not None
            else None
        )
        day_index = problem_data.day_index if problem_data.day_index != -1 else None

    day_data: DefaultDict[int, Dict[str, Any]] = defaultdict(
        lambda: {
            "assignments": defaultdict(set),
            "start_times": {},
            "routes": defaultdict(list),
            "leaders": defaultdict(dict),
            "scheduled": set(),
        }
    )

    for name, val in s_vars.items():
        i, d = parse_indices(name)
        day_info = day_data[d]
        if val >= 0.5:
            day_info["scheduled"].add(i)

    for name, val in t_vars.items():
        i, d = parse_indices(name)
        day_data[d]["start_times"][i] = val

    for name, val in x_vars.items():
        i, j, d, w = parse_indices(name)
        day_info = day_data[d]
        day_info["routes"][w].append((i, j))
        if event_count is not None:
            if i < event_count:
                day_info["assignments"][i].add(w)
            if j < event_count:
                day_info["assignments"][j].add(w)

    for name, val in alpha_vars.items():
        i, d, w = parse_indices(name)
        if val < 0.5:
            continue
        day_data[d]["leaders"][i]["pickup"] = w

    for name, val in beta_vars.items():
        i, d, w = parse_indices(name)
        if val < 0.5:
            continue
        day_data[d]["leaders"][i]["dropoff"] = w

    daily_solutions: List[Solution] = []
    if not day_data:
        daily_solutions.append(
            Solution(
                day=None,
                assignments={},
                leaders={},
                start_times={},
                objective_value=objective_value,
                lower_bound=lower_bound,
                status=status,
                extra={"message": "No positive variable values extracted"},
            )
        )
    else:
        for d in sorted(day_data):
            info = day_data[d]
            assignments = {evt: sorted(nurses) for evt, nurses in info["assignments"].items()}
            leaders = {
                evt: {"pickup": values.get("pickup"), "dropoff": values.get("dropoff")}
                for evt, values in info["leaders"].items()
            }
            routes = {nurse: sorted(arcs) for nurse, arcs in info["routes"].items()}
            daily_solutions.append(
                Solution(
                    day=day_index if day_index is not None else d,
                    assignments=assignments,
                    leaders=leaders,
                    start_times=info["start_times"],
                    objective_value=objective_value,
                    lower_bound=lower_bound,
                    status=status,
                    extra={
                        "scheduled_events": sorted(info["scheduled"]),
                        "routes": routes,
                        "original_event_ids": original_event_ids,
                        "local_event_count": event_count,
                        "day_index": day_index if day_index is not None else d,
                        # "global_event_count_guess": max(original_event_ids) + 1 if original_event_ids else event_count,
                    },
                )
            )

    return MergedSolution(
        daily_solutions=daily_solutions,
        objective_value=objective_value,
        lower_bound=lower_bound,
        status=status,
    )


def _map_node(
    idx: int, local_m: int, full_m: int, original_event_ids: Optional[List[int]]
) -> int:
    """Map local node indices (event/home/depot) back to full problem indices."""
    if idx < local_m:
        if original_event_ids is not None and idx < len(original_event_ids):
            return original_event_ids[idx]
        return idx
    if idx == local_m:
        return full_m  # home
    if idx == local_m + 1:
        return full_m + 1  # depot_am
    if idx == local_m + 2:
        return full_m + 2  # depot_pm
    return idx


def merge_day_solutions(
    day_solutions: List[MergedSolution], full_problem_data: Optional[ProblemData] = None
) -> MergedSolution:
    """Merge multiple per-day solutions into a single MergedSolution with global indices."""
    combined_daily: List[Solution] = []
    statuses = set()
    lower_bounds: List[float] = []
    objective_total = 0.0

    full_m: Optional[int] = full_problem_data.total_event if full_problem_data else None
    if full_m is None:
        for sol in day_solutions:
            for ds in sol.daily_solutions:
                extra = ds.extra or {}
                orig_ids = extra.get("original_event_ids")
                if orig_ids:
                    candidate = max(orig_ids) + 1
                    full_m = candidate if full_m is None else max(full_m, candidate)
    if full_m is None:
        full_m = 0

    for sol in day_solutions:
        statuses.add(sol.status)
        if sol.lower_bound is not None and not math.isnan(sol.lower_bound):
            lower_bounds.append(sol.lower_bound)
        if not math.isnan(sol.objective_value):
            objective_total += sol.objective_value

        for ds in sol.daily_solutions:
            extra = ds.extra or {}
            orig_ids: Optional[List[int]] = extra.get("original_event_ids")
            local_m = extra.get("local_event_count")
            if local_m is None:
                local_m = len(orig_ids) if orig_ids else 0
                if not local_m and ds.assignments:
                    local_m = max(ds.assignments.keys()) + 1

            mapped_assignments = {
                _map_node(evt, local_m, full_m, orig_ids): nurses for evt, nurses in ds.assignments.items()
            }
            mapped_start_times = {
                _map_node(evt, local_m, full_m, orig_ids): start for evt, start in ds.start_times.items()
            }
            mapped_leaders = {
                _map_node(evt, local_m, full_m, orig_ids): leader_info for evt, leader_info in ds.leaders.items()
            }

            mapped_routes: Dict[int, List[Tuple[int, int]]] = {}
            routes = extra.get("routes", {})
            for nurse, arcs in routes.items():
                mapped_routes[nurse] = [
                    (
                        _map_node(i, local_m, full_m, orig_ids),
                        _map_node(j, local_m, full_m, orig_ids),
                    )
                    for i, j in arcs
                ]

            scheduled_events = extra.get("scheduled_events", [])
            mapped_scheduled = [_map_node(evt, local_m, full_m, orig_ids) for evt in scheduled_events]

            mapped_extra = dict(extra)
            mapped_extra["routes"] = mapped_routes
            mapped_extra["scheduled_events"] = mapped_scheduled

            combined_daily.append(
                Solution(
                    day=ds.day,
                    assignments=mapped_assignments,
                    leaders=mapped_leaders,
                    start_times=mapped_start_times,
                    objective_value=ds.objective_value,
                    lower_bound=ds.lower_bound,
                    status=ds.status,
                    extra=mapped_extra,
                )
            )

    merged_status = "MERGED" if len(statuses) == 1 else "MERGED_" + "_".join(sorted(statuses))
    merged_lower_bound = sum(lower_bounds) if lower_bounds else None

    return MergedSolution(
        daily_solutions=combined_daily,
        objective_value=objective_total,
        lower_bound=merged_lower_bound,
        status=merged_status,
    )
