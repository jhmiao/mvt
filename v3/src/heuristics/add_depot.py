from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.io.cumulative_state import load_cumulative_state
from src.solver.config import SolverConfig
from src.solutions.solutions import MergedSolution, Solution
from src.structures.problem_data import ProblemData


DepotDayDict = Dict[int, Dict[int, Dict[str, Any]]]
DepotAssignment = Dict[int, Dict[int, int]]


def _daily_solutions(solution: MergedSolution | Solution) -> List[Solution]:
    if isinstance(solution, MergedSolution):
        return list(solution.daily_solutions)
    if isinstance(solution, Solution):
        return [solution]
    raise TypeError(f"Expected MergedSolution or Solution, got {type(solution).__name__}")


def _as_int_arcs(arcs: Iterable[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(i), int(j)) for i, j in arcs]


def _ordered_event_route(arcs: Iterable[Sequence[int]], event_count: int) -> List[int]:
    """Reconstruct the event order from solution arcs, ignoring home/depot nodes."""
    int_arcs = _as_int_arcs(arcs)
    if not int_arcs:
        return []

    successors: Dict[int, List[int]] = defaultdict(list)
    predecessors: Dict[int, List[int]] = defaultdict(list)
    for i, j in int_arcs:
        successors[i].append(j)
        predecessors[j].append(i)

    start: Optional[int] = None
    for candidate in (event_count, event_count + 1):
        next_events = [j for j in successors.get(candidate, []) if j < event_count]
        if next_events:
            start = next_events[0]
            break

    if start is None:
        event_nodes = sorted({node for arc in int_arcs for node in arc if node < event_count})
        starts = [node for node in event_nodes if not any(pred < event_count for pred in predecessors.get(node, []))]
        start = starts[0] if starts else (event_nodes[0] if event_nodes else None)

    if start is None:
        return []

    route: List[int] = []
    seen: set[int] = set()
    current = start
    while current < event_count and current not in seen:
        route.append(current)
        seen.add(current)
        next_events = [j for j in successors.get(current, []) if j < event_count]
        if not next_events:
            break
        current = next_events[0]

    if not route:
        return []

    routed = set(route)
    extras = sorted({node for arc in int_arcs for node in arc if node < event_count and node not in routed})
    return route + extras


def _routes_by_day(solution: MergedSolution | Solution, event_count: int) -> Dict[int, Dict[int, List[int]]]:
    routes_by_day: Dict[int, Dict[int, List[int]]] = {}

    for fallback_day, daily in enumerate(_daily_solutions(solution)):
        day = int(daily.day) if daily.day is not None else fallback_day
        extra = daily.extra or {}
        raw_routes = extra.get("routes", {})
        routes_by_day.setdefault(day, {})

        for raw_nurse, raw_arcs in raw_routes.items():
            nurse = int(raw_nurse)
            route = _ordered_event_route(raw_arcs, event_count)
            if route:
                routes_by_day[day][nurse] = route

    return routes_by_day


def _events_by_day(solution: MergedSolution | Solution, event_count: int) -> Dict[int, set[int]]:
    events_by_day: Dict[int, set[int]] = defaultdict(set)

    for fallback_day, daily in enumerate(_daily_solutions(solution)):
        day = int(daily.day) if daily.day is not None else fallback_day
        extra = daily.extra or {}

        for source in (daily.assignments.keys(), daily.start_times.keys(), extra.get("scheduled_events", [])):
            for event in source:
                event = int(event)
                if 0 <= event < event_count:
                    events_by_day[day].add(event)

        raw_routes = extra.get("routes", {})
        for raw_arcs in raw_routes.values():
            for i, j in _as_int_arcs(raw_arcs):
                if 0 <= i < event_count:
                    events_by_day[day].add(i)
                if 0 <= j < event_count:
                    events_by_day[day].add(j)

    return events_by_day


def depot_delta_cost(problem_data: ProblemData, solution: MergedSolution | Solution) -> DepotDayDict:
    """
    Compute the marginal cost for each nurse-day route to become a depot route.

    A selected route changes from
        home -> events -> home
    to
        home -> depot_am -> events -> depot_pm -> home.
    """
    event_count = problem_data.total_event
    routes_by_day = _routes_by_day(solution, event_count)
    events_by_day = _events_by_day(solution, event_count)

    day_dict: DepotDayDict = {}
    for day in sorted(set(routes_by_day) | set(events_by_day)):
        day_dict[day] = {}
        for nurse, route in sorted(routes_by_day.get(day, {}).items()):
            first_event = route[0]
            last_event = route[-1]

            current_cost = (
                float(problem_data.home_event_costs[nurse, first_event])
                + float(problem_data.home_event_costs[nurse, last_event])
            )
            depot_cost = (
                float(problem_data.home_depot_costs[nurse])
                + float(problem_data.event_depot_costs[first_event])
                + float(problem_data.event_depot_costs[last_event])
                + float(problem_data.home_depot_costs[nurse])
            )

            day_dict[day][nurse] = {
                "route": route,
                "routes": route,  # backward-compatible key used by the old implementation
                "delta_cost": depot_cost - current_cost,
            }

        for event in events_by_day.get(day, set()):
            if not any(event in info["route"] for info in day_dict[day].values()):
                raise ValueError(f"Event {event} on day {day} is not covered by any nurse route")

    return day_dict


def _status_to_str(status_code: int) -> str:
    lookup = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return lookup.get(status_code, str(status_code))


def solve_depot(
    problem_data: ProblemData,
    day_dict: DepotDayDict,
    solver_config: Optional[SolverConfig] = None,
) -> DepotAssignment:
    """
    Solve depot insertion selection with binary x[w,d].

    x[w,d] = 1 means nurse w visits the depot on day d. Every scheduled event must
    be included on at least one selected nurse-day route.
    """
    config = solver_config or SolverConfig()
    model = gp.Model("depot_assignment")
    model.Params.OutputFlag = int(config.gurobi_outputflag)
    model.Params.Seed = int(config.seed)
    if config.time_limit is not None:
        model.Params.TimeLimit = float(config.time_limit)
    if config.work_limit is not None:
        model.Params.WorkLimit = float(config.work_limit)

    x = {
        (nurse, day): model.addVar(vtype=GRB.BINARY, name=f"x[{nurse},{day}]")
        for day, nurses in day_dict.items()
        for nurse in nurses
    }

    model.setObjective(
        _depot_objective(model, problem_data, day_dict, x, config),
        GRB.MINIMIZE,
    )

    for day, nurses in day_dict.items():
        events = sorted({event for info in nurses.values() for event in info["route"]})
        for event in events:
            covering = [x[nurse, day] for nurse, info in nurses.items() if event in info["route"]]
            if not covering:
                raise ValueError(f"Event {event} on day {day} has no depot-cover candidate")
            model.addConstr(gp.quicksum(covering) >= 1, name=f"depot_cover[{event},{day}]")

    start = time.time()
    model.optimize()
    elapsed = time.time() - start

    status = _status_to_str(model.Status)
    if model.SolCount == 0:
        raise RuntimeError(f"Depot assignment failed with status {status}")

    depot_solution: DepotAssignment = {day: {} for day in day_dict}
    for (nurse, day), var in x.items():
        depot_solution[day][nurse] = int(round(var.X))

    objective_value = float(model.ObjVal)
    print(f"Depot assignment status: {status}; time: {elapsed:.4f}s; objective: {objective_value:.2f}")
    return depot_solution


def _depot_objective(
    model: gp.Model,
    problem_data: ProblemData,
    day_dict: DepotDayDict,
    x: Mapping[Tuple[int, int], gp.Var],
    config: SolverConfig,
):
    travel_delta = gp.quicksum(
        float(info["delta_cost"]) * x[nurse, day]
        for day, nurses in day_dict.items()
        for nurse, info in nurses.items()
    )
    objective = travel_delta

    if config.include_weekly_fairness_penalty_leaders:
        objective += _weekly_leader_fairness_expr(
            model=model,
            problem_data=problem_data,
            day_dict=day_dict,
            x=x,
            penalty_weight=float(config.leaders_penalty_weight),
        )

    if config.include_running_fairness_penalty:
        objective += _running_leader_fairness_expr(
            model=model,
            problem_data=problem_data,
            day_dict=day_dict,
            x=x,
            config=config,
            penalty_weight=float(config.leaders_penalty_weight),
        )

    return objective


def _leader_day_expr_by_nurse(
    problem_data: ProblemData,
    day_dict: DepotDayDict,
    x: Mapping[Tuple[int, int], gp.Var],
) -> Dict[int, gp.LinExpr]:
    return {
        nurse: gp.quicksum(
            x[nurse, day]
            for day, nurses in day_dict.items()
            if nurse in nurses and (nurse, day) in x
        )
        for nurse in range(problem_data.total_nurse)
    }


def _weekly_leader_fairness_expr(
    model: gp.Model,
    problem_data: ProblemData,
    day_dict: DepotDayDict,
    x: Mapping[Tuple[int, int], gp.Var],
    penalty_weight: float,
):
    if problem_data.total_nurse <= 1 or penalty_weight == 0:
        return gp.LinExpr(0.0)

    leader_days = _leader_day_expr_by_nurse(problem_data, day_dict, x)
    max_leader_days = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="depot_weekly_max_leader_days")
    min_leader_days = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="depot_weekly_min_leader_days")
    model.addConstrs(
        (max_leader_days >= leader_days[nurse] for nurse in range(problem_data.total_nurse)),
        name="depot_weekly_max_leader_days_lb",
    )
    model.addConstrs(
        (min_leader_days <= leader_days[nurse] for nurse in range(problem_data.total_nurse)),
        name="depot_weekly_min_leader_days_ub",
    )
    return penalty_weight * (max_leader_days - min_leader_days)


def _running_leader_fairness_expr(
    model: gp.Model,
    problem_data: ProblemData,
    day_dict: DepotDayDict,
    x: Mapping[Tuple[int, int], gp.Var],
    config: SolverConfig,
    penalty_weight: float,
):
    if problem_data.total_nurse <= 1 or penalty_weight == 0:
        return gp.LinExpr(0.0)
    if config.cumulative_state_path is None:
        raise ValueError("include_running_fairness_penalty=True requires cumulative_state_path")

    state = load_cumulative_state(config.cumulative_state_path, nurse_count=problem_data.total_nurse)
    history = [float(v) for v in state["cumu_leader_days"]]
    leader_days = _leader_day_expr_by_nurse(problem_data, day_dict, x)

    max_running = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="depot_running_max_leader_days")
    min_running = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="depot_running_min_leader_days")
    model.addConstrs(
        (
            max_running >= history[nurse] + leader_days[nurse]
            for nurse in range(problem_data.total_nurse)
        ),
        name="depot_running_max_leader_days_lb",
    )
    model.addConstrs(
        (
            min_running <= history[nurse] + leader_days[nurse]
            for nurse in range(problem_data.total_nurse)
        ),
        name="depot_running_min_leader_days_ub",
    )
    return penalty_weight * (max_running - min_running)


def _route_to_arcs(route: Sequence[int], event_count: int, use_depot: bool) -> List[Tuple[int, int]]:
    home = event_count
    depot_am = event_count + 1
    depot_pm = event_count + 2

    if not route:
        return []

    arcs: List[Tuple[int, int]] = []
    if use_depot:
        arcs.append((home, depot_am))
        arcs.append((depot_am, int(route[0])))
    else:
        arcs.append((home, int(route[0])))

    arcs.extend((int(i), int(j)) for i, j in zip(route, route[1:]))

    if use_depot:
        arcs.append((int(route[-1]), depot_pm))
        arcs.append((depot_pm, home))
    else:
        arcs.append((int(route[-1]), home))

    return arcs


def _delta_for_assignment(day_dict: DepotDayDict, depot_solution: DepotAssignment) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    total_delta = 0.0
    day_delta: Dict[int, float] = defaultdict(float)
    nurse_delta: Dict[int, float] = defaultdict(float)

    for day, nurses in day_dict.items():
        for nurse, info in nurses.items():
            if depot_solution.get(day, {}).get(nurse, 0):
                delta = float(info["delta_cost"])
                total_delta += delta
                day_delta[day] += delta
                nurse_delta[nurse] += delta

    return total_delta, dict(day_delta), dict(nurse_delta)


def _leader_days_for_assignment(problem_data: ProblemData, depot_solution: DepotAssignment) -> List[float]:
    leader_days = [0.0 for _ in range(problem_data.total_nurse)]
    for nurses in depot_solution.values():
        for nurse, selected in nurses.items():
            nurse = int(nurse)
            if 0 <= nurse < len(leader_days):
                leader_days[nurse] += float(int(round(selected)))
    return leader_days


def _leader_fairness_penalty(
    problem_data: ProblemData,
    leader_days_by_nurse: Sequence[float],
    solver_config: Optional[SolverConfig],
) -> Tuple[float, Dict[str, float]]:
    if solver_config is None:
        return 0.0, {}

    penalty_weight = float(solver_config.leaders_penalty_weight)
    total_penalty = 0.0
    breakdown: Dict[str, float] = {}

    if solver_config.include_weekly_fairness_penalty_leaders:
        weekly_range = max(leader_days_by_nurse) - min(leader_days_by_nurse) if leader_days_by_nurse else 0.0
        weekly_penalty = penalty_weight * float(weekly_range)
        total_penalty += weekly_penalty
        breakdown["weekly_leader_days_penalty"] = float(weekly_penalty)

    if solver_config.include_running_fairness_penalty:
        if solver_config.cumulative_state_path is None:
            raise ValueError("include_running_fairness_penalty=True requires cumulative_state_path")
        state = load_cumulative_state(solver_config.cumulative_state_path, nurse_count=problem_data.total_nurse)
        history = [float(v) for v in state["cumu_leader_days"]]
        running_totals = [
            history[nurse] + float(leader_days_by_nurse[nurse])
            for nurse in range(problem_data.total_nurse)
        ]
        running_range = max(running_totals) - min(running_totals) if running_totals else 0.0
        running_penalty = penalty_weight * float(running_range)
        total_penalty += running_penalty
        breakdown["running_leader_days_penalty"] = float(running_penalty)

    return float(total_penalty), breakdown


def _add_delta(value: Any, delta: float) -> Any:
    if value is None:
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(numeric):
        return value
    return numeric + delta


def _updated_metrics(
    metrics: Optional[Mapping[str, Any]],
    total_delta: float,
    fairness_penalty: float,
    fairness_breakdown: Mapping[str, float],
    nurse_delta: Mapping[int, float],
    leader_days_by_nurse: Sequence[float],
) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None

    updated = dict(metrics)
    updated["travel_cost_total"] = _add_delta(updated.get("travel_cost_total", 0.0), total_delta)

    by_nurse = updated.get("travel_cost_by_nurse")
    if isinstance(by_nurse, list):
        by_nurse = list(by_nurse)
        for nurse, delta in nurse_delta.items():
            if 0 <= nurse < len(by_nurse):
                by_nurse[nurse] = _add_delta(by_nurse[nurse], delta)
        updated["travel_cost_by_nurse"] = by_nurse

    if "penalty_cost_total" in updated:
        updated["penalty_cost_total"] = _add_delta(updated["penalty_cost_total"], fairness_penalty)
    elif fairness_penalty:
        updated["penalty_cost_total"] = float(fairness_penalty)

    updated["leader_days_by_nurse"] = [float(v) for v in leader_days_by_nurse]
    updated.update({key: float(value) for key, value in fairness_breakdown.items()})

    return updated


def add_depot_to_solution(
    problem_data: ProblemData,
    solution: MergedSolution | Solution,
    depot_solution: DepotAssignment,
    solver_config: Optional[SolverConfig] = None,
) -> MergedSolution:
    """Return a new solution whose nurse-day routes include depot arcs when selected."""
    event_count = problem_data.total_event
    day_dict = depot_delta_cost(problem_data, solution)
    total_delta, day_delta, nurse_delta = _delta_for_assignment(day_dict, depot_solution)
    leader_days_by_nurse = _leader_days_for_assignment(problem_data, depot_solution)
    fairness_penalty, fairness_breakdown = _leader_fairness_penalty(
        problem_data=problem_data,
        leader_days_by_nurse=leader_days_by_nurse,
        solver_config=solver_config,
    )

    new_daily: List[Solution] = []
    for fallback_day, daily in enumerate(_daily_solutions(solution)):
        day = int(daily.day) if daily.day is not None else fallback_day
        extra = dict(daily.extra or {})
        raw_routes = extra.get("routes", {})
        updated_routes: Dict[int, List[Tuple[int, int]]] = {}

        for raw_nurse, raw_arcs in raw_routes.items():
            nurse = int(raw_nurse)
            route = _ordered_event_route(raw_arcs, event_count)
            if route:
                updated_routes[nurse] = _route_to_arcs(
                    route=route,
                    event_count=event_count,
                    use_depot=bool(depot_solution.get(day, {}).get(nurse, 0)),
                )
            else:
                updated_routes[nurse] = _as_int_arcs(raw_arcs)

        extra["routes"] = updated_routes
        extra["depot_assignment"] = dict(depot_solution.get(day, {}))
        extra["depot_delta_cost"] = float(day_delta.get(day, 0.0))

        new_daily.append(
            replace(
                daily,
                objective_value=_add_delta(daily.objective_value, day_delta.get(day, 0.0)),
                extra=extra,
            )
        )

    if isinstance(solution, MergedSolution):
        return replace(
            solution,
            daily_solutions=new_daily,
            objective_value=_add_delta(solution.objective_value, total_delta + fairness_penalty),
            metrics=_updated_metrics(
                solution.metrics,
                total_delta,
                fairness_penalty,
                fairness_breakdown,
                nurse_delta,
                leader_days_by_nurse,
            ),
        )

    return MergedSolution(
        daily_solutions=new_daily,
        objective_value=_add_delta(solution.objective_value, total_delta + fairness_penalty),
        lower_bound=solution.lower_bound,
        status=solution.status,
        metrics=None,
    )


def add_depot(
    problem_data: ProblemData,
    solution: MergedSolution | Solution,
    solver_config: Optional[SolverConfig] = None,
) -> MergedSolution:
    """Convenience wrapper: compute depot costs, solve depot assignment, and update the solution."""
    day_dict = depot_delta_cost(problem_data, solution)
    depot_solution = solve_depot(problem_data, day_dict, solver_config=solver_config)
    return add_depot_to_solution(problem_data, solution, depot_solution, solver_config=solver_config)
