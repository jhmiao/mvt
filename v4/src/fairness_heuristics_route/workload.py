"""A small weekly workload-fairness heuristic built on the route RMP.

The route RMP remains a one-day model.  This module keeps one selected result
per source day and repairs one day at a time, so all route-selection, coverage,
staffing, and depot constraints continue to be enforced by the existing solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional, Sequence, Tuple

from src.solver.route.pool_builder import RoutePoolConfig
from src.solver.route.rmp_runner import (
    PreparedRouteRmp,
    RmpSolveResult,
    prepare_route_rmp,
    solve_rmp_routes,
)
from src.solutions.extract_route_rmp_result import extract_route_rmp_result
from src.solutions.route_rmp_result import RouteRmpResult, RouteSelection
from src.structures.problem_data import ProblemData


@dataclass(frozen=True)
class RouteFairnessConfig:
    """Controls the deliberately small destroy-and-repair neighborhood."""

    max_iterations: int = 20
    workload_penalty_coefficient: float = 1.0
    leader_day_penalty_coefficient: float = 1.0
    top_k_nurses: int = 10
    top_k_leaders: Optional[int] = None
    time_limit: Optional[float] = None
    work_limit: Optional[float] = None
    seed: int = 0
    gurobi_outputflag: int = 0
    improvement_tolerance: float = 1e-6


@dataclass(frozen=True)
class RouteWeekEvaluation:
    travel_cost: float
    workload_penalty: float
    workload_objective: float
    workloads_by_nurse: Tuple[float, ...]
    leader_day_penalty: float
    leader_day_objective: float
    leader_days_by_nurse: Tuple[float, ...]
    objective: float


@dataclass(frozen=True)
class RouteFairnessIteration:
    iteration: int
    neighborhood: str
    accepted: bool
    message: str
    affected_nurse: Optional[int] = None
    affected_day: Optional[int] = None
    objective_before: Optional[float] = None
    objective_after: Optional[float] = None
    final_objective_before: Optional[float] = None
    final_objective_after: Optional[float] = None


@dataclass(frozen=True)
class RouteFairnessResult:
    daily_results: Tuple[RouteRmpResult, ...]
    evaluation: RouteWeekEvaluation
    history: Tuple[RouteFairnessIteration, ...]


class RouteWeekHeuristic:
    """Alternate weekly workload and leader-day destroy-and-repair phases."""

    def __init__(
        self,
        problem: ProblemData,
        pool_config: RoutePoolConfig,
        config: RouteFairnessConfig = RouteFairnessConfig(),
    ) -> None:
        if config.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if config.workload_penalty_coefficient < 0:
            raise ValueError("workload_penalty_coefficient must be non-negative")
        if config.leader_day_penalty_coefficient < 0:
            raise ValueError("leader_day_penalty_coefficient must be non-negative")
        if config.top_k_nurses < 1:
            raise ValueError("top_k_nurses must be at least one")
        if config.top_k_leaders is not None and config.top_k_leaders < 1:
            raise ValueError("top_k_leaders must be at least one when provided")
        self.problem = problem
        # Daily fairness in the RMP would optimize the wrong quantity.  Weekly
        # fairness is evaluated below after all daily selections are combined.
        self.pool_config = RoutePoolConfig(
            **{**pool_config.__dict__, "workload_fairness": False}
        )
        self.config = config
        self.rng = random.Random(config.seed)
        self.day_problems = tuple(problem.split_by_day())
        self.prepared_days: Tuple[PreparedRouteRmp, ...] = tuple(
            prepare_route_rmp(day_problem, self.pool_config)
            for day_problem in self.day_problems
        )

    def run(self) -> RouteFairnessResult:
        incumbent = [self._solve_day(day) for day in range(self.problem.total_day)]
        evaluation = self._evaluate_week(incumbent)
        history: List[RouteFairnessIteration] = []

        for iteration in range(self.config.max_iterations):
            if iteration % 2 == 0:
                incumbent, evaluation, record = self._workload_iteration(iteration, incumbent, evaluation)
            else:
                incumbent, evaluation, record = self._leader_day_iteration(iteration, incumbent, evaluation)
            history.append(record)

        return RouteFairnessResult(tuple(incumbent), evaluation, tuple(history))

    def _solve_day(
        self,
        day: int,
        forbidden_route_keys: Optional[Sequence[Tuple[int, int, int]]] = None,
        maximum_work_by_nurse_day: Optional[dict[Tuple[int, int], float]] = None,
        prior_travel_cost: float = 0.0,
        prior_workloads_by_nurse: Optional[Sequence[float]] = None,
        prior_leader_days_by_nurse: Optional[Sequence[float]] = None,
        forbid_depot_routes_for_nurse_day: Optional[Tuple[int, int]] = None,
        stop_when_objective_below: Optional[float] = None,
    ) -> RouteRmpResult:
        result: RmpSolveResult = solve_rmp_routes(
            problem=self.day_problems[day],
            pool_cfg=self.pool_config,
            time_limit=self.config.time_limit,
            work_limit=self.config.work_limit,
            seed=self.config.seed + day,
            outputflag=self.config.gurobi_outputflag,
            forbidden_route_keys=forbidden_route_keys,
            forbid_depot_routes_by_nurse_day=(
                (forbid_depot_routes_for_nurse_day,)
                if forbid_depot_routes_for_nurse_day is not None
                else None
            ),
            maximum_work_by_nurse_day=maximum_work_by_nurse_day,
            weekly_workload_objective=prior_workloads_by_nurse is not None,
            prior_travel_cost=prior_travel_cost,
            prior_workloads_by_nurse=prior_workloads_by_nurse,
            weekly_workload_penalty_coefficient=self.config.workload_penalty_coefficient,
            weekly_leader_day_objective=prior_leader_days_by_nurse is not None,
            prior_leader_days_by_nurse=prior_leader_days_by_nurse,
            weekly_leader_day_penalty_coefficient=self.config.leader_day_penalty_coefficient,
            stop_when_objective_below=stop_when_objective_below,
            prepared=self.prepared_days[day],
        )
        try:
            if getattr(result.model, "SolCount", 0) < 1:
                raise RuntimeError(f"No feasible route-RMP solution for source day {day}; status={result.status}")
            return extract_route_rmp_result(result, self.day_problems[day])
        finally:
            # Extracted results are plain data, so release the Gurobi model.
            result.model.dispose()

    def _workload_iteration(
        self,
        iteration: int,
        incumbent: List[RouteRmpResult],
        current: RouteWeekEvaluation,
    ) -> Tuple[List[RouteRmpResult], RouteWeekEvaluation, RouteFairnessIteration]:
        affected = _random_route_of_top_k_burden_nurses(
            incumbent, current.workloads_by_nurse, self.config.top_k_nurses, self.rng
        )
        if affected is None:
            return incumbent, current, RouteFairnessIteration(
                iteration, "workload", False, "No positive-work route is available to destroy.",
                objective_before=current.workload_objective,
                final_objective_before=current.objective,
                final_objective_after=current.objective,
            )

        nurse, day, route = affected
        # Split problems have local day 0, even though the extracted route keeps
        # the original day number for reporting.
        try:
            day_travel, day_workloads = _day_contributions(incumbent[day], self.problem.total_nurse)
            candidate_day = self._solve_day(
                day,
                forbidden_route_keys=((nurse, 0, route.route_index),),
                # Route workloads are integral minutes.  This prevents an
                # equivalent route for the same nurse from undoing the repair.
                maximum_work_by_nurse_day={(nurse, 0): route.work - 1},
                prior_travel_cost=current.travel_cost - day_travel,
                prior_workloads_by_nurse=tuple(
                    workload - day_workloads[w] for w, workload in enumerate(current.workloads_by_nurse)
                ),
                stop_when_objective_below=current.workload_objective - self.config.improvement_tolerance,
            )
        except RuntimeError as error:
            return incumbent, current, RouteFairnessIteration(
                iteration, "workload", False, f"Workload repair was infeasible: {error}",
                affected_nurse=nurse, affected_day=day,
                objective_before=current.workload_objective,
                final_objective_before=current.objective,
                final_objective_after=current.objective,
            )
        candidate = list(incumbent)
        candidate[day] = candidate_day
        candidate_eval = self._evaluate_week(candidate)
        accepted = (
            candidate_eval.workload_objective
            < current.workload_objective - self.config.improvement_tolerance
        )
        record = RouteFairnessIteration(
            iteration=iteration,
            neighborhood="workload",
            accepted=accepted,
            message="Accepted workload route repair." if accepted else "Rejected workload route repair: no weekly-objective improvement.",
            affected_nurse=nurse,
            affected_day=day,
            objective_before=current.workload_objective,
            objective_after=candidate_eval.workload_objective,
            final_objective_before=current.objective,
            final_objective_after=candidate_eval.objective,
        )
        return (candidate, candidate_eval, record) if accepted else (incumbent, current, record)

    def _leader_day_iteration(
        self,
        iteration: int,
        incumbent: List[RouteRmpResult],
        current: RouteWeekEvaluation,
    ) -> Tuple[List[RouteRmpResult], RouteWeekEvaluation, RouteFairnessIteration]:
        affected = _random_leader_day_of_top_k_nurses(
            incumbent,
            current.leader_days_by_nurse,
            self.config.top_k_leaders or self.config.top_k_nurses,
            self.rng,
        )
        if affected is None:
            return incumbent, current, RouteFairnessIteration(
                iteration, "leader_day", False, "No leader-day route is available to destroy.",
                objective_before=current.leader_day_objective,
                final_objective_before=current.objective,
                final_objective_after=current.objective,
            )

        nurse, day = affected
        try:
            day_travel, day_leader_days = _day_leader_contributions(incumbent[day], self.problem.total_nurse)
            candidate_day = self._solve_day(
                day,
                prior_travel_cost=current.travel_cost - day_travel,
                prior_leader_days_by_nurse=tuple(
                    burden - day_leader_days[w]
                    for w, burden in enumerate(current.leader_days_by_nurse)
                ),
                forbid_depot_routes_for_nurse_day=(nurse, 0),
                stop_when_objective_below=current.leader_day_objective - self.config.improvement_tolerance,
            )
        except RuntimeError as error:
            return incumbent, current, RouteFairnessIteration(
                iteration, "leader_day", False, f"Leader-day repair was infeasible: {error}",
                affected_nurse=nurse, affected_day=day, objective_before=current.leader_day_objective,
                final_objective_before=current.objective,
                final_objective_after=current.objective,
            )
        candidate = list(incumbent)
        candidate[day] = candidate_day
        candidate_eval = self._evaluate_week(candidate)
        accepted = candidate_eval.leader_day_objective < current.leader_day_objective - self.config.improvement_tolerance
        record = RouteFairnessIteration(
            iteration=iteration,
            neighborhood="leader_day",
            accepted=accepted,
            message="Accepted leader-day route repair." if accepted else "Rejected leader-day route repair: no weekly leader objective improvement.",
            affected_nurse=nurse,
            affected_day=day,
            objective_before=current.leader_day_objective,
            objective_after=candidate_eval.leader_day_objective,
            final_objective_before=current.objective,
            final_objective_after=candidate_eval.objective,
        )
        return (candidate, candidate_eval, record) if accepted else (incumbent, current, record)

    def _evaluate_week(self, daily_results: Sequence[RouteRmpResult]) -> RouteWeekEvaluation:
        return evaluate_week(
            daily_results,
            self.problem.total_nurse,
            self.config.workload_penalty_coefficient,
            self.config.leader_day_penalty_coefficient,
        )


def evaluate_week(
    daily_results: Sequence[RouteRmpResult],
    nurse_count: int,
    workload_penalty_coefficient: float,
    leader_day_penalty_coefficient: float = 0.0,
) -> RouteWeekEvaluation:
    """Evaluate the final combined score and the phase-specific repair scores."""
    workloads = [0.0] * nurse_count
    leader_days = [0.0] * nurse_count
    travel_cost = 0.0
    for day_result in daily_results:
        for route in day_result.selected_routes:
            travel_cost += float(route.cost)
            workloads[route.nurse] += float(route.work)
            leader_days[route.nurse] += float(bool(route.depot_incl))
    mean = sum(workloads) / nurse_count if nurse_count else 0.0
    workload_penalty = float(workload_penalty_coefficient) * sum(abs(work - mean) for work in workloads)
    leader_mean = sum(leader_days) / nurse_count if nurse_count else 0.0
    leader_day_penalty = float(leader_day_penalty_coefficient) * sum(
        abs(leader_days_for_nurse - leader_mean) for leader_days_for_nurse in leader_days
    )
    workload_objective = travel_cost + workload_penalty
    leader_day_objective = travel_cost + leader_day_penalty
    final_objective = travel_cost + workload_penalty + leader_day_penalty
    return RouteWeekEvaluation(
        travel_cost=travel_cost,
        workload_penalty=workload_penalty,
        workload_objective=workload_objective,
        workloads_by_nurse=tuple(workloads),
        leader_day_penalty=leader_day_penalty,
        leader_day_objective=leader_day_objective,
        leader_days_by_nurse=tuple(leader_days),
        objective=final_objective,
    )


def _day_contributions(day_result: RouteRmpResult, nurse_count: int) -> Tuple[float, Tuple[float, ...]]:
    """Return the travel and workload contributions to remove for one source day."""
    workloads = [0.0] * nurse_count
    travel_cost = 0.0
    for route in day_result.selected_routes:
        travel_cost += float(route.cost)
        workloads[route.nurse] += float(route.work)
    return travel_cost, tuple(workloads)


def _day_leader_contributions(day_result: RouteRmpResult, nurse_count: int) -> Tuple[float, Tuple[float, ...]]:
    """Return the travel and leader-day contributions to remove for one source day."""
    leader_days = [0.0] * nurse_count
    travel_cost = 0.0
    for route in day_result.selected_routes:
        travel_cost += float(route.cost)
        leader_days[route.nurse] += float(bool(route.depot_incl))
    return travel_cost, tuple(leader_days)


def _random_route_of_top_k_burden_nurses(
    daily_results: Sequence[RouteRmpResult],
    workloads: Sequence[float],
    top_k_nurses: int,
    rng: random.Random,
) -> Optional[Tuple[int, int, RouteSelection]]:
    """Uniformly choose a positive-work nurse-day route from top-k burdens.

    Nurses tied with the kth workload are included.  The seeded RNG avoids
    repeating the deterministic largest-route choice after a rejected repair.
    """
    if not workloads:
        return None
    ordered = sorted(range(len(workloads)), key=lambda nurse: (-workloads[nurse], nurse))
    cutoff = workloads[ordered[min(top_k_nurses, len(ordered)) - 1]]
    selected_nurses = {nurse for nurse in ordered if workloads[nurse] >= cutoff}
    choices = [
        (route.nurse, day, route)
        for day, result in enumerate(daily_results)
        for route in result.selected_routes
        if route.nurse in selected_nurses and route.work > 0
    ]
    if not choices:
        return None
    return rng.choice(choices)


def _random_leader_day_of_top_k_nurses(
    daily_results: Sequence[RouteRmpResult],
    leader_days: Sequence[float],
    top_k_nurses: int,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    """Uniformly choose a depot-using nurse-day route from top-k leader burdens."""
    if not leader_days:
        return None
    ordered = sorted(range(len(leader_days)), key=lambda nurse: (-leader_days[nurse], nurse))
    cutoff = leader_days[ordered[min(top_k_nurses, len(ordered)) - 1]]
    selected_nurses = {nurse for nurse in ordered if leader_days[nurse] >= cutoff}
    choices = [
        (route.nurse, day)
        for day, result in enumerate(daily_results)
        for route in result.selected_routes
        if route.nurse in selected_nurses and route.depot_incl
    ]
    return rng.choice(choices) if choices else None
