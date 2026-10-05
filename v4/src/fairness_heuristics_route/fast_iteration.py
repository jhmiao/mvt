"""Fast route-pool ALNS with roulette neighborhoods and SA acceptance.

The daily RMP is deliberately kept separate from the weekly fairness score:
Gurobi repairs a small, temporarily restricted route pool using travel cost
only.  The weekly score is computed after extraction and controls simulated
annealing acceptance and best-solution tracking.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import gurobipy as gp
from gurobipy import GRB

from src.fairness_heuristics_route.workload import RouteWeekEvaluation, evaluate_week
from src.solver.config import SolverConfig
from src.solver.route.master_builder import RouteMasterContext, build_master_model
from src.solver.route.pool_builder import RoutePoolConfig
from src.solver.route.rmp_runner import (
    PreparedRouteRmp,
    RmpSolveResult,
    prepare_route_rmp,
)
from src.solutions.extract_route_rmp_result import extract_route_rmp_result
from src.solutions.route_rmp_result import RouteRmpResult, RouteSelection
from src.structures.problem_data import ProblemData


T = TypeVar("T")
NurseDayKey = Tuple[int, int]
RouteKey = Tuple[int, int, int]
RouteSignature = Tuple[Tuple[Tuple[int, int, int], ...], int]
StateFingerprint = Tuple[Tuple[int, int, int], ...]


@dataclass(frozen=True)
class _RouteSearchIndex:
    """Immutable attribute indexes for one prepared daily route pool."""

    depot_keys: Dict[NurseDayKey, Tuple[RouteKey, ...]]
    work_values: Dict[NurseDayKey, Tuple[float, ...]]
    work_keys: Dict[NurseDayKey, Tuple[RouteKey, ...]]
    equivalent_key: Dict[Tuple[int, int, RouteSignature], RouteKey]


@dataclass(frozen=True)
class FastRouteFairnessConfig:
    """Configuration for fast route-based destroy, repair, and SA search."""

    max_iterations: int = 1_000
    wall_time_limit: Optional[float] = None
    workload_penalty_coefficient: float = 1.0
    leader_day_penalty_coefficient: float = 1.0
    top_k_nurses: int = 3
    top_k_leaders: Optional[int] = None
    rank_biased_selection: bool = True
    workload_route_restriction_factor: float = 1.0
    repair_time_limit: Optional[float] = 0.25
    repair_work_limit: Optional[float] = None
    repair_solution_limit: Optional[int] = 1
    initial_time_limit: Optional[float] = None
    initial_work_limit: Optional[float] = None
    initial_solution_limit: Optional[int] = None
    initial_temperature: float = 0.05
    cooling_rate: float = 0.995
    epsilon: float = 1e-9
    seed: int = 0
    gurobi_outputflag: int = 0


@dataclass(frozen=True)
class FastRouteIteration:
    iteration: int
    neighborhood: str
    feasible: bool
    accepted: bool
    improving: bool
    new_best: bool
    message: str
    affected_nurse: Optional[int] = None
    affected_nurses: Tuple[int, ...] = ()
    affected_day: Optional[int] = None
    repair_seconds: float = 0.0
    duplicate: bool = False
    temperature: Optional[float] = None
    objective_before: Optional[float] = None
    candidate_objective: Optional[float] = None
    travel_cost_before: Optional[float] = None
    candidate_travel_cost: Optional[float] = None
    workload_penalty_before: Optional[float] = None
    candidate_workload_penalty: Optional[float] = None
    leader_day_penalty_before: Optional[float] = None
    candidate_leader_day_penalty: Optional[float] = None
    best_objective_after: Optional[float] = None
    relative_delta: Optional[float] = None
    acceptance_probability: Optional[float] = None


@dataclass(frozen=True)
class FastOperatorStats:
    attempts: int = 0
    feasible_repairs: int = 0
    duplicate_candidates: int = 0
    accepted_moves: int = 0
    improving_moves: int = 0
    new_best_moves: int = 0
    total_repair_seconds: float = 0.0

    @property
    def average_repair_seconds(self) -> float:
        return self.total_repair_seconds / self.attempts if self.attempts else 0.0


@dataclass(frozen=True)
class FastRouteFairnessResult:
    """Best schedule is returned in ``daily_results`` and ``evaluation``."""

    daily_results: Tuple[RouteRmpResult, ...]
    evaluation: RouteWeekEvaluation
    current_daily_results: Tuple[RouteRmpResult, ...]
    current_evaluation: RouteWeekEvaluation
    history: Tuple[FastRouteIteration, ...]
    operator_stats: Dict[str, FastOperatorStats]


class FastRouteWeekHeuristic:
    """Roulette-selected route neighborhoods with SA on a weekly score."""

    def __init__(
        self,
        problem: ProblemData,
        pool_config: RoutePoolConfig,
        config: FastRouteFairnessConfig = FastRouteFairnessConfig(),
    ) -> None:
        _validate_config(config)
        self.problem = problem
        # The RMP's base objective is travel.  Disable the legacy in-model
        # workload fairness switch even if a caller supplied it in the pool cfg.
        self.pool_config = RoutePoolConfig(**{**pool_config.__dict__, "workload_fairness": False})
        self.config = config
        self.rng = random.Random(config.seed)
        self.day_problems = tuple(problem.split_by_day())
        self.prepared_days: Tuple[PreparedRouteRmp, ...] = tuple(
            prepare_route_rmp(day_problem, self.pool_config)
            for day_problem in self.day_problems
        )
        self._route_search_indices: Tuple[_RouteSearchIndex, ...] = tuple(
            _build_route_search_index(prepared) for prepared in self.prepared_days
        )
        self._daily_rmp_bases: Tuple[Tuple[gp.Model, RouteMasterContext], ...] = ()

    def run(self) -> FastRouteFairnessResult:
        self._daily_rmp_bases = self._build_daily_rmp_bases()
        try:
            return self._run_search()
        finally:
            self._dispose_daily_rmp_bases()

    def _run_search(self) -> FastRouteFairnessResult:
        current = [self._solve_day(day, initial=True) for day in range(self.problem.total_day)]
        current_eval = self._evaluate_week(current)
        best = list(current)
        best_eval = current_eval
        temperature = self.config.initial_temperature
        history: List[FastRouteIteration] = []
        operator_names = ("workload", "leader_day", "route_swap", "combined", "multi_destroy")
        stats: Dict[str, Dict[str, float]] = {
            name: _empty_stats() for name in operator_names
        }
        seen_fingerprints = {_state_fingerprint(current)}
        started = time.monotonic()

        for iteration in range(self.config.max_iterations):
            if self.config.wall_time_limit is not None and time.monotonic() - started >= self.config.wall_time_limit:
                break
            neighborhood = _choose_neighborhood(self.rng)
            stats[neighborhood]["attempts"] += 1
            candidate, affected_nurses, affected_day, repair_seconds, error = self._propose(
                current, current_eval, neighborhood
            )
            stats[neighborhood]["total_repair_seconds"] += repair_seconds
            if candidate is None:
                history.append(
                    FastRouteIteration(
                        iteration=iteration,
                        neighborhood=neighborhood,
                        feasible=False,
                        accepted=False,
                        improving=False,
                        new_best=False,
                        message=error or "No feasible repair found within the repair budget.",
                        affected_nurse=affected_nurses[0] if affected_nurses else None,
                        affected_nurses=affected_nurses,
                        affected_day=affected_day,
                        repair_seconds=repair_seconds,
                        temperature=temperature,
                        objective_before=current_eval.objective,
                        travel_cost_before=current_eval.travel_cost,
                        workload_penalty_before=current_eval.workload_penalty,
                        leader_day_penalty_before=current_eval.leader_day_penalty,
                        best_objective_after=best_eval.objective,
                    )
                )
                continue

            stats[neighborhood]["feasible_repairs"] += 1
            candidate_eval = self._evaluate_week(candidate)
            candidate_fingerprint = _state_fingerprint(candidate)
            duplicate = candidate_fingerprint in seen_fingerprints
            if duplicate:
                stats[neighborhood]["duplicate_candidates"] += 1
                history.append(
                    FastRouteIteration(
                        iteration=iteration,
                        neighborhood=neighborhood,
                        feasible=True,
                        accepted=False,
                        improving=False,
                        new_best=False,
                        message="Rejected duplicate candidate; temperature unchanged.",
                        affected_nurse=affected_nurses[0] if affected_nurses else None,
                        affected_nurses=affected_nurses,
                        affected_day=affected_day,
                        repair_seconds=repair_seconds,
                        duplicate=True,
                        temperature=temperature,
                        objective_before=current_eval.objective,
                        candidate_objective=candidate_eval.objective,
                        travel_cost_before=current_eval.travel_cost,
                        candidate_travel_cost=candidate_eval.travel_cost,
                        workload_penalty_before=current_eval.workload_penalty,
                        candidate_workload_penalty=candidate_eval.workload_penalty,
                        leader_day_penalty_before=current_eval.leader_day_penalty,
                        candidate_leader_day_penalty=candidate_eval.leader_day_penalty,
                        best_objective_after=best_eval.objective,
                    )
                )
                continue
            seen_fingerprints.add(candidate_fingerprint)
            evaluation_before = current_eval
            objective_before = current_eval.objective
            new_best = candidate_eval.objective < best_eval.objective - self.config.epsilon
            if new_best:
                best = list(candidate)
                best_eval = candidate_eval
                stats[neighborhood]["new_best_moves"] += 1

            relative_delta = (candidate_eval.objective - objective_before) / max(
                abs(objective_before), self.config.epsilon
            )
            improving = relative_delta <= 0.0
            probability = 1.0 if improving else _sa_probability(relative_delta, temperature)
            accepted = improving or self.rng.random() < probability
            if accepted:
                current = candidate
                current_eval = candidate_eval
                stats[neighborhood]["accepted_moves"] += 1
            if improving:
                stats[neighborhood]["improving_moves"] += 1

            history.append(
                FastRouteIteration(
                    iteration=iteration,
                    neighborhood=neighborhood,
                    feasible=True,
                    accepted=accepted,
                    improving=improving,
                    new_best=new_best,
                    message="Accepted by SA." if accepted else "Rejected by SA.",
                    affected_nurse=affected_nurses[0] if affected_nurses else None,
                    affected_nurses=affected_nurses,
                    affected_day=affected_day,
                    repair_seconds=repair_seconds,
                    temperature=temperature,
                    objective_before=objective_before,
                    candidate_objective=candidate_eval.objective,
                    travel_cost_before=evaluation_before.travel_cost,
                    candidate_travel_cost=candidate_eval.travel_cost,
                    workload_penalty_before=evaluation_before.workload_penalty,
                    candidate_workload_penalty=candidate_eval.workload_penalty,
                    leader_day_penalty_before=evaluation_before.leader_day_penalty,
                    candidate_leader_day_penalty=candidate_eval.leader_day_penalty,
                    best_objective_after=best_eval.objective,
                    relative_delta=relative_delta,
                    acceptance_probability=probability,
                )
            )
            temperature *= self.config.cooling_rate

        return FastRouteFairnessResult(
            daily_results=tuple(best),
            evaluation=best_eval,
            current_daily_results=tuple(current),
            current_evaluation=current_eval,
            history=tuple(history),
            operator_stats={name: _freeze_stats(values) for name, values in stats.items()},
        )

    def _build_daily_rmp_bases(self) -> Tuple[Tuple[gp.Model, RouteMasterContext], ...]:
        """Build one travel-only daily RMP per source day for the full search."""
        bases: List[Tuple[gp.Model, RouteMasterContext]] = []
        try:
            for day, (day_problem, prepared) in enumerate(zip(self.day_problems, self.prepared_days)):
                solver_config = SolverConfig(
                    backend="route_pool",
                    seed=self.config.seed + day,
                    gurobi_outputflag=self.config.gurobi_outputflag,
                    include_depot=True,
                    route_workload_fairness=False,
                )
                model, ctx = build_master_model(
                    day_problem,
                    prepared.copy_index,
                    prepared.pool,
                    solver_config,
                )
                model.update()
                bases.append((model, ctx))
        except Exception:
            for model, _ in bases:
                model.dispose()
            raise
        return tuple(bases)

    def _dispose_daily_rmp_bases(self) -> None:
        for model, _ in self._daily_rmp_bases:
            model.dispose()
        self._daily_rmp_bases = ()

    def _propose(
        self,
        current: Sequence[RouteRmpResult],
        evaluation: RouteWeekEvaluation,
        neighborhood: str,
    ) -> Tuple[Optional[List[RouteRmpResult]], Tuple[int, ...], Optional[int], float, Optional[str]]:
        if neighborhood == "route_swap":
            started = time.monotonic()
            candidate, nurses, day = self._propose_route_swap(current, evaluation)
            if candidate is None:
                return (
                    None, nurses, day, time.monotonic() - started,
                    "No same-type overloaded/underloaded route swap improves the full objective.",
                )
            return candidate, nurses, day, time.monotonic() - started, None

        forbid_depot_routes_by_nurse_day: Optional[Tuple[Tuple[int, int], ...]] = None
        forbid_routes_with_work_at_least_by_nurse_day: Optional[Dict[Tuple[int, int], float]] = None
        forbidden_route_keys: Optional[Tuple[RouteKey, ...]] = None
        affected_nurses: Tuple[int, ...]
        if neighborhood == "workload":
            choice = _choose_workload_nurse_day(
                current, evaluation.workloads_by_nurse, self.config.top_k_nurses,
                self.config.rank_biased_selection, self.rng,
            )
            if choice is None:
                return None, (), None, 0.0, "No positive-work nurse-day is available to destroy."
            nurse, day, route = choice
            threshold = math.ceil(route.work * self.config.workload_route_restriction_factor)
            forbid_routes_with_work_at_least_by_nurse_day = {(nurse, 0): float(threshold)}
            affected_nurses = (nurse,)
        elif neighborhood == "leader_day":
            choice = _choose_leader_nurse_day(
                current, evaluation.leader_days_by_nurse,
                self.config.top_k_leaders or self.config.top_k_nurses,
                self.config.rank_biased_selection, self.rng,
            )
            if choice is None:
                return None, (), None, 0.0, "No leader-day nurse-day is available to destroy."
            nurse, day = choice
            forbid_depot_routes_by_nurse_day = ((nurse, 0),)
            affected_nurses = (nurse,)
        elif neighborhood == "combined":
            choice = _choose_combined_nurse_day(
                current,
                evaluation.workloads_by_nurse,
                evaluation.leader_days_by_nurse,
                self.config.top_k_nurses,
                self.config.rank_biased_selection,
                self.rng,
            )
            if choice is None:
                return None, (), None, 0.0, "No positive-work depot route is available for combined repair."
            nurse, day, route = choice
            threshold = math.ceil(route.work * self.config.workload_route_restriction_factor)
            forbid_depot_routes_by_nurse_day = ((nurse, 0),)
            forbid_routes_with_work_at_least_by_nurse_day = {(nurse, 0): float(threshold)}
            affected_nurses = (nurse,)
        elif neighborhood == "multi_destroy":
            choice = _choose_multi_destroy(
                current,
                evaluation.workloads_by_nurse,
                evaluation.leader_days_by_nurse,
                self.config.top_k_nurses,
                self.rng,
            )
            if choice is None:
                return None, (), None, 0.0, "No day has routes available for a top-k multi-destroy."
            day, selected = choice
            affected_nurses = tuple(route.nurse for route in selected)
            forbidden_route_keys = tuple((route.nurse, 0, route.route_index) for route in selected)
        else:
            raise ValueError(f"Unknown fast-route neighborhood: {neighborhood}")

        started = time.monotonic()
        try:
            repaired_day = self._solve_day(
                day,
                initial=False,
                forbid_depot_routes_by_nurse_day=forbid_depot_routes_by_nurse_day,
                forbid_routes_with_work_at_least_by_nurse_day=(
                    forbid_routes_with_work_at_least_by_nurse_day
                ),
                forbidden_route_keys=forbidden_route_keys,
            )
        except RuntimeError as error:
            return None, affected_nurses, day, time.monotonic() - started, str(error)
        candidate = list(current)
        candidate[day] = repaired_day
        return candidate, affected_nurses, day, time.monotonic() - started, None

    def _propose_route_swap(
        self,
        current: Sequence[RouteRmpResult],
        evaluation: RouteWeekEvaluation,
    ) -> Tuple[Optional[List[RouteRmpResult]], Tuple[int, ...], Optional[int]]:
        """Return an improving same-day swap between same-type nurses."""
        improving: List[Tuple[float, List[RouteRmpResult], Tuple[int, int], int]] = []
        mean_workload = (
            sum(evaluation.workloads_by_nurse) / len(evaluation.workloads_by_nurse)
            if evaluation.workloads_by_nurse else 0.0
        )
        for day, day_result in enumerate(current):
            routes_by_nurse = {route.nurse: route for route in day_result.selected_routes}
            for donor in range(self.problem.total_nurse):
                donor_route = routes_by_nurse.get(donor)
                if donor_route is None:
                    continue
                for receiver in range(self.problem.total_nurse):
                    if donor >= receiver or not _same_nurse_type(donor, receiver, self.problem.total_rn):
                        continue
                    receiver_route = routes_by_nurse.get(receiver)
                    if receiver_route is None or donor_route.work == receiver_route.work:
                        continue
                    overloaded, underloaded = (donor, receiver)
                    high_route, low_route = donor_route, receiver_route
                    if evaluation.workloads_by_nurse[overloaded] < evaluation.workloads_by_nurse[underloaded]:
                        overloaded, underloaded = underloaded, overloaded
                        high_route, low_route = low_route, high_route
                    if (
                        evaluation.workloads_by_nurse[overloaded] <= mean_workload + self.config.epsilon
                        or evaluation.workloads_by_nurse[underloaded] >= mean_workload - self.config.epsilon
                    ):
                        continue
                    if high_route.work <= low_route.work:
                        continue
                    swapped_day = self._swap_day_routes(day_result, day, high_route, low_route)
                    if swapped_day is None:
                        continue
                    candidate = list(current)
                    candidate[day] = swapped_day
                    candidate_eval = self._evaluate_week(candidate)
                    improvement = evaluation.objective - candidate_eval.objective
                    if improvement > self.config.epsilon:
                        improving.append((improvement, candidate, (overloaded, underloaded), day))
        if not improving:
            return None, (), None
        improving.sort(key=lambda item: -item[0])
        _, candidate, nurses, day = _choose_ranked_value(
            improving[:max(1, self.config.top_k_nurses)],
            self.config.rank_biased_selection,
            self.rng,
        )
        return candidate, nurses, day

    def _swap_day_routes(
        self,
        day_result: RouteRmpResult,
        source_day: int,
        first: RouteSelection,
        second: RouteSelection,
    ) -> Optional[RouteRmpResult]:
        """Exchange two route skeletons using their nurse-specific pool entries."""
        prepared = self.prepared_days[source_day]
        index = self._route_search_indices[source_day]
        first_pool_route = prepared.pool[(first.nurse, 0)][first.route_index]
        second_pool_route = prepared.pool[(second.nurse, 0)][second.route_index]
        first_key = index.equivalent_key.get(
            (first.nurse, 0, (second_pool_route.visits, int(second_pool_route.depot_incl)))
        )
        second_key = index.equivalent_key.get(
            (second.nurse, 0, (first_pool_route.visits, int(first_pool_route.depot_incl)))
        )
        if first_key is None or second_key is None:
            return None
        replacements = {
            first.nurse: _selection_from_pool_route(prepared, self.day_problems[source_day], first_key),
            second.nurse: _selection_from_pool_route(prepared, self.day_problems[source_day], second_key),
        }
        selected_routes = [replacements.get(route.nurse, route) for route in day_result.selected_routes]
        travel_cost = sum(float(route.cost) for route in selected_routes)
        return replace(day_result, obj_val=travel_cost, selected_routes=selected_routes)

    def _solve_day(
        self,
        day: int,
        *,
        initial: bool,
        forbid_depot_routes_by_nurse_day: Optional[Tuple[Tuple[int, int], ...]] = None,
        forbid_routes_with_work_at_least_by_nurse_day: Optional[Dict[Tuple[int, int], float]] = None,
        forbidden_route_keys: Optional[Tuple[RouteKey, ...]] = None,
    ) -> RouteRmpResult:
        if not self._daily_rmp_bases:
            raise RuntimeError("Daily RMP bases are available only while FastRouteWeekHeuristic.run() is active")

        model, ctx = self._daily_rmp_bases[day]
        _configure_model_for_solve(
            model,
            time_limit=self.config.initial_time_limit if initial else self.config.repair_time_limit,
            work_limit=self.config.initial_work_limit if initial else self.config.repair_work_limit,
            solution_limit=self.config.initial_solution_limit if initial else self.config.repair_solution_limit,
            seed=self.config.seed + day,
            outputflag=self.config.gurobi_outputflag,
        )
        original_upper_bounds = _temporarily_forbid_routes(
            ctx,
            self._route_search_indices[day],
            forbid_depot_routes_by_nurse_day,
            forbid_routes_with_work_at_least_by_nurse_day,
            forbidden_route_keys,
        )
        try:
            model.optimize()
            result = _rmp_result_from_model(model, ctx)
            if getattr(result.model, "SolCount", 0) < 1:
                raise RuntimeError(f"No feasible route-RMP solution for source day {day}; status={result.status}")
            return extract_route_rmp_result(result, self.day_problems[day])
        finally:
            _restore_route_upper_bounds(ctx, original_upper_bounds)

    def _evaluate_week(self, daily_results: Sequence[RouteRmpResult]) -> RouteWeekEvaluation:
        return evaluate_week(
            daily_results,
            self.problem.total_nurse,
            self.config.workload_penalty_coefficient,
            self.config.leader_day_penalty_coefficient,
        )


def _configure_model_for_solve(
    model: gp.Model,
    *,
    time_limit: Optional[float],
    work_limit: Optional[float],
    solution_limit: Optional[int],
    seed: int,
    outputflag: int,
) -> None:
    """Set per-solve Gurobi limits, resetting prior repair settings."""
    if solution_limit is not None and solution_limit < 1:
        raise ValueError("solution_limit must be at least one when provided")
    model.Params.OutputFlag = int(outputflag)
    model.Params.Seed = int(seed)
    model.Params.TimeLimit = GRB.INFINITY if time_limit is None else float(time_limit)
    model.Params.WorkLimit = GRB.INFINITY if work_limit is None else float(work_limit)
    model.Params.SolutionLimit = GRB.MAXINT if solution_limit is None else int(solution_limit)


def _temporarily_forbid_routes(
    ctx: RouteMasterContext,
    search_index: _RouteSearchIndex,
    depot_nurse_days: Optional[Tuple[Tuple[int, int], ...]],
    minimum_work_by_nurse_day: Optional[Dict[Tuple[int, int], float]],
    forbidden_route_keys: Optional[Tuple[RouteKey, ...]],
) -> Dict[Tuple[int, int, int], float]:
    """Find indexed routes and set their variable upper bounds to zero."""
    keys_to_forbid: set[RouteKey] = set(forbidden_route_keys or ())

    if depot_nurse_days:
        for w, d in depot_nurse_days:
            nurse_day = (w, d)
            if nurse_day not in search_index.depot_keys:
                raise ValueError(f"Cannot forbid depot routes for nurse-day not present in the RMP pool: {nurse_day}")
            keys_to_forbid.update(search_index.depot_keys[nurse_day])

    if minimum_work_by_nurse_day:
        for (w, d), minimum_work in minimum_work_by_nurse_day.items():
            nurse_day = (w, d)
            if nurse_day not in search_index.work_values:
                raise ValueError(f"Cannot restrict routes for nurse-day not present in the RMP pool: {nurse_day}")
            first_forbidden = bisect_left(
                search_index.work_values[nurse_day],
                float(minimum_work),
            )
            keys_to_forbid.update(search_index.work_keys[nurse_day][first_forbidden:])

    original_upper_bounds: Dict[Tuple[int, int, int], float] = {}
    for key in keys_to_forbid:
        variable = ctx.z[key]
        original_upper_bounds[key] = float(variable.UB)
        variable.UB = 0.0
    return original_upper_bounds


def _build_route_search_index(prepared: PreparedRouteRmp) -> _RouteSearchIndex:
    """Index depot inclusion and workload once for a prepared daily pool."""
    depot_keys: Dict[NurseDayKey, Tuple[RouteKey, ...]] = {}
    work_values: Dict[NurseDayKey, Tuple[float, ...]] = {}
    work_keys: Dict[NurseDayKey, Tuple[RouteKey, ...]] = {}
    equivalent_key: Dict[Tuple[int, int, RouteSignature], RouteKey] = {}

    for (w, d), routes in prepared.pool.items():
        nurse_day = (w, d)
        depot_keys[nurse_day] = tuple(
            (w, d, k) for k, route in enumerate(routes) if route.depot_incl
        )
        work_entries = sorted(
            (float(route.work), (w, d, k))
            for k, route in enumerate(routes)
        )
        work_values[nurse_day] = tuple(work for work, _ in work_entries)
        work_keys[nurse_day] = tuple(key for _, key in work_entries)
        for k, route in enumerate(routes):
            equivalent_key[(w, d, (route.visits, int(route.depot_incl)))] = (w, d, k)

    return _RouteSearchIndex(
        depot_keys=depot_keys,
        work_values=work_values,
        work_keys=work_keys,
        equivalent_key=equivalent_key,
    )


def _restore_route_upper_bounds(
    ctx: RouteMasterContext,
    original_upper_bounds: Dict[Tuple[int, int, int], float],
) -> None:
    for key, upper_bound in original_upper_bounds.items():
        ctx.z[key].UB = upper_bound


def _rmp_result_from_model(model: gp.Model, ctx: RouteMasterContext) -> RmpSolveResult:
    """Collect the result metadata while retaining the reusable model/context."""
    status = int(model.Status)
    has_primal = int(getattr(model, "SolCount", 0)) > 0
    return RmpSolveResult(
        model=model,
        ctx=ctx,
        status=status,
        runtime=float(getattr(model, "Runtime", 0.0)),
        obj_val=_safe_model_float(getattr(model, "ObjVal", None)) if has_primal else None,
        obj_bound=_safe_model_float(getattr(model, "ObjBound", None)),
        mip_gap=_safe_model_float(getattr(model, "MIPGap", None)),
        node_count=int(getattr(model, "NodeCount", 0)),
    )


def _safe_model_float(value: object) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _choose_ranked_value(
    values: Sequence[T],
    rank_biased: bool,
    rng: random.Random,
) -> T:
    """Choose one value from a non-empty sequence, preserving its type."""
    if not values:
        raise ValueError("values must be non-empty")
    if not rank_biased:
        return rng.choice(values)
    return rng.choices(values, weights=list(range(len(values), 0, -1)), k=1)[0]

def _choose_workload_nurse_day(
    daily_results: Sequence[RouteRmpResult], burdens: Sequence[float], top_k: int,
    rank_biased: bool, rng: random.Random,
) -> Optional[Tuple[int, int, RouteSelection]]:
    nurse = _choose_top_ranked(burdens, top_k, rank_biased, rng)
    if nurse is None:
        return None
    choices = [
        (day, route) for day, result in enumerate(daily_results)
        for route in result.selected_routes if route.nurse == nurse and route.work > 0
    ]
    if not choices:
        return None
    choices.sort(key=lambda item: (-item[1].work, item[0]))
    day, route = _choose_ranked_value(choices, rank_biased, rng)
    return nurse, day, route


def _choose_leader_nurse_day(
    daily_results: Sequence[RouteRmpResult], burdens: Sequence[float], top_k: int,
    rank_biased: bool, rng: random.Random,
) -> Optional[Tuple[int, int]]:
    nurse = _choose_top_ranked(burdens, top_k, rank_biased, rng)
    if nurse is None:
        return None
    days = [
        day for day, result in enumerate(daily_results)
        if any(route.nurse == nurse and route.depot_incl for route in result.selected_routes)
    ]
    if not days:
        return None
    return nurse, _choose_ranked_value(sorted(days), rank_biased, rng)


def _choose_combined_nurse_day(
    daily_results: Sequence[RouteRmpResult],
    workloads: Sequence[float],
    leader_days: Sequence[float],
    top_k: int,
    rank_biased: bool,
    rng: random.Random,
) -> Optional[Tuple[int, int, RouteSelection]]:
    """Choose one high-burden nurse-day having both work and depot duty."""
    workload_scale = max(workloads, default=0.0) or 1.0
    leader_scale = max(leader_days, default=0.0) or 1.0
    combined = [
        workloads[nurse] / workload_scale + leader_days[nurse] / leader_scale
        for nurse in range(len(workloads))
    ]
    top_nurses = set(
        sorted(range(len(combined)), key=lambda nurse: (-combined[nurse], nurse))[:top_k]
    )
    choices = [
        (route.nurse, day, route)
        for day, result in enumerate(daily_results)
        for route in result.selected_routes
        if route.nurse in top_nurses and route.work > 0 and route.depot_incl
    ]
    if not choices:
        return None
    choices.sort(key=lambda item: (-combined[item[0]], -item[2].work, item[1], item[0]))
    return _choose_ranked_value(choices, rank_biased, rng)


def _choose_multi_destroy(
    daily_results: Sequence[RouteRmpResult],
    workloads: Sequence[float],
    leader_days: Sequence[float],
    top_k: int,
    rng: random.Random,
) -> Optional[Tuple[int, Tuple[RouteSelection, ...]]]:
    """Choose one day and every selected route of the top-k weekly burdens."""
    if not daily_results or not workloads:
        return None
    workload_scale = max(workloads, default=0.0) or 1.0
    leader_scale = max(leader_days, default=0.0) or 1.0
    combined = [
        workloads[nurse] / workload_scale + leader_days[nurse] / leader_scale
        for nurse in range(len(workloads))
    ]
    top_nurses = tuple(
        sorted(range(len(combined)), key=lambda nurse: (-combined[nurse], nurse))[:top_k]
    )
    day_choices: List[Tuple[float, int, Tuple[RouteSelection, ...]]] = []
    for day, result in enumerate(daily_results):
        routes_by_nurse = {route.nurse: route for route in result.selected_routes}
        selected = tuple(routes_by_nurse[nurse] for nurse in top_nurses if nurse in routes_by_nurse)
        if len(selected) != len(top_nurses):
            continue
        strength = sum(float(route.work) + float(bool(route.depot_incl)) for route in selected)
        day_choices.append((strength, day, selected))
    if not day_choices:
        return None
    day_choices.sort(key=lambda item: (-item[0], item[1]))
    _, day, selected = _choose_ranked_value(day_choices[:min(3, len(day_choices))], True, rng)
    return day, selected


def _choose_top_ranked(values: Sequence[float], top_k: int, rank_biased: bool, rng: random.Random) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))[:top_k]
    return _choose_ranked_value(ordered, rank_biased, rng)


def _choose_neighborhood(rng: random.Random) -> str:
    """Use the fixed 35/35/15/10/5 operator roulette."""
    draw = rng.random()
    if draw < 0.20:
        return "workload"
    if draw < 0.40:
        return "leader_day"
    if draw < 0.85:
        return "route_swap"
    if draw < 0.95:
        return "combined"
    return "multi_destroy"


def _same_nurse_type(first: int, second: int, total_rn: int) -> bool:
    return (first < total_rn) == (second < total_rn)


def _selection_from_pool_route(
    prepared: PreparedRouteRmp,
    day_problem: ProblemData,
    key: RouteKey,
) -> RouteSelection:
    nurse, local_day, route_index = key
    route = prepared.pool[(nurse, local_day)][route_index]
    original_day = day_problem.day_index if day_problem.day_index >= 0 else local_day
    visits = tuple(
        (
            int(day_problem.original_event_ids[event])
            if day_problem.original_event_ids is not None else int(event),
            int(original_day),
            int(slot),
        )
        for event, _, slot in route.visits
    )
    return RouteSelection(
        nurse=nurse,
        day=int(original_day),
        route_index=route_index,
        visits=visits,
        cost=float(route.cost),
        work=int(route.work),
        depot_incl=int(route.depot_incl),
        travel=float(route.travel),
        waiting=int(route.waiting),
    )


def _state_fingerprint(daily_results: Sequence[RouteRmpResult]) -> StateFingerprint:
    """Identify a weekly state by its selected source-day/nurse/route keys."""
    return tuple(
        sorted(
            (source_day, int(route.nurse), int(route.route_index))
            for source_day, result in enumerate(daily_results)
            for route in result.selected_routes
        )
    )


def _sa_probability(relative_delta: float, temperature: float) -> float:
    if temperature <= 0.0:
        return 0.0
    return math.exp(-relative_delta / temperature)


def _empty_stats() -> Dict[str, float]:
    return {
        "attempts": 0.0, "feasible_repairs": 0.0, "duplicate_candidates": 0.0,
        "accepted_moves": 0.0,
        "improving_moves": 0.0, "new_best_moves": 0.0, "total_repair_seconds": 0.0,
    }


def _freeze_stats(values: Dict[str, float]) -> FastOperatorStats:
    return FastOperatorStats(
        attempts=int(values["attempts"]), feasible_repairs=int(values["feasible_repairs"]),
        duplicate_candidates=int(values["duplicate_candidates"]),
        accepted_moves=int(values["accepted_moves"]), improving_moves=int(values["improving_moves"]),
        new_best_moves=int(values["new_best_moves"]), total_repair_seconds=values["total_repair_seconds"],
    )


def _validate_config(config: FastRouteFairnessConfig) -> None:
    if config.max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if config.wall_time_limit is not None and config.wall_time_limit <= 0:
        raise ValueError("wall_time_limit must be positive when provided")
    if config.top_k_nurses < 1 or (config.top_k_leaders is not None and config.top_k_leaders < 1):
        raise ValueError("top-k values must be at least one")
    if not 0 < config.workload_route_restriction_factor <= 1:
        raise ValueError("workload_route_restriction_factor must be in (0, 1]")
    if config.workload_penalty_coefficient < 0 or config.leader_day_penalty_coefficient < 0:
        raise ValueError("fairness penalty coefficients must be non-negative")
    if config.repair_time_limit is not None and config.repair_time_limit <= 0:
        raise ValueError("repair_time_limit must be positive when provided")
    if config.repair_solution_limit is not None and config.repair_solution_limit < 1:
        raise ValueError("repair_solution_limit must be at least one when provided")
    if config.initial_temperature < 0 or not 0 < config.cooling_rate <= 1:
        raise ValueError("initial_temperature must be non-negative and cooling_rate must be in (0, 1]")
