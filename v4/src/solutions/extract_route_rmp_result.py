from __future__ import annotations

from typing import List, Optional

from src.solver.route.rmp_runner import RmpSolveResult
from src.solutions.route_rmp_result import EventSchedule, RouteRmpResult, RouteSelection
from src.structures.problem_data import ProblemData


def extract_route_rmp_result(result: RmpSolveResult, problem: Optional[ProblemData] = None) -> RouteRmpResult:
    model = result.model
    ctx = result.ctx

    event_schedule: List[EventSchedule] = []
    selected_routes: List[RouteSelection] = []

    if getattr(model, "SolCount", 0) > 0:
        for (i, d, tau), var in ctx.y.items():
            if var.X > 0.5:
                event_schedule.append(
                    EventSchedule(
                        event=_original_event_id(problem, i),
                        day=_original_day(problem, d),
                        time_slot=tau,
                    )
                )

        for (w, d, k), var in ctx.z.items():
            if var.X > 0.5:
                route = ctx.pool[(w, d)][k]
                selected_routes.append(
                    RouteSelection(
                        nurse=w,
                        day=_original_day(problem, d),
                        route_index=k,
                        visits=tuple(
                            (_original_event_id(problem, i), _original_day(problem, day), tau)
                            for i, day, tau in route.visits
                        ),
                        cost=route.cost,
                        work=route.work,
                        depot_incl=route.depot_incl,
                        travel=getattr(route, "travel", None),
                        waiting=getattr(route, "waiting", None),
                    )
                )

    event_schedule.sort(key=lambda item: (item.event, item.day, item.time_slot))
    selected_routes.sort(key=lambda item: (item.day, item.nurse, item.route_index))

    return RouteRmpResult(
        status=result.status,
        runtime=result.runtime,
        obj_val=result.obj_val,
        obj_bound=result.obj_bound,
        mip_gap=result.mip_gap,
        node_count=result.node_count,
        event_schedule=event_schedule,
        selected_routes=selected_routes,
    )


def _original_event_id(problem: Optional[ProblemData], event: int) -> int:
    if problem is None or problem.original_event_ids is None:
        return event
    return int(problem.original_event_ids[event])


def _original_day(problem: Optional[ProblemData], day: int) -> int:
    if problem is None or problem.day_index < 0:
        return day
    return problem.day_index
