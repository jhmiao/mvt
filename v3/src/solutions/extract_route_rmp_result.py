from __future__ import annotations

from typing import List

from src.solver.route.rmp_runner import RmpSolveResult
from src.solutions.route_rmp_result import EventSchedule, RouteRmpResult, RouteSelection


def extract_route_rmp_result(result: RmpSolveResult) -> RouteRmpResult:
    model = result.model
    ctx = result.ctx

    event_schedule: List[EventSchedule] = []
    selected_routes: List[RouteSelection] = []

    if getattr(model, "SolCount", 0) > 0:
        for (i, d, tau), var in ctx.y.items():
            if var.X > 0.5:
                event_schedule.append(EventSchedule(event=i, day=d, time_slot=tau))

        for (w, d, k), var in ctx.z.items():
            if var.X > 0.5:
                route = ctx.pool[(w, d)][k]
                selected_routes.append(
                    RouteSelection(
                        nurse=w,
                        day=d,
                        route_index=k,
                        visits=tuple(route.visits),
                        cost=route.cost,
                        work=route.work,
                        depot_ok=route.depot_ok,
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
