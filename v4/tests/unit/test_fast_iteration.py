from __future__ import annotations

from src.fairness_heuristics_route.fast_iteration import (
    _choose_neighborhood,
    _same_nurse_type,
    _state_fingerprint,
)
from src.solutions.route_rmp_result import RouteRmpResult, RouteSelection


class _FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _day(*routes: tuple[int, int]) -> RouteRmpResult:
    return RouteRmpResult(
        status=2,
        runtime=0.0,
        obj_val=0.0,
        obj_bound=0.0,
        mip_gap=0.0,
        node_count=0,
        selected_routes=[
            RouteSelection(
                nurse=nurse,
                day=0,
                route_index=route_index,
                visits=(),
                cost=0.0,
                work=0,
                depot_incl=0,
            )
            for nurse, route_index in routes
        ],
    )


def test_operator_roulette_boundaries() -> None:
    assert _choose_neighborhood(_FixedRandom(0.00)) == "workload"
    assert _choose_neighborhood(_FixedRandom(0.349999)) == "workload"
    assert _choose_neighborhood(_FixedRandom(0.35)) == "leader_day"
    assert _choose_neighborhood(_FixedRandom(0.70)) == "route_swap"
    assert _choose_neighborhood(_FixedRandom(0.85)) == "combined"
    assert _choose_neighborhood(_FixedRandom(0.95)) == "multi_destroy"


def test_fingerprint_includes_source_day_nurse_and_route() -> None:
    first = _state_fingerprint((_day((0, 2), (1, 4)), _day((0, 3), (1, 1))))
    reordered = _state_fingerprint((_day((1, 4), (0, 2)), _day((1, 1), (0, 3))))
    changed = _state_fingerprint((_day((0, 2), (1, 4)), _day((0, 9), (1, 1))))
    assert first == reordered
    assert first != changed


def test_same_nurse_type_uses_rn_boundary() -> None:
    assert _same_nurse_type(0, 2, total_rn=3)
    assert _same_nurse_type(3, 5, total_rn=3)
    assert not _same_nurse_type(2, 3, total_rn=3)
