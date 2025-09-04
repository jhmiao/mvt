# src/heuristics/local_search/operators/two_opt.py
from typing import List, Optional, Tuple

Route = List[int]  # e.g., [HOME, v1, v2, ..., HOME]

def two_opt_delta(route: Route, i: int, k: int, dist) -> float:
    """Return cost change for reversing route[i:k+1]. Assumes 0<i<k<len-1."""
    a, b, c, d = route[i-1], route[i], route[k], route[k+1]
    before = dist[a, b] + dist[c, d]
    after  = dist[a, c] + dist[b, d]
    return after - before

def apply_two_opt(route: Route, i: int, k: int) -> Route:
    new_r = route[:i] + list(reversed(route[i:k+1])) + route[k+1:]
    return new_r

def best_two_opt(route: Route, dist, feasible_check) -> Tuple[Optional[Route], float]:
    """Try all (i,k); return best improving route and delta. None if no improvement."""
    best = None
    best_delta = 0.0
    n = len(route)
    for i in range(1, n-2):
        for k in range(i+1, n-1):
            d = two_opt_delta(route, i, k, dist)
            if d < -1e-9:
                cand = apply_two_opt(route, i, k)
                if feasible_check(cand):  # time windows, capacity, max shift, etc.
                    if d < best_delta:
                        best, best_delta = cand, d
    return best, best_delta
