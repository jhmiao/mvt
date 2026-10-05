# src/solver/route/pool_builder.py
"""
Efficient route-pool builder (up to three events per route) for route-based master.

Key assumptions:
1) All event-time copies (i,d,tau) are feasible w.r.t. time windows by construction.
2) Route feasibility w.r.t. between-event constraints is nurse-independent.
3) Home affects only route cost (home->first, last->home) but not feasibility.

Design:
- Build nurse-agnostic "visit skeletons" (tuples of EventCopy) for each day:
    * idle
    * all single-event
    * two- and three-event routes where each consecutive leg is feasible
- Then instantiate for each nurse-day, computing nurse-specific cost by adding home legs.

This keeps pool generation fast and avoids per-nurse feasibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import random

from .copies import CopyIndex, EventCopy, RouteVisits
from .routes import Route
from src.structures.problem_data import ProblemData


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class RoutePoolConfig:
    seed: int = 0
    include_idle_route: bool = True
    include_single_event_routes: bool = True
    include_two_event_routes: bool = True
    include_three_event_routes: bool = True

    # caps to control size
    max_single_per_day: Optional[int] = None      # None = all
    max_two_per_day: Optional[int] = 2000         # cap per day (two-event skeletons); None = all
    max_three_per_day: Optional[int] = 2000       # cap per day (three-event skeletons); None = all
    max_routes_per_nurse_day: Optional[int] = 500 # final per (w,d), including idle; None = all

    # Optional objective term: coefficient * (maximum workload - minimum workload).
    workload_fairness: bool = False
    workload_fairness_penalty_coefficient: float = 1.0

    # two-event generation strategy
    # if True: sample pairs (fast for large days); if False: enumerate all pairs then cap
    sample_two_event_pairs: bool = True
    two_event_pair_samples: int = 40000           # number of random pairs to test per day
    sample_three_event_triples: bool = True
    three_event_triple_samples: int = 40000       # number of random triples to test per day



# -----------------------------
# Public API
# -----------------------------
def build_route_pool(
    problem: ProblemData,
    copy_index: CopyIndex,
    cfg: RoutePoolConfig,
) -> Dict[Tuple[int, int], List[Route]]:
    """
    Returns:
      pool[(w,d)] -> list[Route] for nurse w on day d.

    Each nurse-day includes:
      - idle route (recommended)
      - instantiated routes from day skeletons (one, two, or three events)

    Note:
      This function builds day-level skeletons once, then instantiates per nurse.
    """
    rng = random.Random(cfg.seed)

    # 1) Build nurse-agnostic skeletons per day (visits tuples)
    day_skeletons: Dict[int, List[RouteVisits]] = build_day_skeletons(problem, copy_index, cfg, rng)

    # 2) Cache skeleton attributes that are nurse-independent.
    # Internal cost depends on whether the route uses the depot.
    internal_cost_cache: Dict[Tuple[RouteVisits, int], float] = {}
    work_cache: Dict[RouteVisits, int] = {}

    for d, skels in day_skeletons.items():
        for visits in skels:
            if visits in work_cache:
                continue
            
            work_cache[visits] = compute_work_minutes(problem, visits)
            for depot_incl in _depot_options(visits):
                internal_cost_cache[(visits, depot_incl)] = compute_internal_travel_cost(
                    problem, visits, depot_incl
                )

    # 3) Instantiate for each nurse-day: add home legs to internal cost
    pool: Dict[Tuple[int, int], List[Route]] = {}
    for w in range(problem.total_nurse):
        for d in range(problem.total_day):
            routes: List[Route] = []

            # Always include idle route (if enabled)
            if cfg.include_idle_route:
                routes.append(make_idle_route(w, d))

            # Instantiate day skeletons
            skels = day_skeletons.get(d, [])
            # (Optional) shuffle skeletons to diversify if we cap
            skels = list(skels)
            rng.shuffle(skels)

            for visits in skels:
                if len(visits) == 0:
                    continue  # idle already handled

                for depot_incl in _depot_options(visits):
                    cost = compute_total_cost_with_home(
                        problem,
                        w,
                        visits,
                        internal_cost_cache[(visits, depot_incl)],
                        depot_incl,
                    )
                    routes.append(
                        Route(
                            w=w,
                            d=d,
                            visits=visits,
                            cost=cost,
                            work=work_cache[visits],
                            depot_incl=depot_incl,
                        )
                    )

                    if (
                        cfg.max_routes_per_nurse_day is not None
                        and len(routes) >= cfg.max_routes_per_nurse_day
                    ):
                        break

                if (
                    cfg.max_routes_per_nurse_day is not None
                    and len(routes) >= cfg.max_routes_per_nurse_day
                ):
                    break

            # Deduplicate by visits (same w,d,visits)
            routes = dedup_routes(routes)

            pool[(w, d)] = routes

    return pool


# -----------------------------
# Day skeleton construction
# -----------------------------
def build_day_skeletons(
    problem: ProblemData,
    copy_index: CopyIndex,
    cfg: RoutePoolConfig,
    rng: random.Random,
) -> Dict[int, List[RouteVisits]]:
    """
    Build nurse-agnostic skeletons per day:
      - optionally idle (empty visits)
      - all / sampled single-event visits
      - feasible two-event visits (v1, v2) where v2 can follow v1
      - feasible three-event visits (v1, v2, v3) with feasible consecutive legs
    """
    day_skeletons: Dict[int, List[RouteVisits]] = {}

    for d in range(problem.total_day):
        skels: List[RouteVisits] = []

        if cfg.include_idle_route:
            skels.append(tuple())  # idle skeleton

        day_copies = list(copy_index.copies_by_day.get(d, []))

        # --- Single-event skeletons ---
        if cfg.include_single_event_routes:
            if cfg.max_single_per_day is None or cfg.max_single_per_day >= len(day_copies):
                singles = [(v,) for v in day_copies]
            else:
                rng.shuffle(day_copies)
                singles = [(v,) for v in day_copies[: cfg.max_single_per_day]]
            skels.extend(singles)

        # --- Two-event skeletons ---
        if cfg.include_two_event_routes:
            twos = build_two_event_skeletons(problem, day_copies, cfg, rng)
            skels.extend(twos)

        # --- Three-event skeletons ---
        if cfg.include_three_event_routes:
            threes = build_three_event_skeletons(problem, day_copies, cfg, rng)
            skels.extend(threes)

        # Dedup skeletons by exact visits tuple
        skels = list(dict.fromkeys(skels))  # preserves order, python3.7+
        day_skeletons[d] = skels

    return day_skeletons


def build_two_event_skeletons(
    problem: ProblemData,
    day_copies: List[EventCopy],
    cfg: RoutePoolConfig,
    rng: random.Random,
) -> List[RouteVisits]:
    """
    Build feasible two-event skeletons (v1,v2) for a given day using nurse-independent feasibility:
      end(v1) + travel(event1,event2) <= start(v2)

    We cap the number of returned skeletons to cfg.max_two_per_day.

    For large |day_copies|, enumerating all pairs is O(N^2).
    We support either:
      - random sampling of pairs (fast)
      - full enumeration then cap (simple)
    """
    n = len(day_copies)
    if n <= 1:
        return []

    out: List[RouteVisits] = []

    # Precompute start/end times in minutes and event IDs
    # v = (i, d, tau)
    starts: Dict[EventCopy, int] = {v: slot_to_minute(v[2]) for v in day_copies}
    ends: Dict[EventCopy, int] = {v: starts[v] + problem.event_durations[v[0]] for v in day_copies}

    def feasible_pair(v1: EventCopy, v2: EventCopy) -> bool:
        if v1[0] == v2[0]:
            return False  # can't serve same event twice
        i1, i2 = v1[0], v2[0]
        travel_12 = problem.event_event_costs[i1][i2]
        return ends[v1] + travel_12 <= starts[v2]

    if cfg.sample_two_event_pairs:
        # Sample random ordered pairs (v1,v2)
        samples = min(cfg.two_event_pair_samples, n * (n - 1))
        for _ in range(samples):
            v1 = day_copies[rng.randrange(n)]
            v2 = day_copies[rng.randrange(n)]
            if v1 == v2:
                continue
            if feasible_pair(v1, v2):
                out.append((v1, v2))
                if cfg.max_two_per_day is not None and len(out) >= cfg.max_two_per_day:
                    break
    else:
        # Enumerate all ordered pairs then cap
        for v1 in day_copies:
            for v2 in day_copies:
                if v1 == v2:
                    continue
                if feasible_pair(v1, v2):
                    out.append((v1, v2))
                    if cfg.max_two_per_day is not None and len(out) >= cfg.max_two_per_day:
                        return out

    # Deduplicate (sampling can produce duplicates)
    out = list(dict.fromkeys(out))
    if cfg.max_two_per_day is not None and len(out) > cfg.max_two_per_day:
        rng.shuffle(out)
        out = out[: cfg.max_two_per_day]
    return out


def build_three_event_skeletons(
    problem: ProblemData,
    day_copies: List[EventCopy],
    cfg: RoutePoolConfig,
    rng: random.Random,
) -> List[RouteVisits]:
    """Build feasible ordered triples ``(v1, v2, v3)`` for one day.

    A triple is feasible when all three copies are for different events and both
    consecutive legs satisfy the time-and-travel condition:

    ``end(v1) + travel(v1, v2) <= start(v2)`` and
    ``end(v2) + travel(v2, v3) <= start(v3)``.

    As with two-event skeletons, candidates can be sampled or fully enumerated;
    ``max_three_per_day=None`` retains every feasible triple.
    """
    n = len(day_copies)
    if n <= 2:
        return []

    out: List[RouteVisits] = []
    starts: Dict[EventCopy, int] = {v: slot_to_minute(v[2]) for v in day_copies}
    ends: Dict[EventCopy, int] = {v: starts[v] + problem.event_durations[v[0]] for v in day_copies}

    def feasible_leg(first: EventCopy, second: EventCopy) -> bool:
        return ends[first] + problem.event_event_costs[first[0]][second[0]] <= starts[second]

    def feasible_triple(v1: EventCopy, v2: EventCopy, v3: EventCopy) -> bool:
        if len({v1[0], v2[0], v3[0]}) != 3:
            return False
        return feasible_leg(v1, v2) and feasible_leg(v2, v3)

    if cfg.sample_three_event_triples:
        samples = min(cfg.three_event_triple_samples, n * (n - 1) * (n - 2))
        for _ in range(samples):
            v1 = day_copies[rng.randrange(n)]
            v2 = day_copies[rng.randrange(n)]
            v3 = day_copies[rng.randrange(n)]
            if feasible_triple(v1, v2, v3):
                out.append((v1, v2, v3))
                if cfg.max_three_per_day is not None and len(out) >= cfg.max_three_per_day:
                    break
    else:
        for v1 in day_copies:
            for v2 in day_copies:
                if v1[0] == v2[0] or not feasible_leg(v1, v2):
                    continue
                for v3 in day_copies:
                    if feasible_triple(v1, v2, v3):
                        out.append((v1, v2, v3))
                        if cfg.max_three_per_day is not None and len(out) >= cfg.max_three_per_day:
                            return out

    out = list(dict.fromkeys(out))
    if cfg.max_three_per_day is not None and len(out) > cfg.max_three_per_day:
        rng.shuffle(out)
        out = out[: cfg.max_three_per_day]
    return out


# -----------------------------
# Route attribute computation
# -----------------------------
def make_idle_route(w: int, d: int) -> Route:
    return Route(w=w, d=d, visits=tuple(), cost=0.0, work=0, depot_incl=0)


def compute_work_minutes(problem: ProblemData, visits: RouteVisits) -> int:
    """Sum of service durations (minutes). Adjust if your definition differs."""
    return sum(problem.event_durations[v[0]] for v in visits)


def compute_internal_travel_cost(problem: ProblemData, visits: RouteVisits, depot_incl: int) -> float:
    """
    Internal travel excludes home legs.
    For now: sum of event->event travel between consecutive visits.
    If you later include depot between events, incorporate it here.
    """
    if len(visits) < 1:
        return 0.0
    total = 0.0
    for a, b in zip(visits[:-1], visits[1:]):
        total += float(problem.event_event_costs[a[0]][b[0]])

    if depot_incl:
        # add costs: depot -> first event + last event -> depot
        first_i = visits[0][0]
        last_i = visits[-1][0]
        total += float(problem.event_depot_costs[first_i] + problem.event_depot_costs[last_i])
    return total


def compute_total_cost_with_home(
    problem: ProblemData,
    w: int,
    visits: RouteVisits,
    internal_cost: float,
    depot_incl: int = 0,
) -> float:
    """Total cost = home->first + internal + last->home (+ depot legs if you later add them)."""
    if len(visits) == 0:
        return 0.0
    first_i = visits[0][0]
    last_i = visits[-1][0]
    if not depot_incl:
        home_legs = float(problem.home_event_costs[w][first_i] + problem.home_event_costs[w][last_i])
    else:
        # goes from home to depot in the morning, then from depot to home in the evening
        home_legs = 2 * float(problem.home_depot_costs[w])
    return home_legs + internal_cost


def _depot_options(visits: RouteVisits) -> Tuple[int, ...]:
    """Return the depot-use variants available for a visit sequence.

    ``depot_incl=1`` denotes a route that actually goes through the depot; it is
    not a blanket feasibility marker.  Idle routes never use the depot.
    """
    if not visits:
        return (0,)
    return (0, 1)


# -----------------------------
# ProblemData accessors (adjust these to match your structures)
# -----------------------------
def slot_to_minute(tau: int) -> int:
    return 30 * int(tau)



# -----------------------------
# Utilities
# -----------------------------
def dedup_routes(routes: List[Route]) -> List[Route]:
    """Deduplicate by route identity, including whether it uses the depot."""
    seen = set()
    out: List[Route] = []
    for r in routes:
        key = (r.w, r.d, r.visits, r.depot_incl)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
