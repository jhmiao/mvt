# src/solver/route/pool_builder.py
"""
Efficient route-pool builder (single-event + two-event) for route-based master.

Key assumptions (as you stated):
1) All event-time copies (i,d,tau) are feasible w.r.t. time windows by construction.
2) Route feasibility w.r.t. between-event constraints is nurse-independent.
   (Nurses can always leave early; no earliest-start / latest-finish bound.)
3) Home affects only route cost (home->first, last->home) but not feasibility.

Design:
- Build nurse-agnostic "visit skeletons" (tuples of EventCopy) for each day:
    * idle
    * all single-event
    * two-event where sequence is feasible (time / travel)
- Then instantiate for each nurse-day, computing nurse-specific cost by adding home legs.

This keeps pool generation fast and avoids per-nurse feasibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import random

from .copies import CopyIndex, EventCopy
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

    # caps to control size
    max_single_per_day: Optional[int] = None      # None = all
    max_two_per_day: int = 2000                   # cap per day (two-event skeletons)
    max_routes_per_nurse_day: int = 500           # final per (w,d), including idle

    # two-event generation strategy
    # if True: sample pairs (fast for large days); if False: enumerate all pairs then cap
    sample_two_event_pairs: bool = True
    two_event_pair_samples: int = 20000           # number of random pairs to test per day

    # depot flag behavior (keep simple now)
    # If depot_ok is determined solely by "this route type is depot-compliant",
    # you can set this True to mark all non-idle routes as depot_ok=1.
    # Otherwise leave False and fill in your real logic later.
    force_depot_ok_for_working_routes: bool = True


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
      - instantiated routes from day skeletons (single/two-event)

    Note:
      This function builds day-level skeletons once, then instantiates per nurse.
    """
    rng = random.Random(cfg.seed)

    # 1) Build nurse-agnostic skeletons per day (visits tuples)
    day_skeletons: Dict[int, List[Tuple[EventCopy, ...]]] = build_day_skeletons(problem, copy_index, cfg, rng)

    # 2) Cache skeleton attributes that are nurse-independent:
    #    internal_cost(visits), work(visits), depot_ok(visits)
    internal_cost_cache: Dict[Tuple[EventCopy, ...], float] = {}
    work_cache: Dict[Tuple[EventCopy, ...], int] = {}
    depot_ok_cache: Dict[Tuple[EventCopy, ...], int] = {}

    for d, skels in day_skeletons.items():
        for visits in skels:
            if visits in internal_cost_cache:
                continue
            internal_cost_cache[visits] = compute_internal_travel_cost(problem, visits, cfg)
            work_cache[visits] = compute_work_minutes(problem, visits)
            depot_ok_cache[visits] = compute_depot_ok(problem, visits, cfg)

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

                cost = compute_total_cost_with_home(problem, w, visits, internal_cost_cache[visits], depot_ok_cache[visits])
                work = work_cache[visits]
                depot_ok = depot_ok_cache[visits]

                routes.append(
                    Route(
                        w=w,
                        d=d,
                        visits=visits,
                        cost=cost,
                        work=work,
                        depot_ok=depot_ok,
                    )
                )

                if len(routes) >= cfg.max_routes_per_nurse_day:
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
) -> Dict[int, List[Tuple[EventCopy, ...]]]:
    """
    Build nurse-agnostic skeletons per day:
      - optionally idle (empty visits)
      - all / sampled single-event visits
      - feasible two-event visits (v1, v2) where v2 can follow v1
    """
    day_skeletons: Dict[int, List[Tuple[EventCopy, ...]]] = {}

    for d in range(problem.total_day):
        skels: List[Tuple[EventCopy, ...]] = []

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

        # Dedup skeletons by exact visits tuple
        skels = list(dict.fromkeys(skels))  # preserves order, python3.7+
        day_skeletons[d] = skels

    return day_skeletons


def build_two_event_skeletons(
    problem: ProblemData,
    day_copies: List[EventCopy],
    cfg: RoutePoolConfig,
    rng: random.Random,
) -> List[Tuple[EventCopy, ...]]:
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

    out: List[Tuple[EventCopy, ...]] = []

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
                if len(out) >= cfg.max_two_per_day:
                    break
    else:
        # Enumerate all ordered pairs then cap
        for v1 in day_copies:
            for v2 in day_copies:
                if v1 == v2:
                    continue
                if feasible_pair(v1, v2):
                    out.append((v1, v2))
                    if len(out) >= cfg.max_two_per_day:
                        return out

    # Deduplicate (sampling can produce duplicates)
    out = list(dict.fromkeys(out))
    if len(out) > cfg.max_two_per_day:
        rng.shuffle(out)
        out = out[: cfg.max_two_per_day]
    return out


# -----------------------------
# Route attribute computation
# -----------------------------
def make_idle_route(w: int, d: int) -> Route:
    return Route(w=w, d=d, visits=tuple(), cost=0.0, work=0, depot_ok=0)


def compute_work_minutes(problem: ProblemData, visits: Tuple[EventCopy, ...]) -> int:
    """Sum of service durations (minutes). Adjust if your definition differs."""
    return sum(problem.event_durations[v[0]] for v in visits)


def compute_internal_travel_cost(problem: ProblemData, visits: Tuple[EventCopy, ...], cfg: RoutePoolConfig) -> float:
    """
    Internal travel excludes home legs.
    For now: sum of event->event travel between consecutive visits.
    If you later include depot between events, incorporate it here.
    """
    if len(visits) <= 1:
        return 0.0
    total = 0.0
    for a, b in zip(visits[:-1], visits[1:]):
        total += float(problem.event_event_costs[a[0]][b[0]])

    if cfg.force_depot_ok_for_working_routes:
        # add costs: depot -> first event + last event -> depot
        first_i = visits[0][0]
        last_i = visits[-1][0]
        total += float(problem.event_depot_costs[first_i] + problem.event_depot_costs[last_i])
    return total


def compute_total_cost_with_home(
    problem: ProblemData,
    w: int,
    visits: Tuple[EventCopy, ...],
    internal_cost: float,
    depot_ok: int = 0,
) -> float:
    """Total cost = home->first + internal + last->home (+ depot legs if you later add them)."""
    if len(visits) == 0:
        return 0.0
    first_i = visits[0][0]
    last_i = visits[-1][0]
    if not depot_ok:
        home_legs = float(problem.home_event_costs[w][first_i] + problem.home_event_costs[w][last_i])
    else:
        # goes from home to depot in the morning, then from depot to home in the evening
        home_legs = 2 * float(problem.home_depot_costs[w])
    return home_legs + internal_cost


def compute_depot_ok(problem: ProblemData, visits: Tuple[EventCopy, ...], cfg: RoutePoolConfig) -> int:
    """
    Placeholder depot logic.
    Keep it simple while you get the master running.

    Options:
    - If cfg.force_depot_ok_for_working_routes: mark all non-idle routes as depot_ok=1.
    - Else: mark all routes as depot_ok=0 (and add real logic later).
    """
    if len(visits) == 0:
        return 0
    if cfg.force_depot_ok_for_working_routes:
        return 1
    return 0


# -----------------------------
# ProblemData accessors (adjust these to match your structures)
# -----------------------------
def slot_to_minute(tau: int) -> int:
    return 30 * int(tau)



# -----------------------------
# Utilities
# -----------------------------
def dedup_routes(routes: List[Route]) -> List[Route]:
    """Deduplicate by (w,d,visits)."""
    seen = set()
    out: List[Route] = []
    for r in routes:
        key = (r.w, r.d, r.visits)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
