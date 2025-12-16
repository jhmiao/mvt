import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time

from src.structures.problem_data import ProblemData
from src.heuristics.config import HeuristicConfig
from src.solutions.solutions import Solution

from typing import TYPE_CHECKING, List, Dict, Tuple, Any



def _status_to_str(status_code: int) -> str:
    lookup = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return lookup.get(status_code, str(status_code))


def assign_nurses_with_depot(
    data: ProblemData,
    day_schedule: Dict[Dict[int, int], Dict[Tuple[int, int], Any]],
    day: int,
    config: HeuristicConfig | None = None,
):
    """
    Assign nurses for a single day's schedule and return a Solution object.
    `schedule` maps event -> start slot (30-minute blocks).
    """

    chosen_schedule = day_schedule[list(day_schedule.keys())[0]]
    feasible_arcs = day_schedule[list(day_schedule.keys())[1]]

    cfg = config or HeuristicConfig()
    nurses = range(data.total_nurse)

    # discretized travel & duration (30 min)
    C_event = data.event_event_costs   # C_event[i,j] = travel time from event i to event j
    C_home = data.home_event_costs     # C_home[w,i] = travel time from home of w to event i
    C_depot_e = data.event_depot_costs   # C_depot[i] = travel time from depot to event i
    C_depot_h = data.home_depot_costs   # C_depot[h] = travel time from home to depot

    # Build trip patterns for this day
    trips = set()

    # Add 2-event feasible trips (convert list -> tuple for hashing)
    day_feasible = feasible_arcs.get(day, []) if isinstance(feasible_arcs, dict) else feasible_arcs
    for pair in day_feasible:
        i, j = pair
        trips.add((i, j))

    events_today = list(chosen_schedule.keys())

    # Add 1-event trips
    for e in events_today:
        trips.add((e,))

    # create a copy of trips with depot added (add n at the start, add n+1 at the end for each trip)
    trips_with_depot = set()

    depot_am = data.total_event
    depot_pm = data.total_event + 1

    for trip in trips:
        trip_with_depot = (depot_am,) + trip + (depot_pm,)
        trips_with_depot.add(trip_with_depot)

    # add trips_with_depot to trips
    trips.update(trips_with_depot)

    # Compute cost for each nurse and each trip
    cost = {}
    for n in nurses:
        for trip in trips:
            if len(trip) == 1:
                (e,) = trip
                # home -> event -> home
                c = C_home[n][e] + C_home[n][e]
            elif len(trip) == 2:
                i, j = trip
                # home -> i -> j -> home
                c = C_home[n][i] + C_event[i][j] + C_home[n][j]
            elif len(trip) == 3:
                i, j, k = trip
                if i == depot_am and k == depot_pm:
                    # home -> depot -> j -> depot -> home
                    c = C_depot_h[n] + C_depot_e[j] + C_depot_e[j] + C_depot_h[n]
                else:
                    raise ValueError("Invalid trip with depot.")
            elif len(trip) == 4:
                i, j, k, l = trip
                if i == depot_am and l == depot_pm:
                    # home -> depot -> j -> k -> depot -> home
                    c = C_depot_h[n] + C_depot_e[j] + C_event[j][k] + C_depot_e[k] + C_depot_h[n]
                else:
                    raise ValueError("Invalid trip with depot.")
            else:
                raise ValueError("Trip length must be 1 or 2.")

            cost[(n, trip)] = c

    m = gp.Model()

    # Use a single index list to avoid Gurobi flattening tuple trip indices
    x = m.addVars([(n, trip) for n in nurses for trip in trips], vtype=GRB.BINARY, name="x")

    # Each nurse takes at most one trip
    for n in nurses:
        m.addConstr(gp.quicksum(x[n, t] for t in trips) <= 1)

    for e in events_today:
        required_rn = data.min_nurses[e][0]
        required_lvn = data.min_nurses[e][1]

        m.addConstr(
            gp.quicksum(
                x[n, t] for n in range(data.total_rn) for t in trips if e in t  # RNs
            )
            >= required_rn,
            name=f"event_RN_demand_day{day}_event{e}",
        )
        m.addConstr(
            gp.quicksum(
                x[n, t] for n in range(data.total_rn, data.total_rn + data.total_lvn) for t in trips if e in t  # LVNs
            )
            >= required_lvn,
            name=f"event_LVN_demand_day{day}_event{e}",
        )

        # Add depot constraints
        m.addConstr(
            gp.quicksum(
                x[n, t] for n in nurses for t in trips if e in t and (t[0] == depot_am and t[-1] == depot_pm)
            )
            >= 1,
            name=f"event_depot_limit_day{day}_event{e}",
        )

    # Objective: min sum(cost(n, t) * x[n,t])
    m.setObjective(gp.quicksum(cost[n, t] * x[n, t] for n in nurses for t in trips), GRB.MINIMIZE)

    m.setParam("OutputFlag", getattr(cfg, "gurobi_outputflag", 0))
    if getattr(cfg, "time_limit", None) is not None:
        m.setParam("TimeLimit", cfg.time_limit)
    if getattr(cfg, "work_limit", None) is not None:
        m.setParam("WorkLimit", cfg.work_limit)

    m.optimize()

    status_str = _status_to_str(m.Status)
    try:
        lower_bound = float(m.ObjBound)
    except gp.GurobiError:
        lower_bound = None

    if m.Status == GRB.INFEASIBLE:
        return Solution(
            day=day,
            assignments={},
            leaders={},
            start_times={},
            objective_value=float("inf"),
            lower_bound=lower_bound,
            status=status_str,
            extra={"message": "Infeasible nurse assignment"},
        )

    chosen = [(n, t) for n in nurses for t in trips if x[n, t].X > 0.5]

    assignments = {e: [] for e in events_today}
    leaders = {}
    for n, trip in chosen:
        has_depot_am = depot_am in trip
        has_depot_pm = depot_pm in trip
        for node in trip:
            if node in (depot_am, depot_pm):
                continue
            assignments.setdefault(node, []).append(n)
            if has_depot_am and node not in leaders:
                leaders[node] = {"pickup": n, "dropoff": None}
            if has_depot_pm:
                leaders.setdefault(node, {"pickup": None, "dropoff": None})
                if leaders[node].get("dropoff") is None:
                    leaders[node]["dropoff"] = n

    start_times = {e: 570 + 30 * slot for e, slot in chosen_schedule.items()}

    return Solution(
        day=day,
        assignments=assignments,
        leaders=leaders,
        start_times=start_times,
        objective_value=float(m.objVal) if m.SolCount else float("nan"),
        lower_bound=lower_bound,
        status=status_str,
        extra={"feasible_arcs": feasible_arcs,
               "chosen_trips": chosen},
    )
