import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import time

from src.models.problem_data import ProblemData
from src.models.solution import Solution, Route
from src.solver.continuous import continuous_algorithm

from typing import TYPE_CHECKING, List, Dict, Tuple, Any
import pickle


def assign_nurses(data: ProblemData, nurse_type, chosen_schedules, feasible_arcs):
    
    if nurse_type == 'RN':
        nurses = range(data.nr)
    elif nurse_type == 'LVN':
        nurses = range(data.nr, data.nr + data.nl)

    days = data.day

    # original travel & duration (30 min)
    C_event = data.C_event   # C_event[i,j] = travel time from event i to event j
    C_home = data.C_home     # C_home[w,i] = travel time from home of w to event i
    time_window = data.time_window  # shape (m, day, 2)

    all_assignments = []
    objective_value = 0

    for day in range(days):
        # Build trip patterns for this day
        trips = set()

        # Add 2-event feasible trips (convert list -> tuple for hashing)
        for pair in feasible_arcs.get(day, []):
            i, j = pair
            trips.add((i, j))

        # collect events_today from time_window
        events_today = [i for i in range(data.m) if time_window[i,day,0] > 0]

        # Add 1-event trips
        for e in events_today:
            trips.add((e,))
        # print(f"Day {day}, #events={len(events_today)}, trips = {trips}")

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
                else:
                    raise ValueError("Trip length must be 1 or 2.")

                cost[(n, trip)] = c

        m = gp.Model()

        # Use a single index list to avoid Gurobi flattening tuple trip indices
        x = m.addVars([(n, trip) for n in nurses for trip in trips],
                      vtype=GRB.BINARY, name="x")

        # Each nurse takes at most one trip
        for n in nurses:
            m.addConstr(gp.quicksum(x[n, t] for t in trips) <= 1)

        for e in events_today:
            if nurse_type == 'RN':
                required = data.min_nurse[e][0]
            elif nurse_type == 'LVN':
                required = data.min_nurse[e][1]

            m.addConstr(
                gp.quicksum(
                    x[n, t]
                    for n in nurses
                    for t in trips
                    if e in t
                ) == required,
                name=f"event_{nurse_type}_demand_day{day}_event{e}"
            )

        # Objective: min sum(cost(n, t) * x[n,t])
        m.setObjective(
            gp.quicksum(cost[n, t] * x[n, t] for n in nurses for t in trips),
            GRB.MINIMIZE
        )
        m.setParam('OutputFlag', 0)
        # start time
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        print(f"{nurse_type} assignment time: {end_time - start_time:.4f} seconds")

        day_assignments = [(n, t) for n in nurses for t in trips if x[n, t].x > 0.5]
        all_assignments.append(day_assignments)
        
        objective_value += m.objVal


    print(f"{nurse_type} assignment objective value: {objective_value}")

    return all_assignments

def merge_assignments(assignments_rn, assignments_lvn):
    merged_assignments = []
    # Both assignments_rn and assignments_lvn are lists of length D (days), D = 5
    # for each day, append the assignments from both nurse types
    for day in range(len(assignments_rn)):
        day_assignments = []
        day_assignments.extend(assignments_rn[day])
        day_assignments.extend(assignments_lvn[day])
        merged_assignments.append(day_assignments)

    return merged_assignments


def solution_from_assignments(data: ProblemData,
                              assignments, chosen_schedules) -> Solution:
    """
    Build a Solution from the (day -> nurse, trip) output of assign_nurses.

    Parameters
    ----------
    data : ProblemData
        Source data (only C_dur, nr/nl indexing are used here).
    assignments : List[List[Tuple[int, Tuple[int, ...]]]]
        Per-day assignments; each element is a list of (nurse_idx, trip) where
        trip is a tuple of one or two event indices.
    chosen_schedules : Dict
        Dict day -> {event_idx: start_slot}. Start slots are 30-minute blocks
        from a base of 570 minutes (09:30), consistent with continuous solver.
    """
    sol = Solution()
    base_minutes = 570  # align with continuous_algorithm_fix_events hinting

    def get_start(day_idx: int, event_idx: int) -> int:
        # allow both int and str keys coming from JSON
        day_sched = chosen_schedules.get(day_idx, {}) or chosen_schedules.get(str(day_idx), {})
        if event_idx in day_sched:
            return int(day_sched[event_idx]) * 30 + base_minutes
        if str(event_idx) in day_sched:
            return int(day_sched[str(event_idx)]) * 30 + base_minutes
        return base_minutes

    D = getattr(data, "day", len(assignments) if assignments is not None else 0)
    n = getattr(data, "n", 0)
    assignments = assignments or []

    for d in range(D):
        day_assignments = assignments[d] if d < len(assignments) else []
        nurse_trip = {int(w): tuple(trip) for w, trip in day_assignments}

        for w in range(n):
            home = 100 + w
            nodes = [home]
            arrival = [0]
            start = [0]
            depart = [0]

            trip = nurse_trip.get(w, ())
            for event in trip:
                e = int(event)
                st = get_start(d, e)
                nodes.append(e)
                arrival.append(st)
                start.append(st)
                dur = int(data.C_dur[e]) if hasattr(data, "C_dur") else 0
                depart.append(st + dur)

            nodes.append(home)
            arrival.append(1440)
            start.append(1440)
            depart.append(1440)

            sol.day_routes[(d, w)] = Route(
                day_idx=d,
                nurse=w,
                nodes=nodes,
                arrival=arrival,
                start=start,
                depart=depart,
            )

    return sol


def assign_nurses_with_depot(data: ProblemData, nurse_type, chosen_schedules, feasible_arcs):
    
    if nurse_type == 'RN':
        nurses = range(data.nr)
    elif nurse_type == 'LVN':
        nurses = range(data.nr, data.nr + data.nl)
    elif nurse_type == 'Both':
        nurses = range(data.nr + data.nl)

    days = data.day

    # discretized travel & duration (30 min)
    C_event = data.C_event   # C_event[i,j] = travel time from event i to event j
    C_home = data.C_home     # C_home[w,i] = travel time from home of w to event i
    C_depot_e = data.C_depot_e   # C_depot[i] = travel time from depot to event i
    C_depot_h = data.C_depot_h   # C_depot[h] = travel time from home to depot
    time_window = data.time_window  # shape (m, day, 2)

    all_assignments = []
    objective_value = 0

    for day in range(days):
        # Build trip patterns for this day
        trips = set()

        # Add 2-event feasible trips (convert list -> tuple for hashing)
        for pair in feasible_arcs.get(day, []):
            i, j = pair
            trips.add((i, j))

        # collect events_today from time_window
        events_today = [i for i in range(data.m) if time_window[i,day,0] > 0]

        # Add 1-event trips
        for e in events_today:
            trips.add((e,))
        # print(f"Day {day}, #events={len(events_today)}, trips = {trips}")

        # create a copy of trips with depot added (add n at the start, add n+1 at the end for each trip)
        trips_with_depot = set()

        depot_am = data.m
        depot_pm = data.m + 1

        for trip in trips:
            trip_with_depot = (depot_am,) + trip + (depot_pm,)
            trips_with_depot.add(trip_with_depot)
        # print('trips_with_depot:', trips_with_depot)

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
        x = m.addVars([(n, trip) for n in nurses for trip in trips],
                      vtype=GRB.BINARY, name="x")

        # Each nurse takes at most one trip
        for n in nurses:
            m.addConstr(gp.quicksum(x[n, t] for t in trips) <= 1)

        for e in events_today:
            if nurse_type == 'RN':
                required = data.min_nurse[e][0]
            elif nurse_type == 'LVN':
                required = data.min_nurse[e][1]
            elif nurse_type == 'Both':  # Both
                required_rn = data.min_nurse[e][0]
                required_lvn = data.min_nurse[e][1]

            if nurse_type != 'Both':
                m.addConstr(
                    gp.quicksum(
                        x[n, t]
                        for n in nurses
                        for t in trips
                        if e in t
                    ) == required,
                    name=f"event_{nurse_type}_demand_day{day}_event{e}"
                )
            elif nurse_type == 'Both':
                m.addConstr(
                    gp.quicksum(
                        x[n, t]
                        for n in range(data.nr)  # RNs
                        for t in trips
                        if e in t
                    ) >= required_rn,
                    name=f"event_RN_demand_day{day}_event{e}"
                )
                m.addConstr(
                    gp.quicksum(
                        x[n, t]
                        for n in range(data.nr, data.nr + data.nl)  # LVNs
                        for t in trips
                        if e in t
                    ) >= required_lvn,
                    name=f"event_LVN_demand_day{day}_event{e}"
                )
            
            # Add depot constraints
            m.addConstr(
                gp.quicksum(
                    x[n, t]
                    for n in nurses
                    for t in trips
                    if e in t and (t[0] == depot_am and t[-1] == depot_pm)
                ) >= 1,
                name=f"event_{nurse_type}_depot_limit_day{day}_event{e}"
            )

        # Objective: min sum(cost(n, t) * x[n,t])
        m.setObjective(
            gp.quicksum(cost[n, t] * x[n, t] for n in nurses for t in trips),
            GRB.MINIMIZE
        )
        m.setParam('OutputFlag', 0)
        # start time
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        # print(f"{nurse_type} assignment time: {end_time - start_time:.4f} seconds")
        if m.Status == GRB.OPTIMAL:
            day_assignments = [(n, t) for n in nurses for t in trips if x[n, t].x > 0.5]
            objective_value += m.objVal

        elif m.Status == GRB.INFEASIBLE:
            print(f"{nurse_type} assignment problem is infeasible for day {day}.")
            day_assignments = None
            return None
        
        all_assignments.append(day_assignments)  # Append empty assignments for this day


    print(f"{nurse_type} assignment objective value: {objective_value}")

    # return all_assignments
    return objective_value

def assign_nurses_with_depot_day(data: ProblemData, nurse_type, day, day_chosen_schedules, day_feasible_arcs):
