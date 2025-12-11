import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import pickle

from src.models.problem_data import ProblemData
from src.models.solution import Solution
from src.solver.continuous import continuous_algorithm
from src.solver.extract import routes_from_active_x_t
from src.io.data_loader import load_problem_data


import pandas as pd


def depot_delta_cost(problem_data, sol):
    """
    Given problem data and a solution (possibly with or without depot), evaluate the delta cost to add depot nodes by first constructing solution without depot nodes, then computing delta costs.

    Parameters:
    problem_data (ProblemData): Contains cost matrices and other parameters.
    sol (Solution): Solution object containing routes without depot.

    Returns:
    day_dict (dict): 
        {
          day_idx: {
            nurse_idx: {
              "routes": list of event nodes,
              "delta_cost_AM": float,
              "delta_cost_PM": float
            }, ...
          }, ...
        }
    """
    C_event = problem_data.C_event
    C_home = problem_data.C_home
    C_depot_e = problem_data.C_depot_e
    C_depot_h = problem_data.C_depot_h
    
    day_dict = {}

    for d in range(problem_data.day):
        day_dict[d] = {}
        for w in range(problem_data.n):
            # find route for this nurse on this day
            route_found = None
            for route in sol.iter_routes():
                if route.day_idx == d and route.nurse == w:
                    route_found = route
                    break

            if route_found is None:
                continue  # this nurse didn't work that day

            event_nodes = [node for node in route_found.nodes if 0 <= node < problem_data.m]
            if not event_nodes:
                continue

            first_event = event_nodes[0]
            last_event = event_nodes[-1]

            # current AM/PM costs (home -> first/last event)
            current_AM_cost = C_home[w, first_event]
            current_PM_cost = C_home[w, last_event]

            # new AM/PM costs (depot -> home + depot -> event)
            new_AM_cost = C_depot_h[w] + C_depot_e[first_event]
            new_PM_cost = C_depot_h[w] + C_depot_e[last_event]

            delta_cost_AM = new_AM_cost - current_AM_cost
            delta_cost_PM = new_PM_cost - current_PM_cost

            day_dict[d][w] = {
                "routes": event_nodes,
                "delta_cost_AM": float(delta_cost_AM),
                "delta_cost_PM": float(delta_cost_PM)
            }

    return day_dict

def solve_depot(problem_data, day_dict, mode="AM"):
    """
    Solve the depot assignment problem (AM or PM) using integer programming.

    Parameters
    ----------
    problem_data : object
        Has attributes:
          - n: number of nurses
          - m: number of events
          - day: number of days
    day_dict : dict
        From depot_delta_cost(), structured as:
        {
          day_idx: {
            nurse_idx: {
              "routes": [event indices],
              "delta_cost_AM": float,
              "delta_cost_PM": float
            }, ...
          }, ...
        }
    mode : str, optional
        Either "AM" or "PM" (default: "AM")

    Returns
    -------
    depot_solution : dict
        { day_idx: { nurse_idx: 0/1 } } indicating depot assignment.
    """

    assert mode in ["AM", "PM"], "mode must be 'AM' or 'PM'"

    model = gp.Model(f"depot_{mode}_assignment")
    model.Params.LogToConsole = 0  # silent mode

    # --- Decision variables
    x = {}
    for d, nurses in day_dict.items():
        for w in nurses.keys():
            x[w, d] = model.addVar(vtype=GRB.BINARY, name=f"x_{w}_{d}")

    # --- Objective: minimize total delta cost
    cost_key = f"delta_cost_{mode}"
    obj = gp.quicksum(
        nurses[w][cost_key] * x[w, d]
        for d, nurses in day_dict.items()
        for w in nurses.keys()
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints: event coverage per day
    for d, nurses in day_dict.items():
        for j in range(problem_data.m):
            covered_by = [
                x[w, d]
                for w, info in nurses.items()
                if j in info["routes"]
            ]
            if covered_by:  # event exists that day
                model.addConstr(gp.quicksum(covered_by) >= 1,
                                name=f"cover_d{d}_e{j}")

    # --- Constraints: prevent nurse with identical routes to both be selected
    # for d, nurses in day_dict.items():
    #     groups = {}
    #     for w, info in nurses.items():
    #         key = tuple(info["routes"])
    #         groups.setdefault(key, []).append(w)
    #     for key, wlist in groups.items():
    #         if len(wlist) > 1:
    #             print(f"Day {d} identical routes for nurses {wlist}, only one can have depot")
    #             model.addConstr(
    #                 gp.quicksum(x[w, d] for w in wlist) <= 1,
    #                 name=f"uniq_route_d{d}_grp_{abs(hash(key)) % 10000}"
    #             )

    # start time
    start_time = time.time()
    # --- Optimize
    model.optimize()
    end_time = time.time()
    print(f"Depot {mode} assignment time: {end_time - start_time:.4f} seconds")

    # --- Collect results
    depot_solution = {d: {} for d in day_dict}
    for (w, d), var in x.items():
        depot_solution[d][w] = int(round(var.X))

    total_cost = model.objVal
    print(f"Optimal total delta {mode} cost: {total_cost:.2f}")

    return depot_solution

def add_depot_to_solution(problem_data, sol, depot_solution, mode="AM"):
    """
    Given problem data, a solution (possibly) without depot, and depot assignment,
    return a new solution with depot nodes added.

    Parameters:
    problem_data (ProblemData): Problem data containing parameters.
    sol (Solution): Original solution without depot nodes.
    depot_solution (dict): From solve_depot(), indicating depot assignments.

    Returns:
    new_sol (Solution): New solution with depot nodes added.
    """
    m = problem_data.m
    n = problem_data.n

    DEPOT_AM = m + 1
    DEPOT_PM = m + 2

    new_sol = Solution()

    for route in sol.iter_routes():
        d = route.day_idx
        w = route.nurse
        nodes = route.nodes.copy()
        arrival = route.arrival.copy()
        start = route.start.copy()
        depart = route.depart.copy()

        if mode == "AM":
            # remove existing depot AM if present
            if DEPOT_AM in nodes:
                idx = nodes.index(DEPOT_AM)
                nodes.pop(idx)
                arrival.pop(idx)
                start.pop(idx)
                depart.pop(idx)
            # Add depot AM if assigned
            if depot_solution.get(d, {}).get(w, 0) == 1:
                # for day 2 only
                # if d == 2:
                #     print(f"Adding depot AM for day {d} nurse {w}")
                # if not already present
                if DEPOT_AM not in nodes:
                    nodes.insert(1, DEPOT_AM)  # after home
                    # insert 0 arrival/start/depart times for depot
                    arrival.insert(1, 0)
                    start.insert(1, 0)
                    depart.insert(1, 0)
            # else:
                # print(f"No depot AM for day {d} nurse {w}")
                # print the original nodes
                # print(f"Original nodes: {nodes}")
        elif mode == "PM":
            # remove existing depot PM if present
            if DEPOT_PM in nodes:
                idx = nodes.index(DEPOT_PM)
                nodes.pop(idx)
                arrival.pop(idx)
                start.pop(idx)
                depart.pop(idx)
            # Add depot PM if assigned
            if depot_solution.get(d, {}).get(w, 0) == 1:
                # if d == 2:
                #     print(f"Adding depot PM for day {d} nurse {w}")
                # if not already present
                if DEPOT_PM not in nodes:
                    nodes.insert(-1, DEPOT_PM)  # after home
                    # insert 0 arrival/start/depart times for depot
                    arrival.insert(-1, 1440)
                    start.insert(-1, 1440)
                    depart.insert(-1, 1440)

        new_sol.set_nodes(d, w, nodes)
        new_sol.set_arrival_start_depart(d, w, nodes, arrival, start, depart)

    return new_sol

def greedy_add_depot_and_evaluate(problem_data, sol):
    """
    Given problem data and a solution without depot, add depot nodes and evaluate the cost.
    
    Parameters:
    problem_data (ProblemData): The problem data containing cost matrices and other parameters.
    sol (Solution): The solution object containing routes without depot.
    
    Returns:
    total_cost (float): The total cost after adding depot nodes.
    """
    total_cost = 0.0
    # build a loop for each day

    C_event = problem_data.C_event
    C_home = problem_data.C_home
    C_depot_e = problem_data.C_depot_e
    C_depot_h = problem_data.C_depot_h
    
    for d in range(problem_data.day):
        print(f"Day {d}:")
        rows_events = []
        rows_firstlast = []
        for w in range(problem_data.n):
            found = False
            for route in sol.iter_routes():
                if route.day_idx == d and route.nurse == w:
                    event_nodes = [node for node in route.nodes if 0 <= node < problem_data.m]
                    if event_nodes:
                        first_event = event_nodes[0]
                        last_event = event_nodes[-1]
                        current_AM_cost = C_home[w, first_event]
                        current_PM_cost = C_home[w, last_event]
                        new_AM_cost = C_depot_h[w] + C_depot_e[first_event]
                        new_PM_cost = C_depot_h[w] + C_depot_e[last_event]
                        delta_cost_AM = new_AM_cost - current_AM_cost
                        delta_cost_PM = new_PM_cost - current_PM_cost
                        rows_firstlast.append({
                            "day": d,
                            "nurse": route.nurse,
                            "first_event": first_event,
                            "last_event": last_event,
                            "routes": event_nodes,
                            "delta_cost_AM": delta_cost_AM,
                            "delta_cost_PM": delta_cost_PM
                        })
                    found = True
                    break
            if not found:
                pass
        df2 = pd.DataFrame(rows_firstlast)
        print(df2)

        # --- AM depot cover greedy selection ---
        events_today = set()
        for route in df2["routes"]:
            events_today.update(route)

        uncovered_events = set(events_today)
        selected_nurses = set()
        total_am_cost = 0.0

        df2_am = df2.copy()
        while uncovered_events:
            # Find nurses who cover any uncovered event
            candidates = df2_am[df2_am["routes"].apply(lambda r: any(e in uncovered_events for e in r))]
            if candidates.empty:
                print("Warning: Not all events can be covered for AM!")
                break
            # Select nurse with lowest delta_cost_AM
            best_row = candidates.loc[candidates["delta_cost_AM"].idxmin()]
            total_am_cost += best_row["delta_cost_AM"]
            selected_nurses.add(best_row["nurse"])
            # Remove covered events
            uncovered_events -= set(best_row["routes"])
            # Remove this nurse from future consideration
            df2_am = df2_am[df2_am["nurse"] != best_row["nurse"]]
        print(f"AM depot cover cost (day {d}): {total_am_cost}")

        # --- PM depot cover greedy selection ---
        uncovered_events = set(events_today)
        selected_nurses = set()
        total_pm_cost = 0.0

        df2_pm = df2.copy()
        while uncovered_events:
            candidates = df2_pm[df2_pm["routes"].apply(lambda r: any(e in uncovered_events for e in r))]
            if candidates.empty:
                print("Warning: Not all events can be covered for PM!")
                break
            best_row = candidates.loc[candidates["delta_cost_PM"].idxmin()]
            total_pm_cost += best_row["delta_cost_PM"]
            selected_nurses.add(best_row["nurse"])
            uncovered_events -= set(best_row["routes"])
            df2_pm = df2_pm[df2_pm["nurse"] != best_row["nurse"]]
        print(f"PM depot cover cost (day {d}): {total_pm_cost}")



# ...existing code...
from itertools import combinations

def debug_day_nurses(day_dict, day_idx, nurse_list):
    print(f"\n--- DEBUG day {day_idx} nurses {nurse_list} ---")
    nurses = day_dict.get(day_idx, {})
    if not nurses:
        print("No data for this day in day_dict")
        return

    # Print entries for requested nurses
    for w in nurse_list:
        info = nurses.get(w)
        if info is None:
            print(f"nurse {w}: NOT PRESENT")
        else:
            print(f"nurse {w}: delta_AM={info['delta_cost_AM']:.2f}, delta_PM={info['delta_cost_PM']:.2f}, routes={info['routes']}")

    # Events present that day (from all nurses in day_dict)
    events_today = set()
    for info in nurses.values():
        events_today.update(info["routes"])
    print("events_today:", sorted(events_today))

    # Evaluate all subsets of the specified nurse_list (and also full set of nurses that day)
    relevant_nurses = sorted(nurses.keys())
    print("\nBrute-force best among ALL nurses that day:")
    best_all = (float('inf'), None)
    for r in range(1, len(relevant_nurses)+1):
        for subset in combinations(relevant_nurses, r):
            covered = set()
            cost = 0.0
            for w in subset:
                covered |= set(nurses[w]["routes"])
                cost += nurses[w]["delta_cost_AM"]  # use AM or PM as needed
            if covered >= events_today and cost < best_all[0]:
                best_all = (cost, subset)
    print("best_all (AM):", best_all)

    # Now specifically evaluate subsets of the provided nurse_list only
    print(f"\nEvaluate subsets of {nurse_list} (AM):")
    best_spec = (float('inf'), None)
    for r in range(0, len(nurse_list)+1):
        for subset in combinations(nurse_list, r):
            covered = set()
            cost = 0.0
            for w in subset:
                info = nurses.get(w)
                if info is None:
                    cost = float('inf'); break
                covered |= set(info["routes"])
                cost += info["delta_cost_AM"]
            print(f"subset={subset} covers={sorted(covered)} cost={cost:.2f}")
            if covered >= events_today and cost < best_spec[0]:
                best_spec = (cost, subset)
    print("best among specified nurses (AM):", best_spec)

    # Repeat for PM if desired
    print("\nEvaluate subsets of {0} (PM):".format(nurse_list))
    best_spec_pm = (float('inf'), None)
    for r in range(0, len(nurse_list)+1):
        for subset in combinations(nurse_list, r):
            covered = set()
            cost = 0.0
            for w in subset:
                info = nurses.get(w)
                if info is None:
                    cost = float('inf'); break
                covered |= set(info["routes"])
                cost += info["delta_cost_PM"]
            print(f"subset={subset} covers={sorted(covered)} cost={cost:.2f}")
            if covered >= events_today and cost < best_spec_pm[0]:
                best_spec_pm = (cost, subset)
    print("best among specified nurses (PM):", best_spec_pm)

# Usage: after you compute day_dict
# day_dict = depot_delta_cost(problem_data, sol)
# debug_day_nurses(day_dict, 2, [25, 26])
# ...existing code...