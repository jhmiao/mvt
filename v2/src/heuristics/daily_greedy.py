# src/heuristics/by_day.py

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

from src.models.problem_data import ProblemData
from src.solver.continuous import continuous_algorithm

from typing import TYPE_CHECKING, List, Dict, Tuple, Any
import pickle

# if TYPE_CHECKING:
#     from src.models.solution import Solution, Route
#     from src.models.context import Context
    

def continuous_algorithm_heuristic (data: ProblemData, work_limit, seed_number, event_limit=None, pruning=1, min_hour=0):
    """
    Continuous algorithm for the MVT scheduling problem broken down by day.
    Parameters: 
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    """
    
    C_event = data.C_event
    C_home = data.C_home
    C_depot_e = data.C_depot_e
    C_depot_h = data.C_depot_h
    # C_depot = np.concatenate([C_depot_e, C_depot_h])
    C_dur = data.C_dur
    time_window = data.time_window
    min_nurse = data.min_nurse
    nr = data.nr
    nl = data.nl
    n = data.n
    m = data.m
    days = data.day

    day_event_counts = []
    for run_id in range(days):
        events_today = np.where(time_window[:, run_id, 1] > 0)[0]
        m_today = len(events_today)
        day_event_counts.append(m_today)
    
    day_weights = [count / m for count in day_event_counts]

    current_hours = {w: 0.0 for w in range(n)}  # Initialize for all nurses
    total_objective = 0.0
    # cumu_hours_bound = 0.0

    master = {
        "active_x":       [],
        "active_s":       [],
        "active_t":       {},
        "active_alpha":   [],
        "active_beta":    [],
        "objective_value": [],
        "runtime_sec":     [],
        "gap":             [],
        "work_time_by_nurse": current_hours
    }
    
    # custom_day_order = [3, 2, 4, 0, 1]

    # define custom day order by descending number of events
    custom_day_order = sorted(range(days), key=lambda d: day_event_counts[d], reverse=True)
    print("Custom day order:", custom_day_order)

    day_count = 0

    scheduled_events = set()

    for run_id in custom_day_order:

        day_count += 1

        # print("Max hours for day", d+1, ":", max_hour)

        # filter data for day d
        # get the index of events scheduled on day d: where time_window[:, d, 1] > 0
        events_today = np.where(time_window[:, run_id, 1] > 0)[0]

        # Compute future_event_indices as those not yet scheduled
        all_events = set(np.arange(m))
        future_event_indices = np.array(list(all_events - scheduled_events - set(events_today)))

        # filter C_event, C_home, C_depot, C_dur, time_window, min_nurse by events_today
        C_event_today = C_event[np.ix_(events_today, events_today)]
        C_home_today = C_home[:, events_today]
        C_depot_e_today = C_depot_e[events_today]
        C_dur_today = C_dur[events_today]
        time_window_today = time_window[events_today, run_id, :].reshape((len(events_today), 1, 2))
        min_nurse_today = min_nurse[events_today, :]

        m_today = len(events_today)

        # create a new ProblemData object for day d
        data_today = ProblemData(
            C_event=C_event_today,
            C_home=C_home_today,
            C_depot_e=C_depot_e_today,
            C_depot_h=C_depot_h,
            C_dur=C_dur_today,
            time_window=time_window_today,
            min_nurse=min_nurse_today,
            nurse_type=data.nurse_type,
            nr=nr,
            nl=nl,
            n=n,
            m=m_today,
            day=1  # only one day
        )

        total_RN_hours = np.sum(C_dur[:m_today] * min_nurse_today[:m_today, 0])
        total_LVN_hours = np.sum(C_dur[:m_today] * min_nurse_today[:m_today, 1])

        total_RN_hours /= 60
        total_LVN_hours /= 60

        avg_RN_hours = total_RN_hours / nr if nr > 0 else 0
        avg_LVN_hours = total_LVN_hours / nl if nl > 0 else 0

        print(f"\n\nRunning continuous algorithm for day {run_id+1} with {len(events_today)} events, {events_today}...")
        print(f"shape of C_event_today: {C_event_today.shape}, C_home_today: {C_home_today.shape}, C_depot_e_today: {C_depot_e_today.shape}, C_dur_today: {C_dur_today.shape}, time_window_today: {time_window_today.shape}, min_nurse_today: {min_nurse_today.shape}")
        print(f"Average RN hours: {avg_RN_hours}, Average LVN hours: {avg_LVN_hours}")

        # Call the continuous_algorithm function with the filtered data

        allocated_time = work_limit * day_weights[run_id]

        max_hour = {w: 25 - current_hours.get(w, 0.0) for w in range(n)}

        # define a subroutine to modify max_hour
        # new_max_hour = modify_max_hour_reserve_future_days(max_hour, day_count)
        # new_max_hour = modify_max_hour_remaining_weights(max_hour, day_count, day_weights)
        new_max_hour = modify_max_hour_by_nurse(max_hour, C_home, events_today, future_event_indices, scale=5, k=4)
        # new_max_hour = max_hour

        summary = continuous_algorithm(
            data_today, 
            work_limit=allocated_time, 
            seed_number=seed_number, 
            multiple_tw=None, 
            event_limit=None,
            pruning=pruning, 
            min_hour=None, 
            max_hour=new_max_hour
        )

        work_time_by_nurse = summary.get("work_time_by_nurse", {}) # new work time
        # Check for infeasibility (you may need to adapt this check to your summary structure)
        if not work_time_by_nurse or summary.get("objective_value", None) is None:
            print("No feasible solution found with current max_hour. Retrying with relaxed max_hour...")
            # Relax max_hour: e.g., don't reserve any hours, or increase by a factor
            relaxed_max_hour = max_hour
            summary = continuous_algorithm(
                data_today,
                work_limit=allocated_time,
                seed_number=seed_number,
                multiple_tw=None,
                event_limit=None,
                pruning=1,
                min_hour=None,
                max_hour=relaxed_max_hour
            )
            work_time_by_nurse = summary.get("work_time_by_nurse", {})
    
        scheduled_events.update(events_today)
        for w in range(n):
            # current_hours[w] += work_time_by_nurse.get(w, 0.0)
            current_hours[w] = current_hours.get(w, 0.0) + work_time_by_nurse.get(w, 0.0)

        # append this day's summary to the overall summary

        print(f"Objective value: {summary['objective_value']}, Current hours: {current_hours}")

        total_objective = summary['objective_value'] + total_objective

        # add home and depot to events_today
        event_index = np.concatenate([events_today, [m, m+1, m+2]])
        # update the day entry of this round of summary to be d
        summary["active_x"]      = [ (event_index[i], event_index[j], run_id, w) for (i,j,d,w) in summary["active_x"] ]
        summary["active_s"]      = [ (event_index[i],run_id)   for (i,d)   in summary["active_s"] ]
        summary["active_alpha"]  = [ (event_index[i],run_id,w)   for (i,d,w) in summary["active_alpha"] ]
        summary["active_beta"]   = [ (event_index[i],run_id,w)   for (i,d,w) in summary["active_beta"] ]
        summary["active_t"]      = { (event_index[i],run_id): t_val for ((i, d), t_val) in summary["active_t"].items()}
        master["active_x"].extend(      summary["active_x"]      )
        master["active_s"].extend(      summary["active_s"]      )
        master["active_alpha"].extend(  summary["active_alpha"]  )
        master["active_beta"].extend(   summary["active_beta"]   )
        master["active_t"].update(summary["active_t"])

        master["objective_value"] = total_objective
        master["work_time_by_nurse"] = current_hours
        master["runtime_sec"].append(  summary.get("runtime_sec", 0.0)  )

        # event_id += m_today

    return master


def modify_max_hour_reserve_future_days(max_hour, day_count):
    """
    Modify the max_hour dictionary to reserve some time for future days.
    """
    # Example: reserve 3 hours per remaining day
    future_days = 5 - day_count  # assuming max_hour has one entry per nurse
    reserve = 3 * future_days  # reserve 3 hours per remaining day (tune as needed)
    return {w: max(0, max_hour[w] - reserve) for w in max_hour}


def modify_max_hour_remaining_weights(max_hour, day_count, day_weights):
    """
    Modify the max_hour dictionary to reserve some time for future days,
    using the sum of the smallest (5 - day_count) day_weights.
    """
    num_reserve_days = max(0, 5 - day_count)
    # Get the smallest (5 - day_count) day_weights
    smallest_weights = sorted(day_weights)[:num_reserve_days]
    reserve = sum(smallest_weights) * 15  # or multiply by a suitable hour factor

    return {w: max(0, max_hour[w] - reserve) for w in max_hour}


def modify_max_hour_by_nurse(max_hour, C_home, events_today, future_event_indices, scale=5, k=3):
    """
    Prioritize nurses whose travel distance to the k nearest events_today is short compared to their distance to the k nearest future_event_indices.
    Args:
        max_hour: dict of {nurse: hour}
        C_home: 2D array, shape (n_nurses, n_events), distance from each nurse to each event
        events_today: list/array of indices of today's events
        future_event_indices: list/array of indices of future unscheduled events
        scale: scaling factor for how much to adjust based on the ratio
        k: number of nearest events to consider
    Returns:
        new_max_hour: dict of {nurse: adjusted hour}
    """
    print("future_event_indices:", future_event_indices)
    n_nurses = C_home.shape[0]
    ratios = []
    for w in range(n_nurses):
        # Average of k nearest events_today
        if len(events_today) == 0:
            avg_today = 0
        else:
            dists_today = np.array([C_home[w, e] for e in events_today])
            k_today = min(k, len(dists_today))
            avg_today = np.mean(np.partition(dists_today, k_today-1)[:k_today])
            # avg_today = np.mean(dists_today)

        # Average of k nearest future events
        if len(future_event_indices) == 0:
            avg_future = avg_today # avoid division by zero
        else:
            dists_future = np.array([C_home[w, e] for e in future_event_indices])
            # k_future = min(k, len(dists_future))
            k_future = min(k * len(dists_future) // max(1, len(events_today)), len(dists_future))  # scale k based on number of future events
            avg_future = np.mean(np.partition(dists_future, k_future-1)[:k_future])
            # avg_future = np.mean(dists_future)

        ratio = avg_today / avg_future
        ratios.append(ratio)

    # Normalize ratios so that lower ratios (nurse is close to today, far from future) get higher max_hour
    ratios = np.array(ratios)
    if ratios.max() > ratios.min():
        norm_ratios = (ratios.max() - ratios) / (ratios.max() - ratios.min())
    else:
        norm_ratios = np.ones_like(ratios)

    print("Nurse proximity ratios (lower means closer to today's events):\n", ratios)
    new_max_hour = {}
    for w in max_hour:
        # If the original ratio is less than 1, do not modify max_hour for this nurse
        if ratios[w] < 1.2:
            reserve = 0
        else:
            reserve = scale * norm_ratios[w]
        new_max_hour[w] = max(0, max_hour[w] - reserve)
        # reserve = scale * norm_ratios[w]
        # new_max_hour[w] = max(0, max_hour[w] - reserve)
    return new_max_hour
