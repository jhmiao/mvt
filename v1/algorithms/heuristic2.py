import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from data_loader import ProblemData
import pickle

def greedy_event_clustering(data: ProblemData, max_day_time=480):
    """
    Greedily cluster events into groups that can be scheduled within a single day.
    Args:
        data (ProblemData): The problem data containing travel costs, time windows, and nurse requirements.
        max_day_time (int): Maximum time allowed for a single day's schedule in minutes.
    Returns:
        dict: Clusters of events for each day, where keys are day indices and values are lists of event IDs.
    """

    C_event = data.C_event
    C_dur = data.C_dur
    min_nurse = data.min_nurse
    time_window = data.time_window
    m, num_days, _ = time_window.shape

    day_assignment = get_event_day_assignments(time_window)

    clusters_by_day = {}

    for d in range(num_days):
        # Get indices of events assigned to this day
        day_indices = np.where(day_assignment == d)[0]
        if len(day_indices) == 0:
            continue

        df = pd.DataFrame({
            'EventID': day_indices,
            'Duration': C_dur[day_indices],
            'RN_Needed': min_nurse[day_indices, 0],
            'LVN_Needed': min_nurse[day_indices, 1]
        })

        # Sort by priority
        df_sorted = df.sort_values(by=['Duration', 'RN_Needed', 'LVN_Needed'], ascending=[False, False, False]).reset_index(drop=True)

        # Greedy clustering
        used = set()
        clusters = []

        for i, row in df_sorted.iterrows():
            if row['EventID'] in used:
                continue

            cluster = [row['EventID']]
            used.add(row['EventID'])
            total_time = row['Duration']
            current_idx = row['EventID']

            while True:
                candidates = []
                for j in range(len(df_sorted)):
                    next_id = df_sorted.loc[j, 'EventID']
                    if next_id in used:
                        continue

                    travel_time = C_event[current_idx, next_id]
                    next_duration = df_sorted.loc[j, 'Duration']
                    cost = travel_time + next_duration

                    if total_time + cost <= max_day_time:
                        candidates.append((cost, next_id, j))

                if not candidates:
                    break  # No more feasible insertions

                # Pick the best candidate: minimum travel + duration
                _, next_id, next_idx = min(candidates, key=lambda x: x[0])

                cluster.append(next_id)
                used.add(next_id)
                total_time += C_event[current_idx, next_id] + df_sorted.loc[next_idx, 'Duration']
                current_idx = next_id

            clusters.append(cluster)
        
        # reinsertion for small clusters
        # Post-processing: merge singleton clusters
        new_clusters = []
        singletons = []

        for cluster in clusters:
            if len(cluster) == 1:
                singletons.append(cluster[0])
            else:
                new_clusters.append(cluster)

        for singleton in singletons:
            inserted = False
            for i, cluster in enumerate(new_clusters):
                for insert_pos in range(len(cluster) + 1):
                    new_cluster = cluster[:insert_pos] + [singleton] + cluster[insert_pos:]
                    # Estimate total time of new_cluster
                    total_time = estimate_cluster_time(new_cluster, df_sorted, C_event)
                    if total_time <= max_day_time:
                        new_clusters[i] = new_cluster
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                # Couldn't insert into any cluster → keep as singleton
                new_clusters.append([singleton])
        
        print(f"Day {d+1}: Found {len(new_clusters)} clusters")
        # for new_cluster in new_clusters:
        #     print(f"{new_cluster}, Total Time = {estimate_cluster_time(new_cluster, df_sorted, C_event)}")

        clusters_by_day[d] = new_clusters

    return clusters_by_day

def get_event_day_assignments(time_window: np.ndarray) -> np.ndarray:
    """
    Assign each event to a day based on its time window.
    Args:
        time_window: shape (m, day, 2), where [:, :, 0] is nonzero only for the assigned day
    Returns: array of shape (m,), with day index assigned to each event
    """
    day_assignment = (time_window[:, :, 0] != 0).argmax(axis=1)
    return day_assignment

def estimate_cluster_time(cluster, df_sorted, travel_time_matrix):
    """
    Estimate the total time required for a cluster of events.
    Args:
        cluster: List of event IDs in the cluster.
        df_sorted: DataFrame containing event durations.
        travel_time_matrix: Matrix of travel times between events.
    Returns:
        int: Estimated total time for the cluster.
    """
    total_time = 0
    for i in range(len(cluster)):
        event_id = cluster[i]
        duration = df_sorted.loc[df_sorted['EventID'] == event_id, 'Duration'].values[0]
        total_time += duration
        if i > 0:
            prev_id = cluster[i - 1]
            total_time += travel_time_matrix[prev_id, event_id]
    return total_time

def assign_nurse_to_cluster(clusters_by_day, min_nurse, nr, nl):
    """
    Assign nurses to a cluster of events based on minimum requirements.
    Args:
        cluster: List of event IDs in the cluster.
        min_nurse: Minimum nurse requirements for each event.
        nr: Number of regular nurses.
        nl: Number of late nurses.
    Returns:
        dict: Nurse assignments for the cluster.
    """
    cluster_requirements = {}  # (day, cluster_idx) -> (max_rn, max_lvn)

    for day, clusters in clusters_by_day.items():
        for cluster_idx, cluster in enumerate(clusters):
            rn_needed = max(min_nurse[event_id, 0] for event_id in cluster)
            lvn_needed = max(min_nurse[event_id, 1] for event_id in cluster)
            cluster_requirements[(day, cluster_idx)] = (rn_needed, lvn_needed)
            print(f"Day {day+1}, Cluster {cluster_idx+1}: RN Needed = {rn_needed}, LVN Needed = {lvn_needed}")

def heuristic_combine(data: ProblemData, max_day_time=480):
    """
    Main function to run the greedy event clustering and nurse assignment.
    Args:
        data (ProblemData): The problem data containing travel costs, time windows, and nurse requirements.
        max_day_time (int): Maximum time allowed for a single day's schedule in minutes.
    Returns:
        dict: Clusters of events for each day with nurse assignments.
    """
    clusters_by_day = greedy_event_clustering(data, max_day_time)
    assign_nurse_to_cluster(clusters_by_day, data.min_nurse, data.nr, data.nl)
    return clusters_by_day
