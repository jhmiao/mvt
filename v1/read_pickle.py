import pickle
import pandas as pd

# file_path = '/Users/jinghongmiao/Code/mvt-code/result-250805/ca_r12_l12_m15_wl1000_max40_18.pkl'
file_path2 = '/Users/jinghongmiao/Code/mvt-code/result-250812/h_r20_l30_m50_wl1000_max25_19.pkl'

with open(file_path2, "rb") as f:
    summary = pickle.load(f)

# now you can inspect what’s inside:
print("Active x:", summary["active_x"])
# print("Active s:", summary["active_s"])
# print("Active t:", summary["active_t"])
# print("Active alpha:", summary["active_alpha"])
# print("Active beta:", summary["active_beta"])
# print("Work time by nurse:", summary["work_time_by_nurse"])
print("Objective value:", summary["objective_value"])

nr = 20
nl = 30
m = 50
n = nr + nl
day = 5

setting_file = 'c101_data.xlsx'

# Read C_travel
C_event = pd.read_excel(setting_file, sheet_name='C_event', index_col=0).to_numpy()[:m, :m]
C_home = pd.read_excel(setting_file, sheet_name='C_home', index_col=0).to_numpy()[:n, :m]
C_depot_e = pd.read_excel(setting_file, sheet_name='C_depot_e', index_col=0).to_numpy().flatten()[:m]
C_depot_h = pd.read_excel(setting_file, sheet_name='C_depot_h', index_col=0).to_numpy().flatten()[:n]
C_dur = pd.read_excel(setting_file, sheet_name='C_dur', index_col=0).to_numpy().flatten()[:m]


def build_active_x_dataframes(
    active_x,
    C_event,
    C_home,
    C_depot_e,
    C_depot_h
):
    """
    Args:
      active_x : list of (i,j,d,w) tuples
      C_event   : 2D ndarray shape (m, m)
      C_home    : 2D ndarray shape (n, m)
      C_depot_e : 1D ndarray length m (cost event→depot)
      C_depot_h : 1D ndarray length n (cost home→depot)
    Returns:
      dict mapping day d -> DataFrame with columns
        ["nurse", "origin", "destination", "cost"]
    """
    # collect unique days
    days = sorted({d for (_, _, d, _) in active_x})
    dfs = {}

    for d in days:
        rows = []
        for (i, j, day_idx, w) in active_x:
            origin  = i 
            destination = j

            if day_idx != d:
                continue

            # figure out the cost by origin/destination type
            if i < m and j < m:
                # event → event
                cost = C_event[i, j]
            elif i == m and j < m:
                # home (w) → event j
                cost = C_home[w, j]
                origin = "home"
            elif i < m and j == m:
                # event i → home
                cost = C_home[w, i]
                destination = "home"
            elif i == m and j == m+1:
                # home (w) → depot
                cost = C_depot_h[w]
                origin = "home"
                destination = "depot"
            elif (i == m+1 and j == m):
                # home (w) → depot
                cost = C_depot_h[w]
                origin = "depot"
                destination = "home"
            elif i < m and j == m+1:
                # event i → depot
                cost = C_depot_e[i]
                destination = "depot"
            elif i == m+1 and j < m:
                # depot → event j
                cost = C_depot_e[j]
                origin = "depot"
                
            else:
                # any other combination (e.g. depot→something) – adjust as needed
                cost = None

            rows.append({
                "nurse":       w,
                "origin":      origin,
                "destination": destination,
                "cost":        cost,
            })

        dfs[d] = pd.DataFrame(rows, columns=["nurse", "origin", "destination", "cost"])

    return dfs

def minutes_to_time_str(minutes, start_hour=8, start_minute=0):
    """
    Convert a number of minutes after a base time into "HH:MM".
    If minutes is None, returns None.
    """
    if minutes is None:
        return None
    # integer minutes
    m = int(minutes)
    # compute total minutes from midnight
    total = start_hour * 60 + start_minute + m
    hour = total // 60
    minute = total % 60
    return f"{hour:02d}:{minute:02d}"

def build_active_event_dfs(active_alpha, active_beta, active_t):
    """
    From
      active_alpha: List of (i, d, w1) tuples
      active_beta : List of (i, d, w2) tuples
      active_t    : Dict of {(i, d) → start_time}
    Returns
      dict of DataFrame per day:
        columns = ["event", "pick-up", "drop-off", "Time"]
    """
    # find all days present in either alpha or beta
    days = sorted({
        d for (_, d, _) in active_alpha
    } | {
        d for (_, d, _) in active_beta
    })

    dfs = {}
    for d in days:
        # filter to today’s transfers
        alpha_today = [(i, w) for (i, d2, w) in active_alpha if d2 == d]
        beta_today  = [(i, w) for (i, d2, w) in active_beta  if d2 == d]

        pick_up  = {i: w for (i, w) in alpha_today}
        drop_off = {i: w for (i, w) in beta_today}

        rows = []
        for i in sorted(set(pick_up) | set(drop_off)):
            raw_minutes = active_t.get((i, d), None)
            time_str    = minutes_to_time_str(raw_minutes)
            rows.append({
                "event":    i,
                "pick-up":  pick_up .get(i, None),
                "drop-off": drop_off.get(i, None),
                "Start":     time_str,
                "End":       minutes_to_time_str(raw_minutes + C_dur[i])  if raw_minutes is not None and i < len(C_dur) else None,
                "Duration": C_dur[i]/60 if i < len(C_dur) else None
            })

        df = pd.DataFrame(
            rows,
            columns=["event", "pick-up", "drop-off", "Start", "End", "Duration"]
        )
        # sort by event if you like:
        df = df.sort_values("event").reset_index(drop=True)

        dfs[d] = df

    return dfs

import pandas as pd

def build_nurse_hours_df(active_x, C_dur, nr, nl, work_time_by_nurse):
    """
    Build a DataFrame indexed by nurse, with one column per event,
    filled with C_dur[event] if that nurse visits the event, else 0.
    Finally adds a 'Sum' column (row sum of event durations)
    and an 'Hours' column from work_time_by_nurse.

    Args:
      active_x: List of (i, j, d, w) tuples from your heuristic
      C_dur:    1D array-like of length m, giving each event’s service time
      nr:       number of RNs
      nl:       number of LVNs (so total nurses = nr + nl)
      work_time_by_nurse: dict { w: total_hours_worked }

    Returns:
      pd.DataFrame with
        - index = nurse (0 … nr+nl-1)
        - columns = 0,1,…,m-1, 'Sum', 'Hours'
    """
    total_nurses = nr + nl
    # 1) collect all events any nurse visits
    events = sorted({ j for (_, j, _, w) in active_x if 0 <= j < len(C_dur) })

    # 2) map each nurse → set of visited events
    nurse_events = { w: set() for w in range(total_nurses) }
    for (i, j, d, w) in active_x:
        if 0 <= j < len(C_dur):
            nurse_events[w].add(j)

    # 3) build rows
    data = []
    for w in range(total_nurses):
        row = { event: (C_dur[event] if event in nurse_events[w] else 0)
                for event in events }
        row["Sum"]   = sum(row[event] for event in events)/60
        row["Hours"] = work_time_by_nurse.get(w, 0.0)
        data.append(row)

    # 4) assemble DataFrame
    df = pd.DataFrame(data, index=range(total_nurses))
    df.index.name = "nurse"
    # ensure columns in order: events, Sum, Hours
    df = df[ events + ["Sum", "Hours"] ]
    return df

def count_events_by_nurse(active_x, nr, nl):
    """
    Count how many events each nurse is assigned to.
    Returns a dict { nurse_id: count } for each nurse.
    """
    total_nurses = nr + nl
    event_count = { w: 0 for w in range(total_nurses) }
    
    for (_, j, _, w) in active_x:
        if 0 <= j < len(C_dur):  # only count valid events
            event_count[w] += 1
            
    return event_count

# --- usage example in main.py ---

# assume you’ve already built summary["active_x"] and loaded your cost matrices:
# C_event, C_home, C_depot_e, C_depot_h

dfs_by_day = build_active_x_dataframes(
    summary["active_x"],
    C_event,
    C_home,
    C_depot_e,
    C_depot_h
)

event_dfs_by_day = build_active_event_dfs(
    summary["active_alpha"],
    summary["active_beta"],
    summary["active_t"]
)

nurse_hours_df = build_nurse_hours_df(
        summary["active_x"],
        C_dur,
        nr,
        nl,
        summary["work_time_by_nurse"]
    )
# now dfs_by_day[d] is a DataFrame for each day d.
# e.g. to peek at day 0:
# print(dfs_by_day[0].sort_values(by="nurse").reset_index(drop=True))

# save count_events_by_nurse as a csv
event_count = count_events_by_nurse(summary["active_x"], nr, nl)
event_count_df = pd.DataFrame(list(event_count.items()), columns=["nurse", "event_count"])
# event_count_df.to_csv("event_count_by_nurse.csv", index=False)

# out_path = "/Users/jinghongmiao/Code/mvt-code/result-250812/active_x_by_day.xlsx"
# with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
#     for d, df in dfs_by_day.items():
#         # 1) sort by nurse (and, if you like, by origin too)
#         sorted_df = df.sort_values(by=["nurse"]).reset_index(drop=True)

#         # 2) write to its own sheet
#         sheet_name = f"Nurse_Day_{d}"
#         sorted_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
#     for d, df in event_dfs_by_day.items():
#         sheet = f"Event_Day_{d}"
#         df.to_excel(writer, sheet_name=sheet, index=False)
    
#     nurse_sheet = "Nurse_Hours"
#     nurse_hours_df.to_excel(writer, sheet_name=nurse_sheet, index=True)

# print(f"All days written to {out_path}")