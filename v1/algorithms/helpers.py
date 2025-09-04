import numpy as np
import pandas as pd
from data_loader import ProblemData, load_problem_data

# helper functions
def get_time(minute):
    """
    Converts a given number of minutes into a time string formatted as HH:MM.
    Args:
        minute (int): The number of minutes to convert.
    Returns:
        str: The time in HH:MM format, where HH is the hour and MM is the minute.
    """
    hour = int(minute // 60 + 9)
    minute = int(minute % 60)
    if minute < 10:
        minute = "0" + str(minute)
    return f"{hour}:{minute}"

# helper functions
def get_timeslot(index1, index2):
    if index1 == 0 and index2 == 0:
        return "Not scheduled"
    elif index1 == 1 and index2 == 0:
        return "Morning"
    elif index1 == 0 and index2 == 1:
        return "Afternoon"
    else:
        return "Full day"
    
def extract_day_data(data, day):
    """
    Extracts the relevant data for a specific day from the problem data.
    
    Args:
        problem_data (ProblemData): The complete problem data.
        day (int): The day for which to extract the data.
        
    Returns:
        data_today (ProblemData): A new ProblemData object containing only the data for the specified day.
    """
    
    m = data.m
    n = data.n
    nr = data.nr
    nl = data.nl
    time_window = data.time_window
    C_event = data.C_event
    C_home = data.C_home
    C_depot = data.C_depot
    C_dur = data.C_dur
    min_nurse = data.min_nurse

    # filter data for day
    # get the index of events scheduled on that day: where time_window[:, d, 1] > 0
    events_today = np.where(time_window[:, day, 1] > 0)[0]
    # filter C_event, C_home, C_depot, C_dur, time_window, min_nurse by events_today
    C_event_today = C_event[np.ix_(events_today, events_today)]
    C_home_today = C_home[events_today]
    C_depot_today = np.concatenate([C_depot[events_today], C_depot[m:]])
    C_dur_today = C_dur[events_today]
    time_window_today = time_window[events_today, day, :].reshape((len(events_today), 1, 2))
    min_nurse_today = min_nurse[events_today, :]

    # create a new ProblemData object for day d
    data_today = ProblemData(
        C_event=C_event_today,
        C_home=C_home_today,
        C_depot=C_depot_today,
        C_dur=C_dur_today,
        time_window=time_window_today,
        min_nurse=min_nurse_today,
        nr=nr,
        nl=nl,
        n=n,
        m=len(events_today),
        day=1  # only one day
    )

    return data_today

def get_sorted_travel_time_cumsum (data: ProblemData):
    """
    Given a ProblemData object, extracts the C_home attribute and computes the
    sorted cumulative travel times for each nurse.
    Returns a n by m array where n is the number of nurses and m is the number of events.
    """

    C_home = data.C_home  # shape (n, m)
    n, m = C_home.shape
    sorted_travel_time_cumsum = np.zeros((n, m))
    for i in range(n):
        sorted_travel_time_cumsum[i, :] = np.cumsum(np.sort(C_home[i, :]))
    return pd.DataFrame(sorted_travel_time_cumsum)

# data_continuous = load_problem_data('c101_data.xlsx', type='continuous')
