import pandas as pd
import numpy as np
# from dataclasses import dataclass
from src.structures.problem_data import ProblemData

def load_problem_data(file_path, sample_k: int | None = None, sample_seed: int | None = None) -> ProblemData:
    """
    Load problem data from a xlsx file and return multiple numpy arrays.
    
    Parameters:
    file_path (str): The path to the xlsx file containing the problem data.
    
    Optional sampling:
    sample_k (int | None): If provided, randomly sample k events from the instance.
    sample_seed (int | None): Seed for reproducible sampling.

    Returns:
    C_event: numpy array of event travel costs (m x m).
    C_home: numpy array of home travel costs (n x m).
    C_depot_e: numpy array of depot travel costs (m,).
    C_depot_h: numpy array of depot travel costs (n,).
    C_dur: numpy array of event durations (m,).
    time_window: numpy array of time windows (m, day, 2).
    min_nurse: numpy array of minimum nurses required (m, 2). First column for RN, second for LVN.
    nr (int): Number of RNs.
    nl (int): Number of LVNs.
    n (int): Total number of nurses (nr + nl).
    m (int): Number of events.
    day (int): Number of days.
    If an error occurs, returns an empty DataFrame.
    """
    try:

        # Read the settings (nr, nl, m, block)
        settings_df = pd.read_excel(file_path, sheet_name='Settings')
        # Convert to a dictionary
        settings = dict(zip(settings_df['Parameter'], settings_df['Value']))

        nr = int(settings['nr'])
        nl = int(settings['nl'])
        m = int(settings['m'])
        n = nr + nl
        
        day = int(settings['day'])

        # Read C_travel
        C_event = pd.read_excel(file_path, sheet_name='C_event', index_col=0).to_numpy()[:m, :m]
        C_home = pd.read_excel(file_path, sheet_name='C_home', index_col=0).to_numpy()[:n, :m]
        C_depot_e = pd.read_excel(file_path, sheet_name='C_depot_e', index_col=0).to_numpy().flatten()[:m]
        C_depot_h = pd.read_excel(file_path, sheet_name='C_depot_h', index_col=0).to_numpy().flatten()[:n]


        # Read the C_dur array
        C_dur = pd.read_excel(file_path, sheet_name='C_dur', index_col=0).to_numpy().flatten()[:m]

        # Read the time_window matrix
        time_window = pd.read_excel(file_path, sheet_name='Time_Windows', index_col=0).to_numpy()[:m, :].reshape((m, day, 2))

        # Read the minimum nurses required matrix
        min_nurse = pd.read_excel(file_path, sheet_name='Min_Nurses', index_col=0).to_numpy()[:m, :]

        original_event_ids = None
        if sample_k is not None:
            if sample_k <= 0:
                raise ValueError(f"sample_k must be positive, got {sample_k}")
            if sample_k > m:
                raise ValueError(f"sample_k ({sample_k}) exceeds total events ({m})")
            rng = np.random.default_rng(sample_seed)
            sampled = rng.choice(m, size=sample_k, replace=False)
            sampled = np.sort(sampled)
            original_event_ids = sampled

            C_event = C_event[np.ix_(sampled, sampled)]
            C_home = C_home[:, sampled]
            C_depot_e = C_depot_e[sampled]
            C_dur = C_dur[sampled]
            time_window = time_window[sampled, :, :]
            min_nurse = min_nurse[sampled, :]
            m = sample_k


        for i in range(m):
            for d in range(day):
                # for potentially scheduled events
                if time_window[i, d, 1] > 0:
                    # Ensure that the latest start time allows for event duration before 1050
                    time_window[i, d, 1] = max(1050 - C_dur[i], time_window[i, d, 0])

        return ProblemData(
            C_event,
            C_home,
            C_depot_e,
            C_depot_h,
            C_dur,
            time_window,
            min_nurse,
            nr,
            nl,
            n,
            m,
            day,
            original_event_ids=original_event_ids,
        )

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise  # Re-raise the exception instead of returning a DataFrame

