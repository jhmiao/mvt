import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ProblemData:
    C_event: np.ndarray
    C_home: np.ndarray
    # C_depot_e: np.ndarray       # (m,)      event i -> depot
    # C_depot_h: np.ndarray       # (n,)      depot -> home(w)  (if used in your model)
    C_depot: np.ndarray      # (m+n,)    combined depot travel costs
    C_dur: np.ndarray
    time_window: np.ndarray
    min_nurse: np.ndarray
    nr: int
    nl: int
    n: int
    m: int
    day: int

def load_problem_data(file_path, type) -> ProblemData:
    """
    Load problem data from a xlsx file and return multiple numpy arrays.
    
    Parameters:
    file_path (str): The path to the xlsx file containing the problem data.
    
    Returns:
    C_event: numpy array of event travel costs (m x m).
    C_home: numpy array of home travel costs (n x m).
    C_depot_e: numpy array of depot travel costs (m,).
    C_depot_h: numpy array of depot travel costs (n,).
    C_dur: numpy array of event durations (m,).
    time_window: numpy array of time windows (m, day, 2).
    min_nurse: numpy array of minimum nurses required (m, day).
    nr (int): Number of RNs.
    nl (int): Number of LVNs.
    n (int): Total number of nurses (nr + nl).
    m (int): Number of events.
    day (int): Number of days.
    If an error occurs, returns an empty DataFrame.
    """
    try:
        # Ensure file_path ends with .xlsx
        if not file_path.endswith('.xlsx'):
            file_path = file_path + '.xlsx'

        # Read the settings (nr, nl, m, block)
        settings_df = pd.read_excel(file_path, sheet_name='Settings')
        # Convert to a dictionary
        settings = dict(zip(settings_df['Parameter'], settings_df['Value']))

        nr = int(settings['nr'])
        nl = int(settings['nl'])
        # nr = 20
        # nl = 30
        m = int(settings['m'])
        n = nr + nl
        
        # m = 50
        day = int(settings['day'])

        # Read C_travel
        C_event = pd.read_excel(file_path, sheet_name='C_event', index_col=0).to_numpy()[:m, :m]
        C_home = pd.read_excel(file_path, sheet_name='C_home', index_col=0).to_numpy()[:n, :m]
        C_depot_e = pd.read_excel(file_path, sheet_name='C_depot_e', index_col=0).to_numpy().flatten()[:m]
        C_depot_h = pd.read_excel(file_path, sheet_name='C_depot_h', index_col=0).to_numpy().flatten()[:n]
        C_depot = np.concatenate([C_depot_e, C_depot_h])

        print(f"shape of C_travel: {C_event.shape}")
        print(f"shape of C_home: {C_home.shape}")
        print(f"shape of C_depot: {C_depot.shape}")

        # Read the C_dur array
        C_dur = pd.read_excel(file_path, sheet_name='C_dur', index_col=0).to_numpy().flatten()[:m]

        # Read the time_window matrix
        time_window = pd.read_excel(file_path, sheet_name='Time_Window', index_col=0).to_numpy()[:m, :].reshape((m, day, 2))

        # Read the minimum nurses required matrix
        min_nurse = pd.read_excel(file_path, sheet_name='Min_Nurse', index_col=0).to_numpy()[:m, :]

        print(f"shape of C_dur: {C_dur.shape}")
        print(f"shape of time_window: {time_window.shape}")
        print(f"shape of min_nurse: {min_nurse.shape}")
        # print(time_window)

        # calculate the expected average working hours for each nurse
        # Calculate total expected working hours for RNs and LVNs
        total_RN_hours = np.sum(C_dur[:m] * min_nurse[:m, 0])
        total_LVN_hours = np.sum(C_dur[:m] * min_nurse[:m, 1])

        # Convert from minutes to hours if needed (assuming C_dur is in minutes)
        total_RN_hours /= 60
        total_LVN_hours /= 60

        # Compute average working hours
        avg_RN_hours = total_RN_hours / nr if nr > 0 else 0
        avg_LVN_hours = total_LVN_hours / nl if nl > 0 else 0


        if type == 'continuous':
            # Ensure time_window end time is feasible
            # for all (m, day, 1) values, replace with min(480-C_dur[m], time_window[m, day, 1])
            for i in range(m):
                for d in range(day):
                    time_window[i, d, 1] = min(600 - C_dur[i], time_window[i, d, 1])
                    
        
        elif type == 'discrete':
            # replace non-zero time windows with 1
            time_window[time_window != 0] = 1
        
        print(f"RN average working hours: {avg_RN_hours}")
        print(f"LVN average working hours: {avg_LVN_hours} \n\n")

        return ProblemData(C_event, C_home, C_depot, C_dur, time_window, min_nurse, nr, nl, n, m, day)

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise  # Re-raise the exception instead of returning a DataFrame



    