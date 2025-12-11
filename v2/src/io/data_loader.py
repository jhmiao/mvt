import pandas as pd
import numpy as np
# from dataclasses import dataclass
from src.models.problem_data import ProblemData

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
    min_nurse: numpy array of minimum nurses required (m, 2). First column for RN, second for LVN.
    nr (int): Number of RNs.
    nl (int): Number of LVNs.
    n (int): Total number of nurses (nr + nl).
    m (int): Number of events.
    day (int): Number of days.
    If an error occurs, returns an empty DataFrame.
    """
    try:
        # file_path = 'problem_data_1.xlsx'

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

        nurse_type = pd.read_excel(file_path, sheet_name='Nurse_Type', index_col=0).to_numpy().flatten()[:n]

        # print(f"shape of C_travel: {C_event.shape}")
        # print(f"shape of C_home: {C_home.shape}")
        # print(f"shape of C_depot: {C_depot.shape}")

        # Read the C_dur array
        C_dur = pd.read_excel(file_path, sheet_name='C_dur', index_col=0).to_numpy().flatten()[:m]

        # Read the time_window matrix
        time_window = pd.read_excel(file_path, sheet_name='Time_Window', index_col=0).to_numpy()[:m, :].reshape((m, day, 2))

        # Read the minimum nurses required matrix
        min_nurse = pd.read_excel(file_path, sheet_name='Min_Nurse', index_col=0).to_numpy()[:m, :]

        # print(f"shape of C_dur: {C_dur.shape}")
        # print(f"shape of time_window: {time_window.shape}")
        # print(f"shape of min_nurse: {min_nurse.shape}")
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
            # assuming latest job ends at 7pm
            for i in range(m):
                for d in range(day):
                    # time_window[i, d, 1] = min(1140 - C_dur[i], time_window[i, d, 1])
                    time_window[i, d, 1] = max(1050 - C_dur[i], time_window[i, d, 0])

                    
        
        elif type == 'discrete':
            # replace non-zero time windows with 1
            time_window[time_window != 0] = 1
        
        # print(f"RN average working hours: {avg_RN_hours}")
        # print(f"LVN average working hours: {avg_LVN_hours} \n\n")

        return ProblemData(C_event, C_home, C_depot_e, C_depot_h, C_dur, time_window, min_nurse, nurse_type, nr, nl, n, m, day)

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise  # Re-raise the exception instead of returning a DataFrame




def load_problem_data_sample(file_path, nr, nl, m, n_tw, seed = 42) -> ProblemData:
    """
    Load problem data from a xlsx file and return multiple numpy arrays.
    Use provided nr, nl, m, and n_tw to adjust data loading.
    
    Parameters:
    file_path (str): The path to the xlsx file containing the problem data.
    nr (int): Number of RNs to sample.
    nl (int): Number of LVNs to sample.
    m (int): Number of events to sample.
    n_tw (int): Number of time windows to sample.
    
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
        # file_path = 'problem_data_1.xlsx'

        # Read the settings (nr, nl, m, block)
        settings_df = pd.read_excel(file_path, sheet_name='Settings')
        # Convert to a dictionary
        settings = dict(zip(settings_df['Parameter'], settings_df['Value']))

        
        nr_full = int(settings['nr'])
        nl_full = int(settings['nl'])
        m_full = int(settings['m'])
        n = nr + nl

        # raise error if requested nr, nl, or m exceeds full data
        if nr > nr_full or nl > nl_full or m > m_full:
            raise ValueError("Requested nr, nl, or m exceeds available data in the file.")
        
        day = int(settings['day'])


        # Read the C_dur array
        C_dur_full = pd.read_excel(file_path, sheet_name='C_dur', index_col=0).to_numpy().flatten()
        # get the index of the m smallest durations
        m_indices = np.argsort(C_dur_full)[:m]
        C_dur = C_dur_full[m_indices]


        # Read C_travel
        C_event = pd.read_excel(file_path, sheet_name='C_event', index_col=0).to_numpy()[np.ix_(m_indices, m_indices)]
        C_home_rn = pd.read_excel(file_path, sheet_name='C_home', index_col=0).to_numpy()[:nr, m_indices]
        C_home_lvn = pd.read_excel(file_path, sheet_name='C_home', index_col=0).to_numpy()[nr_full:nr_full+nl, m_indices]
        C_home = np.vstack([C_home_rn, C_home_lvn])

        C_depot_e = pd.read_excel(file_path, sheet_name='C_depot_e', index_col=0).to_numpy().flatten()[m_indices]
        C_depot_h_rn = pd.read_excel(file_path, sheet_name='C_depot_h', index_col=0).to_numpy().flatten()[:nr]
        C_depot_h_lvn = pd.read_excel(file_path, sheet_name='C_depot_h', index_col=0).to_numpy().flatten()[nr_full:nr_full+nl]
        C_depot_h = np.concatenate([C_depot_h_rn, C_depot_h_lvn])
        C_depot = np.concatenate([C_depot_e, C_depot_h_rn, C_depot_h_lvn])

        nurse_type_rn = pd.read_excel(file_path, sheet_name='Nurse_Type', index_col=0).to_numpy().flatten()[:nr]
        nurse_type_lvn = pd.read_excel(file_path, sheet_name='Nurse_Type', index_col=0).to_numpy().flatten()[nr_full:nr_full+nl]
        nurse_type = np.concatenate([nurse_type_rn, nurse_type_lvn])

        # print(f"shape of C_travel: {C_event.shape}")
        # print(f"shape of C_home: {C_home.shape}")
        # print(f"shape of C_depot: {C_depot.shape}")


        # Read the time_window matrix
        time_window = pd.read_excel(file_path, sheet_name='Time_Window', index_col=0).to_numpy()[m_indices, :].reshape((m, day, 2))

        if n_tw > 1:
            np.random.seed(seed)
        # for each event, randomly select (n_tw-1) [0,0] time window and replace it with [570, 810]
        for i in range(m):
            zero_days = [d for d in range(day) if np.all(time_window[i, d] == 0)]
            
            if zero_days:
                chosen_days = np.random.choice(zero_days, size=n_tw-1, replace=False)
                time_window[i, chosen_days] = [570, 810]

        # Read the minimum nurses required matrix
        min_nurse = pd.read_excel(file_path, sheet_name='Min_Nurse', index_col=0).to_numpy()[m_indices, :]

        # print(f"shape of C_dur: {C_dur.shape}")
        # print(f"shape of time_window: {time_window.shape}")
        # print(f"shape of min_nurse: {min_nurse.shape}")
        # print(time_window)

        # calculate the expected average working hours for each nurse
        # Calculate total expected working hours for RNs and LVNs
        total_RN_hours = np.sum(C_dur[:] * min_nurse[:, 0])
        total_LVN_hours = np.sum(C_dur[:] * min_nurse[:, 1])

        # Convert from minutes to hours if needed (assuming C_dur is in minutes)
        total_RN_hours /= 60
        total_LVN_hours /= 60

        # Compute average working hours
        avg_RN_hours = total_RN_hours / nr if nr > 0 else 0
        avg_LVN_hours = total_LVN_hours / nl if nl > 0 else 0


        if type == 'continuous':
            # Ensure time_window end time is feasible
            # assuming latest job ends at 7pm
            for i in range(m):
                for d in range(day):
                    time_window[i, d, 1] = min(1140 - C_dur[i], time_window[i, d, 1])

        
        elif type == 'discrete':
            # replace non-zero time windows with 1
            time_window[time_window != 0] = 1
        
        # print(f"RN average working hours: {avg_RN_hours}")
        # print(f"LVN average working hours: {avg_LVN_hours} \n\n")

        return ProblemData(C_event, C_home, C_depot_e, C_depot_h, C_dur, time_window, min_nurse, nurse_type, nr, nl, n, m, day)

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise  # Re-raise the exception instead of returning a DataFrame



    