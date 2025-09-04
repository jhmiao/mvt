import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
from typing import Dict, List, Tuple
from pathlib import Path



def read_solomon_file(file_path):
    """Reads Solomon instance file and returns parsed lines starting from the customer section."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.strip().startswith("CUST NO."):
            header_index = i
            break

    data_lines = lines[header_index + 2:]
    data = [
        [int(part) for part in line.split()]
        for line in data_lines if line.strip()
    ]

    columns = ['Index', 'X', 'Y', 'Demand', 'Start', 'End', 'Duration']
    df = pd.DataFrame(data, columns=columns).set_index('Index')
    return df

def assign_event_types(df):
    np.random.seed(42)
    df['Event'] = 0
    df.loc[1:, 'Event'] = np.random.choice(np.arange(1, 101), size=len(df) - 1, replace=False)
    return df

# Generate home, depot, and event DataFrames

def get_home(df):
    """Returns a DataFrame of home locations, reindexed by Event. Only keep rows where Event > 50. Only keep Index, X, Y"""
    home = df[df['Event'] > 50].copy().reset_index()
    home = home[['X', 'Y']]
    home.index.name = 'Home'
    return home

def get_depot(df):
    """Returns a DataFrame of depot locations, reindexed by Event. Only keep rows where Event == 0. Only keep Index, X, Y. Add another row which is the duplicate value of the first row. Index first row as 'Depot_am', second row as 'Depot_pm'."""
    depot = df[df['Event'] == 0].copy().reset_index()
    depot = depot[['X', 'Y']]
    # Duplicate the row if only one depot exists
    if len(depot) == 1:
        depot = pd.concat([depot, depot], ignore_index=True)
    depot.index.name = 'Depot'
    return depot

def get_event(df):
    """Returns a DataFrame of event locations, reindexed by Event. Only keep rows where 0 < Event <= 50. Only keep Index, X, Y"""
    event = df[(df['Event'] > 0) & (df['Event'] <= 50)].copy().reset_index()
    tw = event[['Start', 'End']].copy()
    event = event[['X', 'Y']]
    event.index.name = 'Event'

    # generate duration
    # read from real-nurse-dur.xlsx
    real_data = pd.read_excel('real-nurse-dur.xlsx', index_col=0)
    # randomly sample 50 rows from real_data without replacement
    sampled_data = real_data.sample(n=50, replace=False, random_state=42)
    # assign sampled data to C_dur and Min_Nurse
    event['Duration'] = sampled_data['Duration'].values

    # generate RN and LVN requirements
    event['RN'] = sampled_data['RN'].values
    event['LVN'] = sampled_data['LVN'].values

    # generate time_window using k-means clustering
    # Run k-means clustering on (Start, End)
    kmeans = KMeans(n_clusters=5, random_state=42)
    tw['Cluster'] = kmeans.fit_predict(tw[['Start', 'End']])
    
    # Reindex clusters from 1 to 5 instead of 0 to 4
    tw['Cluster'] += 1

    # Initialize Start_i and End_i columns
    for i in range(1, 6):
        tw[f'Start_{i}'] = (tw['Cluster'] == i).astype(int) * 30
        tw[f'End_{i}'] = (tw['Cluster'] == i).astype(int) * 270

    # Create the interleaved column list
    columns = [f'Start_{i}' for i in range(1, 6)]
    columns = [item for i in range(1, 6) for item in (f'Start_{i}', f'End_{i}')]

    # Add time_window to event DataFrame
    event = event.join(tw[columns])

    return event

# Create distance matrices

def euclidean_distance(row1, row2):
    return np.sqrt((row1['X'] - row2['X']) ** 2 + (row1['Y'] - row2['Y']) ** 2)

def create_distance_matrix(df, condition):
    matrix = pd.DataFrame(index=df.index, columns=df.index)
    for i in df.index:
        for j in df.index:
            if condition(df.loc[i], df.loc[j]):
                matrix.loc[i, j] = euclidean_distance(df.loc[i], df.loc[j])
            else:
                matrix.loc[i, j] = np.nan
    matrix.dropna(how='all', inplace=True)
    matrix.dropna(axis=1, how='all', inplace=True)
    return matrix.astype(int)

def plot_coordinates(df, output_path, title, x_col, y_col):
    plt.figure(figsize=(10, 6))
    # plt.scatter(df[x_col], df[y_col], c=df['Demand'], cmap='viridis', alpha=0.6)
    # plt.colorbar(label='Demand')

    # For rows with 0 < Event <= 50, plot coordinates as "Event"
    # For rows with Event > 50, plot coordinates as "Home"
    # For rows with Event == 0, plot coordinates as "Depot"
    plt.scatter(df.loc[df['Event'] <= 50, x_col], df.loc[df['Event'] <= 50, y_col], c='blue', label='Customer', alpha=0.6)
    plt.scatter(df.loc[df['Event'] > 50, x_col], df.loc[df['Event'] > 50, y_col], c='green', label='Home', alpha=0.6)
    plt.scatter(df.loc[df['Event'] == 0, x_col], df.loc[df['Event'] == 0, y_col], c='red', label='Depot', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.savefig(output_path, dpi=300)
    plt.close()





def main(file_name):

    base_dir = Path(__file__).resolve().parent.parent.parent
    file_path = base_dir / 'data' / 'raw' / 'Solomon_25' / f'{file_name}.txt'

    output_dir = base_dir / 'data' / 'clean' / f'{file_name}'

    df = read_solomon_file(file_path)
    df = assign_event_types(df)
    print(df.head())
    home = get_home(df)
    depot = get_depot(df)
    event = get_event(df)
    print(home.head())
    print(depot)
    print(event)

    # Distance matrices
    C_event = create_distance_matrix(df, lambda r1, r2: 0 < r1['Event'] <= 50 and 0 < r2['Event'] <= 50)
    C_home = create_distance_matrix(df, lambda r1, r2: 0 < r1['Event'] <= 50 and r2['Event'] > 50).T
    C_depot = pd.DataFrame(index=df.index, columns=[0])
    for i in df.index:
        if i != 0:
            C_depot.loc[i, 0] = euclidean_distance(df.loc[i], df.loc[0])
    C_depot.dropna(how='all', inplace=True)
    C_depot_h = C_depot.astype(int).loc[df['Event'] > 50]
    C_depot_e = C_depot.astype(int).loc[(df['Event'] <= 50) & (df['Event'] > 0)]

    # Combine matrices
    # C_travel = pd.concat([C_event.join(C_home), C_depot], axis=0).T

    # Generate C_dur, Time_Window, Min_Nurse
    C_dur, Min_Nurse = generate_c_dur_and_min_nurse(df)
    Time_Window = generate_time_window(df).sort_index()

    # Settings
    Setting = pd.DataFrame({'Parameter': ['nr', 'nl', 'm', 'day'], 'Value': [20, 30, 50, 5]})

    # # Save to Excel
    # with pd.ExcelWriter(f'{file_name}_data.xlsx') as writer:
    #     Setting.to_excel(writer, sheet_name='Settings', index=False)
    #     C_event.to_excel(writer, sheet_name='C_event')
    #     C_home.to_excel(writer, sheet_name='C_home')
    #     C_depot_e.to_excel(writer, sheet_name='C_depot_e')
    #     C_depot_h.to_excel(writer, sheet_name='C_depot_h')
    #     C_dur.to_excel(writer, sheet_name='C_dur')
    #     Time_Window.to_excel(writer, sheet_name='Time_Window')
    #     Min_Nurse.to_excel(writer, sheet_name='Min_Nurse')

# import json
# import numpy as np
# from typing import Dict, List, Tuple

# def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
#     """Compute haversine distance (in km) between two (lat, lon) pairs."""
#     import math
#     R = 6371  # Earth radius in kilometers
#     lat1, lon1 = map(math.radians, coord1)
#     lat2, lon2 = map(math.radians, coord2)
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
#     c = 2 * math.asin(math.sqrt(a))
#     return R * c


# def build_location_index(locations: Dict[str, Tuple[float, float]]) -> Dict[str, int]:
#     """
#     Assigns each location_id (including depot variants like 'depot_am'/'depot_pm') a unique index.
#     """
#     return {loc_id: idx for idx, loc_id in enumerate(locations.keys())}


# def build_distance_matrix(locations: Dict[str, Tuple[float, float]],
#                           location_id_to_index: Dict[str, int]) -> np.ndarray:
#     """
#     Builds the distance matrix between all locations.
#     """
#     n = len(location_id_to_index)
#     dist_matrix = np.zeros((n, n))

#     for id1, idx1 in location_id_to_index.items():
#         for id2, idx2 in location_id_to_index.items():
#             if idx1 == idx2:
#                 continue
#             coord1, coord2 = locations[id1], locations[id2]
#             dist_matrix[idx1, idx2] = haversine_distance(coord1, coord2)

#     return dist_matrix


# def load_data() -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
#     """
#     Loads locations, builds mapping and distance matrix.
#     Example supports depot variants like 'depot_am', 'depot_pm'.
#     """
#     # Example input (could be loaded from JSON/DB/etc.)
#     locations = {
#         "depot_am": (34.0522, -118.2437),   # Los Angeles
#         "depot_pm": (34.0522, -118.2437),   # Same physical coords, different role
#         "customer_1": (34.1400, -118.1250),
#         "customer_2": (34.1800, -118.3000),
#     }

#     location_id_to_index = build_location_index(locations)
#     index_to_location_id = {v: k for k, v in location_id_to_index.items()}

#     distance_matrix = build_distance_matrix(locations, location_id_to_index)

#     # Save for reuse
#     np.save("io/distance_matrix.npy", distance_matrix)
#     with open("io/location_id_to_index.json", "w") as f:
#         json.dump(location_id_to_index, f)

#     return distance_matrix, location_id_to_index, index_to_location_id


# if __name__ == "__main__":
#     dist_matrix, loc_to_idx, idx_to_loc = load_data()
#     print("Location to Index:", loc_to_idx)
#     print("Distance Matrix Shape:", dist_matrix.shape)

if __name__ == "__main__":
    main("c101")
