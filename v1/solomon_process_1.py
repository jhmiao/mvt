import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import os

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
    plt.scatter(df[x_col], df[y_col], c=df['Demand'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Demand')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_c_dur(df):
    subset = df[(df['Event'] <= 50) & (df['Event'] > 0)].copy()
    subset.drop(columns=['X', 'Y', 'Demand', 'Start', 'End'], inplace=True)
    subset['Duration'] = subset['Duration'] * 2
    return subset.set_index('Event').sort_index()

def generate_time_window(df):
    """
    Clusters the (Start, End) pairs into 5 groups.
    For each group i, sets Start_i = 30 and End_i = 270, all others to 0.
    """
    tw = df[(df['Event'] <= 50) & (df['Event'] > 0)].copy()
    tw.set_index('Event', inplace=True)

    # Run k-means clustering on (Start, End)
    kmeans = KMeans(n_clusters=5, random_state=42)
    tw['Cluster'] = kmeans.fit_predict(tw[['Start', 'End']])
    
    # Reindex clusters from 1 to 5 instead of 0 to 4
    tw['Cluster'] += 1

    # Initialize Start_i and End_i columns
    for i in range(1, 6):
        tw[f'Start_{i}'] = (tw['Cluster'] == i).astype(int) * 30
        tw[f'End_{i}'] = (tw['Cluster'] == i).astype(int) * 270

    return tw[[f'Start_{i}' for i in range(1, 6)] + [f'End_{i}' for i in range(1, 6)]]

def generate_min_nurse(df):
    mn = df[(df['Event'] <= 50) & (df['Event'] > 0)].copy()
    mn.drop(columns=['X', 'Y', 'Start', 'End', 'Duration'], inplace=True)
    mn['RN'] = (mn['Demand'] / 10).astype(int)
    np.random.seed(42)
    mn['LVN'] = mn['RN'] + np.random.randint(0, mn['RN'] + 1, size=len(mn))
    return mn.drop(columns=['Demand']).set_index('Event').sort_index()

def main(file_name):
    file_path = f'Solomon_25/{file_name}.txt'
    output_dir = f'Solomon_25'
    df = read_solomon_file(file_path)
    df = assign_event_types(df)

    # Plot locations
    plot_coordinates(df, f'{output_dir}/{file_name}_locations.png', 'Customer Locations', 'X', 'Y')

    # Plot time windows
    plot_coordinates(df, f'{output_dir}/{file_name}_time_windows.png', 'Customer Time Windows', 'Start', 'End')

    # Distance matrices
    C_event = create_distance_matrix(df, lambda r1, r2: 0 < r1['Event'] <= 50 and 0 < r2['Event'] <= 50)
    C_home = create_distance_matrix(df, lambda r1, r2: 0 < r1['Event'] <= 50 and r2['Event'] > 50)
    C_depot = pd.DataFrame(index=df.index, columns=[0])
    for i in df.index:
        if i != 0:
            C_depot.loc[i, 0] = euclidean_distance(df.loc[i], df.loc[0])
    C_depot.dropna(how='all', inplace=True)
    C_depot = C_depot.astype(int).T

    # Combine matrices
    C_travel = pd.concat([C_event.join(C_home), C_depot], axis=0).T

    # Generate C_dur, Time_Window, Min_Nurse
    C_dur = generate_c_dur(df)
    Time_Window = generate_time_window(df)
    Min_Nurse = generate_min_nurse(df)

    # Settings
    Setting = pd.DataFrame({'Parameter': ['nr', 'nl', 'm', 'day'], 'Value': [20, 30, 50, 5]})

    # Save to Excel
    with pd.ExcelWriter(f'{file_name}_data.xlsx') as writer:
        Setting.to_excel(writer, sheet_name='Settings', index=False)
        C_travel.to_excel(writer, sheet_name='C_travel')
        C_dur.to_excel(writer, sheet_name='C_dur')
        Time_Window.to_excel(writer, sheet_name='Time_Window')
        Min_Nurse.to_excel(writer, sheet_name='Min_Nurse')

if __name__ == "__main__":
    main("r211")
