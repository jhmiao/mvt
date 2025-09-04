import pandas as pd
import numpy as np
import json

# read xlsx
file_path = '/Users/jinghongmiao/Code/mvt-code/v1/c101_data.xlsx'
# Read the settings (nr, nl, m, block)
settings_df = pd.read_excel(file_path, sheet_name='Settings')
# Convert to a dictionary
settings = dict(zip(settings_df['Parameter'], settings_df['Value']))

print(settings)
nr = int(settings['nr'])
nl = int(settings['nl'])
# # nr = 20
# # nl = 30
m = int(settings['m'])
n = nr + nl

C_dur = pd.read_excel(file_path, sheet_name='C_dur', index_col=0)
C_dur = C_dur.reset_index(drop=True)
C_dur.index = range(len(C_dur))
# print(C_dur.head())

time_window = pd.read_excel(file_path, sheet_name='Time_Window', index_col=0)
time_window = time_window.reset_index(drop=True)
time_window.index = range(len(time_window))
print(time_window.head())

min_nurse = pd.read_excel(file_path, sheet_name='Min_Nurse', index_col=0)
min_nurse = min_nurse.reset_index(drop=True)
min_nurse.index = range(len(min_nurse))
# print(min_nurse.head())

# # # Read C_travel

event_list = []
for idx in range(len(C_dur)):
    # Duration
    duration = int(C_dur.iloc[idx, 0])
    # Nurses required
    req = {}
    if 'RN' in min_nurse.columns:
        req['RN'] = int(min_nurse.loc[idx, 'RN'])
    if 'LVN' in min_nurse.columns:
        req['LVN'] = int(min_nurse.loc[idx, 'LVN'])
    # Time window: assume columns are like 'Start_1', 'End_1', ...
    time_window_dict = {}
    for col in time_window.columns:
        if 'Start' in col:
            day = int(col.split('_')[1])
            start_val = int(time_window.loc[idx, f'Start_{day}'])
            end_val = int(time_window.loc[idx, f'End_{day}'])
            start_min = start_val + 9 * 60 if start_val != 0 else 0
            end_min = end_val + 9 * 60 if end_val != 0 else 0
            time_window_dict[day] = (start_min, end_min)
    event = {
        'id': idx,
        'loc': None,  # or None if not available
        'req': req,
        'duration': duration,
        'time_window': time_window_dict
    }
    event_list.append(event)

with open('c101-event.json', 'w') as f:
    json.dump(event_list, f, indent=2)
# C_event = pd.read_excel(file_path, sheet_name='C_event', index_col=0)
# C_event = C_event.reset_index(drop=True)
# C_event.index = range(len(C_event))
# C_event.columns = range(len(C_event.columns))
# df_long = C_event.reset_index().melt(id_vars="index", var_name="to", value_name="time")
# # Rename columns
# df_long.rename(columns={"index": "from"}, inplace=True)

# # Add "event" prefix
# df_long["from"] = "event" + df_long["from"].astype(str)
# df_long["to"]   = "event" + df_long["to"].astype(str)

# C_home = pd.read_excel(file_path, sheet_name='C_home', index_col=0)
# C_home = C_home.reset_index(drop=True)
# C_home.index = range(len(C_home))
# C_home.columns = range(len(C_home.columns))
# df_home = C_home.reset_index().melt(id_vars="index", var_name="to", value_name="time")
# # Rename columns
# df_home.rename(columns={"index": "from"}, inplace=True)
# df_home["from"] = "home" + df_home["from"].astype(str)
# df_home["to"]   = "event" + df_home["to"].astype(str)

# # print(df_home.head(20))
# print(df_home.shape)

# # append df_home to df_long
# df_long = pd.concat([df_long, df_home], ignore_index=True)

# # add depot travel
# C_depot_e = pd.read_excel(file_path, sheet_name='C_depot_e', index_col=0)
# C_depot_e = C_depot_e.reset_index(drop=True)
# C_depot_e.index = range(len(C_depot_e))
# C_depot_e.columns = ["depot0"]
# C_depot_e["depot1"] = C_depot_e["depot0"]

# df_depot_e1 = C_depot_e.reset_index().melt(id_vars="index", var_name="from", value_name="time")
# # Rename columns
# df_depot_e1.rename(columns={"index": "to"}, inplace=True)
# df_depot_e1["to"] = "event" + df_depot_e1["to"].astype(str)
# df_depot_e1["from"]   = df_depot_e1["from"].astype(str)

# mask = df_depot_e1['from'] == 'depot1'
# df_depot_e1.loc[mask, ['from', 'to']] = df_depot_e1.loc[mask, ['to', 'from']].values

# # df_depot_e2 = C_depot_e.reset_index().melt(id_vars="index", var_name="to", value_name="time")
# # df_depot_e2.rename(columns={"index": "from"}, inplace=True)
# # df_depot_e2["to"] = df_depot_e2["to"].astype(str)
# # df_depot_e2["from"]   = "event" + df_depot_e2["from"].astype(str)

# print(df_depot_e1)

# df_long = pd.concat([df_long, df_depot_e1], ignore_index=True)

# C_depot_h = pd.read_excel(file_path, sheet_name='C_depot_h', index_col=0)
# C_depot_h = C_depot_h.reset_index(drop=True)
# C_depot_h.index = range(len(C_depot_h))
# C_depot_h.columns = ["depot0"]
# C_depot_h["depot1"] = C_depot_h["depot0"]

# df_depot_h = C_depot_h.reset_index().melt(id_vars="index", var_name="from", value_name="time")
# # Rename columns
# df_depot_h.rename(columns={"index": "to"}, inplace=True)
# df_depot_h["to"] = "home" + df_depot_h["to"].astype(str)
# df_depot_h["from"]   = df_depot_h["from"].astype(str)

# mask = df_depot_h['from'] == 'depot0'
# df_depot_h.loc[mask, ['from', 'to']] = df_depot_h.loc[mask, ['to', 'from']].values

# df_long = pd.concat([df_long, df_depot_h], ignore_index=True)
# print(df_long.tail(20))

# # save to csv
# df_long.to_csv('/Users/jinghongmiao/Code/mvt-code/v1/rc101_travel.csv', index=False)




