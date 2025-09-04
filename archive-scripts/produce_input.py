#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:58:32 2024

@author: jingjin
"""

import numpy as np
import random


time_limit = 600
seed_number = 120

random.seed(seed_number)
np.random.seed(seed_number)

# Coefficients
nr = 15 # number of RNs 
nl = 35 # number of LVNs
n = nr + nl # total number of nurses
m = 75 # number of events
block = 5 # number of days

# helper functions
def get_time(minute):
    hour = int(minute // 60 + 9)
    minute = int(minute % 60)
    if minute < 10:
        minute = "0" + str(minute)
    return f"{hour}:{minute}"

C_event = np.random.randint(10, 26, size=(m, m))
# Set the diagonal elements to 0
np.fill_diagonal(C_event, 0)

# travel cost: event1-20 x (home RN1-5, LVN1-10)
C_home = np.random.randint(10, 21, size=(m, nr+nl))

# travel cost: (event1-20, RN1-5) x depot
C_depot = np.random.randint(10, 21, size=(m+nr))

# event cost: event1-20
# Define the array of possible values
dur_values = np.array([30, 45, 60, 90, 120, 150])

# Define the probability distribution
dur_probabilities = np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.1])  # Adjust probabilities as needed

# Generate the array
C_dur = np.random.choice(dur_values, size=m, p = dur_probabilities)

# feasible time window (earliest, latest) by minutes past 9:00am 
# event1-20 x 5 days

# Function to generate a random time window
def generate_time_window():
    start = np.random.randint(1, 3) * 30  # Start time: 0 to 300 by 30s
    end = np.random.randint(start // 30 + 8, 12) * 30  # End time: start + 30 to 330 by 30s
    return [start, end]

# Initialize the matrix
time_window = np.zeros((m, block, 2), dtype=int)

# Populate the matrix with feasible time windows
for i in range(m):
    for j in range(block):
        # generate a full random time window table
        # time_window[i, j] = generate_time_window()

        # generate a partial time window table
        time_window[i, j] = random.choice([[0,0], generate_time_window()])

# minimum number of nurses required for each event
# event1-20 x (RN, LVN)
min_values = np.array([1, 2, 3, 4, 5, 6, 7])
min_probabilities_RN = np.array([0.4, 0.4, 0.2, 0, 0, 0, 0])
min_probabilities_LVN = np.array([0.1, 0.3, 0.3, 0.1, 0.1, 0.05, 0.05])
min_nurse = np.zeros((m, 2), dtype=int)
for i in range(m):
    min_nurse[i][0] = np.random.choice(min_values, p = min_probabilities_RN)
    min_nurse[i][1] = np.random.choice(min_values, p = min_probabilities_LVN)


import pandas as pd
# Save all input parameters to an Excel file
with pd.ExcelWriter('input_parameters_{}_{}_{}_seed{}.xlsx'.format(nr,nl,m,seed_number)) as writer:
    # Save coefficients and settings
    pd.DataFrame({'Parameter': ['time_limit', 'seed_number', 'nr', 'nl', 'm', 'block'],
                  'Value': [time_limit, seed_number, nr, nl, m, block]}).to_excel(writer, sheet_name='Settings', index=False)

    # Save C_event matrix
    pd.DataFrame(C_event).to_excel(writer, sheet_name='C_event', index=False)

    # Save C_home matrix
    pd.DataFrame(C_home).to_excel(writer, sheet_name='C_home', index=False)

    # Save C_depot matrix
    pd.DataFrame(C_depot, columns=['Depot_Cost']).to_excel(writer, sheet_name='C_depot', index=False)

    # Save C_dur array
    pd.DataFrame(C_dur, columns=['Duration']).to_excel(writer, sheet_name='C_dur', index=False)

    # Save time_window matrix
    time_window_flat = time_window.reshape(m * block, 2)
    pd.DataFrame(time_window_flat, columns=['Start', 'End']).to_excel(writer, sheet_name='Time_Window', index=False)

    # Save minimum nurses required
    pd.DataFrame(min_nurse, columns=['RN', 'LVN']).to_excel(writer, sheet_name='Min_Nurse', index=False)
