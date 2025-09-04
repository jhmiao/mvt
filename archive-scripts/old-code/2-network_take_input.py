#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# helper functions
def get_time(minute):
    hour = int(minute // 60 + 9)
    minute = int(minute % 60)
    if minute < 10:
        minute = "0" + str(minute)
    return f"{hour}:{minute}"

# Read the Excel file
file_path = 'input_parameters_15_35_75_seed120.xlsx'

# Read the settings (time_limit, seed_number, nr, nl, m, block)
settings_df = pd.read_excel(file_path, sheet_name='Settings')
settings = settings_df.set_index('Parameter')['Value'].to_dict()

time_limit = int(settings['time_limit'])
seed_number = int(settings['seed_number'])
nr = int(settings['nr'])
nl = int(settings['nl'])
n = nr + nl
m = int(settings['m'])
block = int(settings['block'])

# Read the C_event matrix
C_event_df = pd.read_excel(file_path, sheet_name='C_event')
C_event = C_event_df.values

# Read the C_home matrix
C_home_df = pd.read_excel(file_path, sheet_name='C_home')
C_home = C_home_df.values

# Read the C_depot array
C_depot_df = pd.read_excel(file_path, sheet_name='C_depot')
C_depot = C_depot_df['Depot_Cost'].values

# Read the C_dur array
C_dur_df = pd.read_excel(file_path, sheet_name='C_dur')
C_dur = C_dur_df['Duration'].values

# Read the time_window matrix
time_window_df = pd.read_excel(file_path, sheet_name='Time_Window')
time_window_flat = time_window_df.values
time_window = time_window_flat.reshape((m, block, 2))

# Read the minimum nurses required matrix
min_nurse_df = pd.read_excel(file_path, sheet_name='Min_Nurse')
min_nurse = min_nurse_df.values


def gurobi_solve():
    iterations = []
    objective_values = []

    # Define the callback function
    def my_callback(model, where):
        if where == GRB.Callback.MIP:
            obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            node_count = model.cbGet(GRB.Callback.MIP_NODCNT)
            if obj < GRB.INFINITY:
                print(f"Node Count: {node_count}, Objective: {obj}")
                iterations.append(node_count)
                objective_values.append(obj)
        elif where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                obj = model.cbGetNodeRel(GRB.Callback.MIPNODE_OBJBST)
                node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                if obj < GRB.INFINITY:
                    print(f"Node Count: {node_count}, Objective: {obj}")
                    iterations.append(node_count)

    model = gp.Model("team_scheduling")
    M = 390 # A large constant

    # Decision variables

    # x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
    x = [[[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}{d}{w}") for w in range(n)] for d in range(block)] for j in range(m)] for i in range(m)]

    # x_0jdw = 1 if nurse w goes from home to event j on day d, 0 otherwise
    xhome_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_0{j}{d}{w}") for w in range(n)] for d in range(block)] for j in range(m)]

    # x_i0dw = 1 if nurse w goes from event i to home on day d, 0 otherwise
    xevent_home = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}0{d}{w}") for w in range(n)] for d in range(block)] for i in range(m)]

    # x_0*dw = 1 if nurse w goes from home to depot on day d, 0 otherwise
    xhome_depot = [[model.addVar(vtype=GRB.BINARY, name=f"x_0*{d}{w}") for w in range(n)] for d in range(block)]

    # x_*0dw = 1 if nurse w goes from depot to home on day d, 0 otherwise
    xdepot_home = [[model.addVar(vtype=GRB.BINARY, name=f"x_*0{d}{w}") for w in range(n)] for d in range(block)]

    # x_*jdw = 1 if nurse w goes from depot to event j on day d, 0 otherwise
    xdepot_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_*{j}{d}{w}") for w in range(n)] for d in range(block)] for j in range(m)]

    # x_i*dw = 1 if nurse w goes from event i to depot on day d, 0 otherwise
    xevent_depot = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}*{d}{w}") for w in range(n)] for d in range(block)] for i in range(m)]

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = [[model.addVar(vtype=GRB.BINARY, name=f"s_{i}{d}") for d in range(block)] for i in range(m)] 

    # t_id time when event i starts on day d
    t = [[model.addVar(0, 390, vtype=GRB.INTEGER, name=f"t_{i}{d}") for d in range(block)] for i in range(m)]

    # alpha_idw = 1 if nurse w (RN) is the pick-up leader for event i on day d, 0 otherwise
    alpha = [[[model.addVar(vtype=GRB.BINARY, name=f"alpha_{i}{d}{w}") for w in range(n)] for d in range(block)] for i in range(m)]

    # # beta_idw = 1 if nurse w (RN) is the drop-off leader for event i on day d, 0 otherwise
    beta = [[[model.addVar(vtype=GRB.BINARY, name=f"beta_{i}{d}{w}") for w in range(n)] for d in range(block)] for i in range(m)]


    # Objective function

    # cost_between_events = np.sum(np.sum(np.sum(x, axis=3), axis=2) * C_event)
    cost_between_events = np.sum(np.multiply(np.sum(np.sum(x, axis=3), axis=2), C_event))
    cost_home_event = np.sum(np.multiply(np.sum(xhome_event, axis=1), C_home) + np.multiply(np.sum(xevent_home, axis=1),C_home))
    cost_home_depot = np.sum(np.sum(xhome_depot, axis=0)*C_depot[-n:] + np.sum(xdepot_home, axis=0)*C_depot[-n:])
    cost_depot_event = np.sum(np.sum(xdepot_event, axis=1) * np.tile(C_depot[:m], (n, 1)).T 
                            + np.sum(xevent_depot, axis=1) * np.tile(C_depot[:m], (n, 1)).T)
    # print(np.shape(cost_depot_event))

    objective = cost_between_events + cost_home_event + cost_home_depot + cost_depot_event

    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # fix all x[i][j][d][w] to 0 if 1) i == j or 2) time window is infeasible: no overlapping days (double check)
    for i in range(m):
        for j in range(m):
            if i == j:
                for d in range(block):
                    for w in range(n):
                            model.addConstr(x[i][j][d][w] == 0)
            

    # Each event is scheduled exactly once during its feasible time window
    # pruning: only add constraints when there exists a time window for the event on that day
    for i in range(m):
        model.addConstr(sum(s[i][d] for d in range(block)) == 1)
        for d in range(block):
            # set s[i][d], t[i][d] = 0 if time window is [0, 0]
            # no flow for events with time window [0, 0]
            if time_window[i][d][1] == 0:
                model.addConstr(s[i][d] == 0)
                model.addConstr(t[i][d] == 0)
                for w in range(n):
                    model.addConstr(xevent_home[i][d][w] == 0)
                    model.addConstr(xhome_event[i][d][w] == 0)
                    model.addConstr(xevent_depot[i][d][w] == 0)
                    model.addConstr(xdepot_event[i][d][w] == 0)
                    for j in range(m):
                        model.addConstr(x[i][j][d][w] == 0)
                        model.addConstr(x[j][i][d][w] == 0)
            else: # time window is feasible
                for w in range(n):
                    model.addConstr(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w] <= s[i][d])

    # Each event happens 
    for i in range(m):
        model.addConstr(sum(t[i][d] for d in range(block)) >= 1)
        for d in range(block):
            if time_window[i][d][1] > 0:
                model.addConstr(t[i][d] >= time_window[i][d][0] * s[i][d])
                model.addConstr(t[i][d] <= time_window[i][d][1] * s[i][d])

    # Each nurse attends at most 2 events per day
    for d in range(block):
        for w in range(n):
            model.addConstr(sum(x[i][j][d][w] for i in range(m) for j in range(m)) <= 1)

    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                for d in range(block):
                    # only add this constraint when 1) event i is scheduled and 2) the latest time event j can happen is feasible for travel
                    if time_window[i][d][1] > 0 and time_window[j][d][1] >= time_window[i][d][0] + C_dur[i] + C_event[i][j]:
                        for w in range(n):
                            model.addConstr(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j] - M * (1 - x[i][j][d][w]))
                        # model.addConstr(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j]).OnlyEnforceIf(x[i][j][d][w])

    # Minimum working hours (20 hours per week)
    for w in range(n):
        model.addConstr(sum(C_dur[j] * 
                    sum(
                        (sum(x[i][j][d][w] for i in range(m)) 
                        + xhome_event[j][d][w]  
                        + xdepot_event[j][d][w]
                        ) for d in range(block)
                        ) for j in range(m)) >= 60*5) 
            
    # Each event is assigned to the correct number of nurses
    for j in range(m):
        # RN
        model.addConstr(sum((sum(x[i][j][d][w] for i in range(m))
                    + xhome_event[j][d][w] + xdepot_event[j][d][w])
                    for d in range(block) for w in range(nr)) >= min_nurse[j][0])
        # LVN
        model.addConstr(sum((sum(x[i][j][d][w] for i in range(m)) 
                    + xhome_event[j][d][w] + xdepot_event[j][d][w])
                    for d in range(block) for w in range(nr, nr+nl)) >= min_nurse[j][1])

    # Network flow constraints
    # inflow = outflow for each event （does NOT involve depots) 
    for j in range(m):
        for d in range(block):
            # only add constraints when the event is feasible on day d
            if time_window[j][d][1] > 0:
                for w in range(n): # all nurses
                    model.addConstr(sum(x[i][j][d][w] for i in range(m))
                            + xhome_event[j][d][w] 
                            + xdepot_event[j][d][w]
                            == sum(x[j][i][d][w] for i in range(m)) 
                                + xevent_home[j][d][w]
                                + xevent_depot[j][d][w])

    # home -> event, event -> home
    for d in range(block):
        for w in range (n):
            model.addConstr(xhome_depot[d][w] + sum(xhome_event[j][d][w] for j in range(m)) <= 1)
            # network flow for depots
            model.addConstr(xhome_depot[d][w] == sum(xdepot_event[j][d][w] for j in range(m)))
            model.addConstr(xdepot_home[d][w] == sum(xevent_depot[j][d][w] for j in range(m)))

    # team leader constraints
    # one pick up + one drop off leader for each event
    for i in range(m):
        for d in range(block):
            # set alpha[i][d][w] = 0 and beta[i][d][w] = 0 if the event is not scheduled
            if time_window[i][d][1] == 0:
                for w in range(n):
                    model.addConstr(alpha[i][d][w] == 0)
                    model.addConstr(beta[i][d][w] == 0)
            else:
                for w in range(n):
                    # team leader goes to the event
                    model.addConstr(alpha[i][d][w] <= sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w])
                    model.addConstr(beta[i][d][w] <= sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w])
        # exactly one nurse is the team leader
        model.addConstr(sum(alpha[i][d][w] for w in range(n) for d in range(block)) == 1)
        model.addConstr(sum(beta[i][d][w] for w in range(n) for d in range(block)) == 1)
                
    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    for w in range(n):
        for d in range(block):
            # team leader goes from home to depot to their first event if they are the team leader for some event on that day
            for i in range(m):
                model.addConstr(sum(alpha[i][d][w] for i in range(m)) <= 5 * xhome_depot[d][w])
                model.addConstr(sum(beta[i][d][w] for i in range(m)) <= 5 * xdepot_home[d][w])
            
    try:
        model.optimize()
    except gp.GurobiError:
        print("Optimize failed due to infeasibility")

    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3
    model.optimize()

    if model.SolCount > 0:
        # print('The optimal objective is %g' % model.objVal)
        
        with open(f"network_{nr}_{nl}_{m}_seed{seed_number}.txt", "w") as file:
            file.write(f"Best feasible solution found within {time_limit} seconds:\n")
            # Write the objective value to the file
            file.write(f"\nObjective value: {model.ObjVal}\n")
            for d in range(block):
                file.write(f"\n=====================\nDay {d+1} \n=====================\n")
                for i in range(m):
                    if s[i][d].x == 1:
                        file.write(f"\nEvent {i+1}: {get_time(t[i][d].x)}-{get_time(t[i][d].x + C_dur[i])} [{min_nurse[i][0]} RN, {min_nurse[i][1]} LVN] \n")
                        for w in range(n):
                            if alpha[i][d][w].x == 1:
                                file.write(f"Nurse {w+1} (pick up leader)\n")
                            if beta[i][d][w].x == 1:
                                file.write(f"Nurse {w+1} (drop off leader)\n")
                            else:
                                if sum(x[j][i][d][w].x for j in range(m)) + xhome_event[i][d][w].x + xdepot_event[i][d][w].x >= 1:
                                    file.write(f"Nurse {w+1}\n")
                        # for w in range(nr, nr+nl):
                        #     if sum(x[j][i][d][w].x for j in range(m)) + xhome_event[i][d][w].x >= 1:
                        #             file.write(f"LVN {w+1}\n")
    else:
        print(f"No feasible solution found within {time_limit} seconds.")

gurobi_solve()



