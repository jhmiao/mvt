#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

# helper functions
def get_time(minute):
    hour = int(minute // 60 + 9)
    minute = int(minute % 60)
    if minute < 10:
        minute = "0" + str(minute)
    return f"{hour}:{minute}"

# Read the Excel file
file_path = 'input_parameters_real.xlsx'

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
# keep the first m rows and m columns
C_event = C_event[:m, :m]

# Read the C_home matrix
C_home_df = pd.read_excel(file_path, sheet_name='C_home')
C_home = C_home_df.values
# keep the first m rows and n columns
C_home = C_home[:m, :]

# Read the C_depot array
C_depot_df = pd.read_excel(file_path, sheet_name='C_depot')
C_depot = C_depot_df['Depot_Cost'].values
# keep the first m rows and last n rows
C_depot = C_depot[:m].tolist() + C_depot[m:].tolist()

# Read the C_dur array
C_dur_df = pd.read_excel(file_path, sheet_name='C_dur')
C_dur = C_dur_df['Duration'].values
# keep the first m rows
C_dur = C_dur[:m]

# Read the time_window matrix
time_window_df = pd.read_excel(file_path, sheet_name='Time_Window')
time_window_flat = time_window_df.values
time_window = time_window_flat.reshape((m, block, 2))
# randomly replace some [0,0] time windows with [30, 270] with p = 0.25
np.random.seed(seed_number)
for i in range(m):
    for d in range(block):
        if time_window[i][d][0] == 0 and time_window[i][d][1] == 0:
            if np.random.rand() < 0.25:
                time_window[i][d][0] = 30
                time_window[i][d][1] = 270

# Read the minimum nurses required matrix
min_nurse_df = pd.read_excel(file_path, sheet_name='Min_Nurse')
min_nurse = min_nurse_df.values
# keep the first m rows and 2 columns
min_nurse = min_nurse[:m, :]


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
    # i, j == -1 for depot, i, j == -2 for home
    x = model.addVars(m+2, m+2, block, n, vtype=GRB.BINARY, name="x") 

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = model.addVars(m, block, vtype=GRB.BINARY, name="s")

    # t_id time when event i starts on day d
    t = model.addVars(m, block, vtype=GRB.INTEGER, name="t")

    # alpha_idw = 1 if nurse w is the pick-up leader for event i on day d, 0 otherwise
    alpha = model.addVars(m, block, n, vtype=GRB.BINARY, name="alpha")

    # # beta_idw = 1 if nurse w is the drop-off leader for event i on day d, 0 otherwise
    beta = model.addVars(m, block, n, vtype=GRB.BINARY, name="beta")

    # Objective function
    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(block) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[i][w] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(block)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+1, d, w] )for d in range(block) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[w + m] * gp.quicksum((x[m+1, m, d, w] + x[m, m+1, d, w]) for d in range(block)) for w in range(n)
    )


    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # fix all x[i][j][d][w] to 0 if 1) i == j or 2) time window is infeasible: no overlapping days (double check)
    for i in range(m+2):
        for j in range(m+2):
            if i == j:
                model.addConstrs(x[i,j,d,w] == 0 for d in range(block) for w in range(n))
            

    # Each event is scheduled exactly once during its feasible time window
    # pruning: only add constraints when there exists a time window for the event on that day
    for i in range(m):
        model.addConstr(sum(s[i,d] for d in range(block)) == 1)
        for d in range(block):
            # set s[i][d], t[i][d] = 0 if time window is [0, 0]
            # no flow for events with time window [0, 0]
            if time_window[i][d][1] == 0:
                model.addConstr(s[i,d] == 0)
                model.addConstr(t[i,d] == 0)
                model.addConstrs(x[i,j,d,w] == 0 for j in range(m+2) for w in range(n))
                model.addConstrs(x[j,i,d,w] == 0 for j in range(m+2) for w in range(n))

            else: # time window is feasible
                model.addConstrs(sum(x[i,j,d,w] for j in range(m)) <= s[i,d] for w in range(n))

    model.addConstrs(t[i,d] <= time_window[i][d][1] for i in range(m) for d in range(block))
    # Each event happens 
    model.addConstrs(gp.quicksum(t[i,d] for d in range(block)) >= 1 for i in range(m))
    for i in range(m):
        if time_window[i][d][1] > 0:
            model.addConstrs(t[i,d] >= time_window[i][d][0] * s[i,d] for d in range(block))
            model.addConstrs(t[i,d] <= time_window[i][d][1] * s[i,d] for d in range(block))

    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                for d in range(block):
                    # only add this constraint when 1) event i is scheduled and 2) the latest time event j can happen is feasible for travel
                    if time_window[i][d][1] > 0 and time_window[j][d][1] >= time_window[i][d][0] + C_dur[i] + C_event[i][j]:
                        model.addConstrs(t[j,d] >= t[i,d] + C_dur[i] + C_event[i][j] - M * (1 - x[i,j,d,w]) for w in range(n))
                    else:
                        # not travelling from i to j
                        model.addConstrs(x[i,j,d,w] == 0 for w in range(n))
 
    # Minimum working hours (10 hours per week)
    model.addConstrs(
        (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(block)) for j in range(m)) >= 10 * 60 for w in range(n)),
        name="min_working_hours"
    )
    
    # staffing
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(block) for w in range(nr)) >= min_nurse[j][0] for j in range(m)),
        name="min_RN"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(block) for w in range(nr, nr+nl)) >= min_nurse[j][1] for j in range(m)),
        name="min_LVN"
    )        
  
    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2)) == gp.quicksum(x[j, i, d, w] for i in range(m+2)) for j in range(m) for d in range(block) for w in range(n)),
        name="event_network_flow"
    )

    # outflow from home is at most 1
    model.addConstrs(
        (gp.quicksum(x[m, i, d, w] for i in range(m+2)) <= 1 for d in range(block) for w in range(n)),
        name="home_outflow"
    )

    # depot inflow = outflow
    model.addConstrs(
        (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(block) for w in range(n)),
        name="morning_depot_flow"
    )

    model.addConstrs(
        (x[m+1, m, d, w] == gp.quicksum(x[j, m+1, d, w] for j in range(m)) for d in range(block) for w in range(n)),
        name="evening_depot_flow"
    )
    
    # team leader: exactly one pick up and one drop off leader for each event
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for w in range(n) for d in range(block)) == 1 for j in range(m)),
        name="pick_up_leader"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for w in range(n) for d in range(block)) == 1 for j in range(m)),
        name="drop_off_leader"
    )

    # team leader goes to the event
    model.addConstrs(
        (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+2)) for j in range(m) for d in range(block) for w in range(n)),
        name="pick_up_leader_event"
    )

    model.addConstrs(
        (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+2)) for j in range(m) for d in range(block) for w in range(n)),
        name="drop_off_leader_event"
    )


    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(block) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+1, m, d, w] for d in range(block) for w in range(n)),
        name="drop_off_leader_depot_home"
    )


    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.Params.Threads = 8
    model.setParam('LogFile', 'gurobi_output_1p2.txt')
    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3

    start_time = time.time()

    model.optimize()
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)  # in seconds, rounded to 2 decimals

    if model.SolCount > 0:
        # print('The optimal objective is %g' % model.objVal)
        
        file_path1 = f'/Users/jinghongmiao/Code/mvt-code/result-250429/1p2_EVENT_{nr}_{nl}_{m}_seed{seed_number}.txt'
        file_path2 = f'/Users/jinghongmiao/Code/mvt-code/result-250429/1p2_NURSE_{nr}_{nl}_{m}_seed{seed_number}.txt'

        with open(file_path1, "w") as file:
            file.write(f"Best feasible solution found within {elapsed_time} seconds:\n")
            file.write(f"Objective value: {model.ObjVal}\n")
            file.write(f"Best bound: {model.ObjBound}\n")
            file.write(f"Gap: {model.MIPGap}\n")

            for d in range(block):
                file.write(f"\n=====================\nDay {d+1} \n=====================\n")
                for i in range(m):
                    if s[i,d].x == 1:
                        file.write(f"\nEvent {i+1}: {get_time(t[i,d].x)}-{get_time(t[i,d].x + C_dur[i])} [{min_nurse[i][0]} RN, {min_nurse[i][1]} LVN] \n")
                        for w in range(n):
                            if alpha[i,d,w].x == 1 and beta[i,d,w].x == 0:
                                file.write(f"Nurse {w+1} (pick up leader)\n")
                            elif beta[i,d,w].x == 1 and alpha[i,d,w].x == 0:
                                file.write(f"Nurse {w+1} (drop off leader)\n")
                            elif alpha[i,d,w].x == 1 and beta[i,d,w].x == 1:
                                file.write(f"Nurse {w+1} (pick up leader, drop off leader)\n")
                            else:
                                if sum(x[j,i,d,w].x for j in range(m+2)) >= 1:
                                    file.write(f"Nurse {w+1}\n")
        
        with open(file_path2, "w") as file:
            file.write(f"Best feasible solution found within {elapsed_time} seconds:\n")
            file.write(f"Objective value: {model.ObjVal}\n")
            file.write(f"Best bound: {model.ObjBound}\n")
            file.write(f"Gap: {model.MIPGap}\n")

            for w in range(n):
                file.write(f"\n=====================\nNurse {w+1} \n=====================\n")
                for d in range(block):
                    file.write(f"\nDay {d+1}:\n")
                    for i in range(m):
                        if alpha[i,d,w].x == 1:
                            file.write(f"Pick up for Event {i+1}\n")
                        if beta[i,d,w].x == 1:
                                file.write(f"Drop off for Event {i+1}\n")

                    if x[m, m+1, d, w].x == 1:
                        file.write("Home -> Depot\n")

                    for i in range(m):
                        if x[m,i,d,w].x == 1:
                            file.write(f"Home -> Event {i+1}, {get_time(t[j,d].x)}\n")   
                        if x[m+1,i,d,w].x == 1:
                            file.write(f"Depot -> Event {i+1}, {get_time(t[j,d].x)}\n")

                        for j in range(m):
                            if x[i,j,d,w].x == 1:
                                file.write(f"Event {i+1} -> Event {j+1}, {get_time(t[j,d].x)}\n")
                        
                        if x[i,m,d,w].x == 1:
                            file.write(f"Event {i+1} -> Home\n")
                        
                        elif x[i,m+1,d,w].x == 1:
                            file.write(f"Event {i+1} -> Depot\n")
                    
                    if x[m+1, m, d, w].x == 1:
                        file.write("Depot -> Home\n")



    else:
        print(f"No feasible solution found within {time_limit} seconds.")

gurobi_solve()



