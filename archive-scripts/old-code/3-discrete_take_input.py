#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# Read 'input_parameter'
file_path = 'input_parameters_15_35_75_seed120.xlsx'


settings_df = pd.read_excel(file_path, sheet_name='Settings')
settings = settings_df.set_index('Parameter')['Value'].to_dict()

time_limit = int(settings['time_limit'])
seed_number = int(settings['seed_number'])
nr = int(settings['nr'])
nl = int(settings['nl'])
m = int(settings['m'])
block = int(settings['block'])
T = 12 

# Read the time_window matrix
time_window_df = pd.read_excel(file_path, sheet_name='Time_Window')
time_window_flat = time_window_df.values
time_window = time_window_flat.reshape((m, block, 2))
C_dur_df = pd.read_excel(file_path, sheet_name='C_dur')

# Read the minimum nurses required matrix
min_nurse_df = pd.read_excel(file_path, sheet_name='Min_Nurse')
min_nurse = min_nurse_df.values

# =============================================================================
# Read the C_event matrix
C_event_df = pd.read_excel(file_path, sheet_name='C_event')
C_event = C_event_df.values

# Read the C_home matrix
C_home_df = pd.read_excel(file_path, sheet_name='C_home')
C_home = C_home_df.values

# Read the C_depot array
C_depot_df = pd.read_excel(file_path, sheet_name='C_depot')
C_depot = C_depot_df['Depot_Cost'].values


# 
# =============================================================================
#Converting network input to discrete input
C_dur = np.ceil(1 + C_dur_df['Duration'].values / 30).astype(int) # round all 0.5 to next 1

# Initialize the binary array (shape: num_events x day_block x time_slots_per_day)
binary_array = np.zeros((m, block, T), dtype=int)

# Convert the time_window array to a binary array
for i in range(m):
    for j in range(block):
        start, end = time_window[i, j]
        if start != 0 or end != 0:  # Check if the time window is non-zero
            start_slot = start // 30
            end_slot = end // 30
            binary_array[i, j, start_slot:end_slot] = 1

# Flatten the binary array for each event across days
binary_array_flat = binary_array.reshape(m, -1)
time_window = binary_array_flat


# helper functions
# translate the time slot to the actual time
def time_slot_to_time(time_slot):
    day = time_slot // T + 1
    hour = int((time_slot % T) // 2 + 9)
    minute = int((time_slot % T) % 2 * 30)
    if minute < 10:
        minute = "0" + str(minute)
    return f"day {day} at {hour}:{minute}"

# def time_slot_to_minute(time_slot):
#     day = time_slot // T + 1
#     hour = int((time_slot % T) // 2 + 9)
#     minute = int((time_slot % T) % 2 * 30)
#     if minute < 10:
#         minute = "0" + str(minute)
#     return f"day {day} at {hour}:{minute}"

# create a vector of length T*block, where s_vector[0:block]=0, s_vector[block:2*block]=1, ...
s_vector = np.zeros(T*block)
for d in range(block):
    s_vector[d*T:(d+1)*T] = d

# create a vector of length T*block, where s_vector[0:block]=[0,30,..., (T_1)*30] for each day, s_vector[block:2*block]=[0,30,..., (T_2)*30] for each day, ...
t_vector = np.zeros(T*block)
for d in range(block):
    t_vector[d*T:(d+1)*T] = np.arange(T) * 30

# returns the day index when the event is scheduled
# def get_s_id (fj):
#     return np.dot(fj, s_vector)

# returns the time slot index when the event is scheduled
# def get_t_id (fj):
#     return np.dot(fj, t_vector)

# returns whether the nurse goes from event i to j on day d
# input: y matrix for nurse w; each row is an event, each column is a time slot
# input: p vector for nurse w: vector of length m, p[j] = 1 if nurse w picks up equipment for event j
# input: d vector for nurse w: vector of length m, d[j] = 1 if nurse w drops off equipment for event j
# output: 
# def get_x_ijd (y_matrix, p_vector, d_vector, d):
#     # initialize the output variables
#     event_1 = None
#     event_2 = None
#     x_12d = None
#     # x_jid = None
#     xhome_depot = 0
#     xdepot_home = 0
#     xdepot_event_1 = 0
#     xhome_event_1 = 0
#     xevent_depot_1 = 0
#     xevent_home_1 = 0
#     xevent_depot_2 = None
#     xevent_home_2 = None
    
#     # sum up each day in the y matrix
#     y_day = np.sum(np.sum(y_matrix[:, d*T:(d+1)*T], axis=0))
#     if y_day == 1:
#         # find the event index
#         event_1 = np.where(np.sum(y_matrix[:, d*T:(d+1)*T], axis=1) == 1)[0][0]
#         if np.sum(p_vector[event_1]) == 1:
#             xhome_depot = 1
#             xdepot_event_1 = 1
#         else:
#             xhome_event_1 = 1

#         if np.sum(d_vector[i]) == 1:
#             xevent_depot_1 = 1
#             xdepot_home= 1
#         else:
#             xevent_home_1 = 1
    
#     elif y_day == 2:
#         # find the event index
#         row_i = np.where(np.sum(y_matrix[:, d*T:(d+1)*T], axis=1) == 1)[0][0]
#         row_j = np.where(np.sum(y_matrix[:, d*T:(d+1)*T], axis=1) == 1)[0][1]
        
#         # set event_1 to be the event that starts first
#         if get_t_id(y_matrix[row_i]) < get_t_id(y_matrix[row_j]):
#             event_1 = row_i
#             event_2 = row_j
#         else:
#             event_1 = row_j
#             event_2 = row_i

#         x_12d = 1
#         if np.sum(p_vector[event_1]) == 1 or np.sum(p_vector[event_2]) == 1:
#             xhome_depot = 1
#             xdepot_event_1 = 1
#             # xhome_event_1 = 0
#         else:
#             xhome_event_1 = 1

#         if np.sum(d_vector[event_1]) == 1 or np.sum(d_vector[event_2]) == 1:
#             xevent_depot_2 = 1
#             xdepot_home = 1
#             xevent_home_2 = 0
#         else:
#             xevent_home_2 = 1
#             xevent_depot_2 = 0
        
#     return event_1, event_2, x_12d, xhome_depot, xdepot_home, xdepot_event_1, xhome_event_1, xevent_depot_1, xevent_home_1, xevent_depot_2, xevent_home_2


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

    # Decision variables

    # f_j: vector of length T * block, f_j[t] = 1 if event j is scheduled to start at time t
    f = [[model.addVar(vtype=GRB.BINARY, name=f"f_{j}_{t}") for t in range(T*block)] for j in range(m)] 

    # g_j: vector of length T * block, g_j[t] = 1 if event j is scheduled during time slot t
    g = [[model.addVar(vtype=GRB.BINARY, name=f"g_{j}_{t}") for t in range(T*block)] for j in range(m)] 

    # x_jw: vector of legnth T * block * (nr+nl), x_jw[t] = 1 if nurse w is assigned to event j at time t
    x = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}_{w}") for w in range(nr+nl)] for t in range(T*block)]  for j in range(m)] 

    # y_jw: vector of length T * block * (nr+nl), y_jw[t] = 1 if nurse w starts event j at time t
    y = [[[model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{t}_{w}") for w in range(nr+nl)] for t in range(T*block)]  for j in range(m)]
    
    # alpha_jw : a binary variable, p_jw = 1 if w is picking up equipment for event j
    alpha = [[model.addVar(vtype=GRB.BINARY, name=f"alpha_{j}_{w}") for w in range(nr+nl)] for j in range(m)]
    
    # beta_jw : a binary variable, d_jw = 1 if w is dropping off equipment for event j
    beta = [[model.addVar(vtype=GRB.BINARY, name=f"beta_{j}_{w}") for w in range(nr+nl)] for j in range(m)]
    
    # Objective function
    
    # obj1
    max_leader_count = model.addVar(vtype=GRB.CONTINUOUS)
    for w in range(nr + nl):
        total_leader_count_w = sum(alpha[j][w] + beta[j][w] for j in range(m))
        model.addConstr(max_leader_count >= total_leader_count_w)
    

    # obj2
    # minimize the total travel cost to leader locations
    leader_cost = 0
    for w in range(nr + nl):
        leader_cost += sum(alpha[j][w] for j in range(m)) * C_depot[w]
        leader_cost += sum(beta[j][w] for j in range(m)) * C_depot[w]

    # obj3
    # minimize more travel costs
    travel_cost = 0
    for w in range(nr+nl):
        for j in range(m):
            travel_cost += sum(y[j][t][w] for t in range(T*block)) * C_home[j][w]
        # np.sum(np.sum(xhome_event_matrix, axis=1)* C_home + np.sum(xevent_home_matrix, axis=1)*C_home)

    # model.setObjective(max_leader_count, GRB.MINIMIZE)
    model.setObjective(leader_cost + travel_cost, GRB.MINIMIZE)
   

    # Constraints

    ### New constraints: force max of 2 events per person per day
    for w in range(nr+nl):
        for d in range(block):
            model.addConstr(sum(y[j][t][w] for j in range(m) for t in range(d*T, (d+1)*T)) <= 2)

    # Each event is scheduled exactly once
    for j in range(m):
        model.addConstr(sum(f[j][t] for t in range(T*block)) == 1)

    # Each event is scheduled exactly once during its feasible time window
    for j in range(m):
        for t in range(T*block):
            model.addConstr(g[j][t] == sum(f[j][s] for s in range(np.maximum(t - C_dur[j]+1, 0), t+1))) 

    # nurse goes to an event only when the event is scheduled
    for j in range (m):
        for w in range(nr+nl):
            for t in range(T*block):
                model.addConstr(x[j][t][w] <= g[j][t])
                model.addConstr(y[j][t][w] <= f[j][t])

    # nurse goes to an event only when the event is scheduled
    for j in range (m):
        for w in range(nr+nl):
            for t in range(T*block):
                model.addConstr(x[j][t][w] == sum(y[j][s][w] for s in range(np.maximum(t - C_dur[j]+1, 0), t+1)))

    # Each nurse is assigned to exactly one event at any time
    for w in range(nr+nl):
        for t in range(T*block):
            model.addConstr(sum(x[j][t][w] for j in range(m)) <= 1)
    
    # each event happens during its feasible time window
    for j in range(m):
        for t in range(T*block):
            if time_window[j][t] == 0:
                model.addConstr(f[j][t] == 0)

    # Minimum working hours (20 hours per week)
    for w in range(nr+nl):
        model.addConstr(sum(x[j][t][w] for j in range(m) for t in range(T*block)) >= 3)
            
    # Each event is assigned to the correct number of nurses
    for j in range(m):
        for t in range(T*block):
            model.addConstr(sum(x[j][t][w] for w in range(nr)) >= min_nurse[j][0] * g[j][t])
            model.addConstr(sum(x[j][t][w] for w in range(nr, nr+nl)) >= min_nurse[j][1] * g[j][t])

            # model.addConstr(sum(x[j][t][w] for w in range(nr)) >= min_nurse[j][0] * f[j][t])
            # model.addConstr(sum(x[j][t][w] for w in range(nr, nr+nl)) >= min_nurse[j][1] * f[j][t])
    
    # Prevent overstaffing:
    for j in range(m):
        model.addConstr(sum(y[j][t][w] for w in range(nr) for t in range(T*block)) <= 2 * min_nurse[j][0])
        model.addConstr(sum(y[j][t][w] for w in range(nr,nr+nl) for t in range(T*block)) <= 2 * min_nurse[j][1])

    # Exactly one pick-up/drop-off leader for each event for each event j
    for j in range (m):
        model.addConstr(sum(alpha[j][w] for w in range(nr + nl)) == 1)
        model.addConstr(sum(beta[j][w] for w in range(nr + nl)) == 1)
    
    # The pick-up / drop-off leader has to be selected from those who go to the event
    for j in range(m):
        for w in range(nr + nl):
            sum_y_w = sum(y[j][t][w] for t in range(T * block))
            model.addConstr(alpha[j][w] <= sum_y_w)
            model.addConstr(beta[j][w] <= sum_y_w)


    # try:
    #     model.optimize()
    # except gp.GurobiError:
    #     print("Optimize failed due to infeasibility")

    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.optimize()

    

    if model.SolCount > 0:
        # print('The optimal objective is %g' % model.objVal)
                # initiate empty arrays
        # recall
        # x = [[[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)] for i in range(m)]
        # xhome_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_0{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)]
        # xevent_home = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}0{d}{w}") for w in range(nr+nl)] for d in range(block)] for i in range(m)]
        # xhome_depot = [[model.addVar(vtype=GRB.BINARY, name=f"x_0*{d}{w}") for w in range(nr+nl)] for d in range(block)]
        # # xdepot_home = [[model.addVar(vtype=GRB.BINARY, name=f"x_*0{d}{w}") for w in range(nr+nl)] for d in range(block)]
        # xdepot_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_*{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)]
        # xevent_depot = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}*{d}{w}") for w in range(nr+nl)] for d in range(block)] for i in range(m)]

        x_matrix = np.zeros((m, m, block, nr+nl))
        xhome_event_matrix = np.zeros((m, block, nr+nl))
        xevent_home_matrix = np.zeros((m, block, nr+nl))
        xhome_depot_matrix = np.zeros((block, nr+nl))
        xdepot_home_matrix = np.zeros((block, nr+nl))
        xdepot_event_matrix = np.zeros((m, block, nr+nl))
        xevent_depot_matrix = np.zeros((m, block, nr+nl))

        for w in range(nr+nl):
            # stack the y[j][t][w] into a matrix for each w
            y_matrix = np.zeros((m, T*block))
            for j in range(m):
                for t in range(T*block):
                    y_matrix[j][t] = y[j][t][w].x
            # stack the p[j][w] into a vector for each w
            p_vector = np.zeros(m)
            for j in range(m):
                p_vector[j] = alpha[j][w].x
            # stack the d[j][w] into a vector for each w
            d_vector = np.zeros(m)
            for j in range(m):
                d_vector[j] = beta[j][w].x
                
            for d in range(block):
                event_1, event_2, x_12d, xhome_depot, xdepot_home, xdepot_event_1, xhome_event_1, xevent_depot_1, xevent_home_1, xevent_depot_2, xevent_home_2 = get_x_ijd(y_matrix, p_vector, d_vector, d)
                # file.write(f"\nNurse {w+1}:\n")
                
                if event_1 is not None:
                    xhome_depot_matrix[d][w] = xhome_depot
                    xdepot_event_matrix[event_1][d][w] = xdepot_event_1
                    xhome_event_matrix[event_1][d][w] = xhome_event_1
                    xevent_home_matrix[event_1][d][w] = xevent_home_1
                    xevent_depot_matrix[event_1][d][w] = xevent_depot_1
                    xdepot_home_matrix[d][w] = xdepot_home

                    if event_2 is not None:
                        x_matrix[event_1][event_2][d][w] = x_12d
                        xevent_depot_matrix[event_2][d][w] = xevent_depot_2
                        xevent_home_matrix[event_2][d][w] = xevent_home_2

            # calculate the travel cost
            # # another objective function

            cost_between_events = np.sum(np.sum(np.sum(x_matrix, axis=3), axis=2) * C_event)
            cost_home_event = np.sum(np.sum(xhome_event_matrix, axis=1)* C_home + np.sum(xevent_home_matrix, axis=1)*C_home)
            cost_home_depot = np.sum(np.sum(xhome_depot_matrix, axis=0)*C_depot[-(nr+nl):] + np.sum(xdepot_home_matrix, axis=0)*C_depot[-(nr+nl):])
            cost_depot_event = np.sum(np.sum(xdepot_event_matrix, axis=1) * np.tile(C_depot[:m], (nr+nl, 1)).T 
                                    + np.sum(xevent_depot_matrix, axis=1) * np.tile(C_depot[:m], (nr+nl, 1)).T)
            # print(np.shape(cost_depot_event))

            travel_cost = cost_between_events + cost_home_event + cost_home_depot + cost_depot_event


        with open(f"discrete_{nr}_{nl}_{m}_seed{seed_number}.txt", "w") as file:
            file.write(f"Best feasible solution found within {time_limit} seconds:\n")
            file.write(f"Seed number: {seed_number}\n")
            file.write(f"Number of RNs: {nr}\n")
            file.write(f"Number of LVNs: {nl}\n")
            file.write(f"Objective value: {model.objVal}\n")
            file.write(f"Total travel cost: {travel_cost}\n")
            # file.write("Iterations: ")
            # file.write(str(iterations))
            # file.write("\n")
            file.write("Schedule: \n\n")

            for j in range(m):
                for t in range(T*block):
                    if f[j][t].x == 1:
                        file.write(f"\nEvent {j+1}: {time_slot_to_time(t)}\n")
                        file.write(f"Duaration: {C_dur[j]*30} minutes \n")
                        file.write(f"Minimum number of nurses required: [{min_nurse[j][0]} RN, {min_nurse[j][1]} LVN] \n")
                        for w in range(nr+nl):
                            if x[j][t][w].x == 1:
                                if w < nr:
                                    file.write(f"RN {w+1}\n")
                                else:
                                    file.write(f"LVN {w+1}\n")
                                    
                        for w in range(nr+nl):
                            if alpha[j][w].x == 1:
                                if w < nr:
                                    file.write(f"Pick up: RN {w+1}\n")
                                else:
                                    file.write(f"Pick up: LVN {w+1}\n")
                                    
                        for w in range(nr+nl):
                            if beta[j][w].x == 1:
                                if w < nr:
                                    file.write(f"Drop off: RN {w+1}\n")
                                else:
                                    file.write(f"Drop off: LVN {w+1}\n")
            file.write(f"\n\nNurse schedule:\n")
            for w in range(nr+nl):
                file.write(f"\nNurse {w+1}:\n")
                for t in range(T*block):
                    for j in range(m):
                        if y[j][t][w].x == 1:
                            file.write(f"Event {j+1}: {time_slot_to_time(t)}, {C_dur[j]*30} minutes \n")
                            break
        # with open(f"input_{nr}_{nl}_{m}_seed{seed_number}.txt", "w") as file:


        #     file.write(f"Seed number: {seed_number}\n")
        #     file.write(f"Number of RNs: {nr}\n")
        #     file.write(f"Number of LVNs: {nl}\n")
        #     file.write(f"Number of events: {m}\n")
        #     file.write(f"\n\nTotal travel cost: {travel_cost}\n")
            
        #     for w in range(nr+nl):
        #         # stack the y[j][t][w] into a matrix for each w
        #         y_matrix = np.zeros((m, T*block))
        #         for j in range(m):
        #             for t in range(T*block):
        #                 y_matrix[j][t] = y[j][t][w].x
        #         # stack the p[j][w] into a vector for each w
        #         p_vector = np.zeros(m)
        #         for j in range(m):
        #             p_vector[j] = alpha[j][w].x
        #         # stack the d[j][w] into a vector for each w
        #         d_vector = np.zeros(m)
        #         for j in range(m):
        #             d_vector[j] = beta[j][w].x
                
        #         for d in range(block):
        #             event_1, event_2, x_12d, xhome_depot, xdepot_home, xdepot_event_1, xhome_event_1, xevent_depot_1, xevent_home_1, xevent_depot_2, xevent_home_2 = get_x_ijd(y_matrix, p_vector, d_vector, d)
        #             # file.write(f"\nNurse {w+1}:\n")
                    
        #             if event_1 is not None:
        #                 file.write(f"x_#_*_{d}_{w}, {xhome_depot}\n")
        #                 file.write(f"x_*_{event_1}_{d}_{w}, {xdepot_event_1}\n")
        #                 file.write(f"x_#_{event_1}_{d}_{w}, {xhome_event_1}\n")

        #                 file.write(f"x_{event_1}_#_{d}_{w}, {xevent_home_1}\n")
        #                 file.write(f"x_{event_1}_*_{d}_{w}, {xevent_depot_1}\n")
        #                 file.write(f"x_*_#_{d}_{w}, {xdepot_home}\n")

        #                 if event_2 is not None:
        #                     file.write(f"x_{event_1}_{event_2}_{d}_{w}, {x_12d}\n")
        #                     # if xevent_depot_2 is not None:
        #                     file.write(f"x_{event_2}_*_{d}_{w}, {xevent_depot_2}\n")
        #                     file.write(f"x_{event_2}_#_{d}_{w}, {xevent_home_2}\n")

            
            
    print("No feasible solution found within 20 minutes.")


gurobi_solve()


