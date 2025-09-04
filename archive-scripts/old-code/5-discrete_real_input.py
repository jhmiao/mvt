#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gurobipy as gp 
from gurobipy import GRB, LinExpr, MVar

# force 2 event per person per day

# Read 'input_parameter'
file_path = 'input_parameters_real.xlsx'


settings_df = pd.read_excel(file_path, sheet_name='Settings')
settings = settings_df.set_index('Parameter')['Value'].to_dict()

time_limit = int(settings['time_limit'])
seed_number = int(settings['seed_number'])
nr = int(settings['nr'])
nl = int(settings['nl'])
n = nr + nl
m = int(settings['m'])
block = int(settings['block'])
T = 16

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
def get_t_id (fj):
    return np.dot(fj, t_vector)

# returns whether the nurse goes from event i to j on day d
# input: y matrix for nurse w; each row is an event, each column is a time slot
# input: p vector for nurse w: vector of length m, p[j] = 1 if nurse w picks up equipment for event j
# input: d vector for nurse w: vector of length m, d[j] = 1 if nurse w drops off equipment for event j
# output: 
def get_x_ijd (y_matrix, p_vector, d_vector, d):
    # initialize the output variables
    event_1 = None
    event_2 = None
    
    xhome_depot = 0
    xdepot_event = 0
    xhome_event = 0

    x_12d = 0

    xevent_depot = 0
    xdepot_home = 0
    xevent_home = 0

    # find the event index

    # check if there are 2 events scheduled on day d
    if np.sum(y_matrix[:, d*T:(d+1)*T]) == 0:
        return event_1, event_2, x_12d, xhome_depot, xdepot_event, xhome_event, xevent_depot, xdepot_home, xevent_home
    
    elif np.sum(y_matrix[:, d*T:(d+1)*T]) == 1:
        event_1 = np.where(np.sum(y_matrix[:, d*T:(d+1)*T], axis=1) == 1)[0][0]
        if np.sum(p_vector[event_1]) == 1:
            xhome_depot = 1
            xdepot_event = 1
        else:
            xhome_event = 1
        if np.sum(d_vector[event_1]) == 1:
            xevent_depot = 1
            xdepot_home = 1
        else:
            xevent_home = 1
        return event_1, event_2, x_12d, xhome_depot, xdepot_event, xhome_event, xevent_depot, xdepot_home, xevent_home
    
    elif np.sum(y_matrix[:, d*T:(d+1)*T]) == 2:

        row_i = np.where(np.sum(y_matrix[:, d*T:(d+1)*T], axis=1) == 1)[0][0]
        row_j = np.where(np.sum(y_matrix[:, d*T:(d+1)*T], axis=1) == 1)[0][1]
        
        # set event_1 to be the event that starts first
        if get_t_id(y_matrix[row_i]) < get_t_id(y_matrix[row_j]):
            event_1 = row_i
            event_2 = row_j
        else:
            event_1 = row_j
            event_2 = row_i

        if np.sum(p_vector[event_1]) == 1 or np.sum(p_vector[event_2]) == 1:
            xhome_depot = 1
            xdepot_event = 1
        else:
            xhome_event = 1

        if np.sum(d_vector[event_1]) == 1 or np.sum(d_vector[event_2]) == 1:
            xevent_depot = 1
            xdepot_home = 1
        else:
            xevent_home = 1
            
        return event_1, event_2, x_12d, xhome_depot, xdepot_event, xhome_event, xevent_depot, xdepot_home, xevent_home


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

    # x_jw: vector of legnth T * block * (nr+nl), x_jw[t] = 1 if nurse w is assigned to event j at time t
    x = model.addVars(m, T*block, n, vtype=GRB.BINARY, name="x")

    # y_jw: vector of length T * block * (nr+nl), y_jw[t] = 1 if nurse w starts event j at time t
    y = model.addVars(m, T*block, n, vtype=GRB.BINARY, name="y")

    # f_j: vector of length T * block, f_j[t] = 1 if event j is scheduled during time t
    f = model.addVars(m, T*block, vtype=GRB.BINARY, name="f")

    # g_j: vector of length T * block, g_j[t] = 1 if event j is scheduled at time slot t
    g = model.addVars(m, T*block, vtype=GRB.BINARY, name="g")

    # alpha_jw : a binary variable, alpha_jw = 1 if w is picking up equipment for event j
    alpha = model.addVars(m, n, vtype=GRB.BINARY, name="alpha")
    
    # beta_jw : a binary variable, beta_jw = 1 if w is dropping off equipment for event j
    beta = model.addVars(m, n, vtype=GRB.BINARY, name="beta")

    # Objective function
    
    # obj1
    # max_leader_count = model.addVar(vtype=GRB.CONTINUOUS)
    # for w in range(nr + nl):
    #     total_leader_count_w = sum(alpha[j,w] + beta[j,w] for j in range(m))
    #     model.addConstr(max_leader_count >= total_leader_count_w)
    
    # obj2
    # minimize the total travel cost to leader locations
    leader_cost = 0
    for w in range(nr + nl):
        leader_cost += sum(alpha[j,w] for j in range(m)) * C_depot[w]
        leader_cost += sum(beta[j,w] for j in range(m)) * C_depot[w]

    # obj3
    # minimize more travel costs
    travel_cost = 0
    for w in range(n):
        for j in range(m):
            travel_cost += sum(y[j,t,w] for t in range(T*block)) * C_home[j][w]
        # np.sum(np.sum(xhome_event_matrix, axis=1)* C_home + np.sum(xevent_home_matrix, axis=1)*C_home)
    total_cost = leader_cost + travel_cost
    
    # # obj4
    # # given each person attends 2 events per day
    # # introduce new variable A_j and P_j
    # a = model.addVars(m, vtype=GRB.BINARY, name="am")
    # p = model.addVars(m, vtype=GRB.BINARY, name="pm")

    # # a_j = 1 if event j is scheduled during morning blocks 
    # # a_j = sum(g_j[t] for t in range(0, T/2)) + sum(g_j[t] for t in range(T, 3*T/2)) + ...
    # for j in range(m):
    #     model.addConstr(
    #         a[j] == gp.quicksum(
    #             g[j, t] for k in range(block) for t in range(k * T, (2 * k + 1) * T // 2)
    #         )
    # )

    # # p_j = 1 if event j is scheduled during afternoon blocks
    # # p_j = sum(g_j[t] for t in range(T/2, T)) + sum(g_j[t] for t in range(3*T/2, 2*T)) + ...
    # for j in range(m):
    #     model.addConstr(
    #         p[j] == gp.quicksum(
    #             g[j, t] for k in range(block) for t in range((2 * k + 1) * T // 2, (k + 1) * T)
    #         )
    # )
    
    # home_depot_cost = 0
    # depot_home_cost = 0
    # home_event_cost = 0
    # event_home_cost = 0
    # event_event_cost = 0

    # # home_depot_cost = gp.quicksum(
    # #     alpha[j,w] * C_depot[-n:][w] * a[j] for j in range(m) for w in range(n)
    # # )

    # # depot_home_cost = gp.quicksum(
    # #     beta[j,w] * C_depot[-n:][w] * p[j] for j in range(m) for w in range(n)
    # # )

    # # home_event_cost = gp.quicksum(
    # #     p[j] * (gp.quicksum(y[j, t, w] for t in range(T * block)) - alpha[j, w]) * C_home[j][w]
    # #     for j in range(m)
    # #     for w in range(n)
    # # )

    # # event_home_cost = gp.quicksum(
    # #     a[j] * (gp.quicksum(y[j, t, w] for t in range(T * block)) - beta[j, w]) * C_home[j][w]
    # #     for j in range(m)
    # #     for w in range(n)
    # # )

    # event_event_cost = gp.quicksum(
    #     y[j1, t1, w] * C_event[j1, j2] * y[j2, t2, w]
    #     for w in range(n)
    #     for d in range(block)
    #     for t1 in range(d * T, (d + 1) * T)
    #     for t2 in range(t1 + 1, (d + 1) * T)  # Ensure t2 > t1
    #     for j1 in range(m)
    #     for j2 in range(m)
    # )


    # total_cost = home_depot_cost + depot_home_cost + home_event_cost + event_home_cost + event_event_cost

    # model.setObjective(max_leader_count, GRB.MINIMIZE)
    model.setObjective(total_cost, GRB.MINIMIZE)
   

    # Constraints
    # pruning: if time window is not scheduled, , set all relevant 

    # New constraints: force 2 events per person per day
    model.addConstrs(
        (
            gp.quicksum(y[j, t, w] for j in range(m) for t in range(d * T, (d + 1) * T)) <= 2
            for w in range(n)
            for d in range(block)
        ),
        name="events_per_person_per_day"
    )

    # Each event is scheduled exactly once
    model.addConstrs(
        (gp.quicksum(g[j, t] for t in range(T * block)) == 1 for j in range(m)),
        name="event_scheduled_once"
    )

    # link f and g
    model.addConstrs(
        (gp.quicksum(g[j, s] for s in range(np.maximum(t - C_dur[j] + 1, 0), t + 1)) == f[j, t] for j in range(m) for t in range(T * block)),
        name="link_f_g"
    )

    # nurse goes to an event only when the event is scheduled
    model.addConstrs(
        (gp.quicksum(y[j, s, w] for s in range(np.maximum(t - C_dur[j] + 1, 0), t + 1)) == x[j, t, w] for j in range(m) for w in range(n) for t in range(T * block)),
        name="link_x_y"
    )

    # nurse goes to an event only when the event is scheduled
    # model.addConstrs(
    #     (x[j, t, w] <= f[j, t] for j in range(m) for w in range(n) for t in range(T * block)),
    #     name="link_x_f"
    # )

    model.addConstrs(
        (y[j, t, w] <= g[j, t] for j in range(m) for w in range(n) for t in range(T * block)),
        name="link_y_g"
    )

    # The pick-up / drop-off leader has to be selected from those who go to the event
    model.addConstrs(
        (alpha[j, w] <= gp.quicksum(y[j, t, w] for t in range(T * block)) for j in range(m) for w in range(n)),
        name="pickup_leader_from_event"
    )
    model.addConstrs(
        (beta[j, w] <= gp.quicksum(y[j, t, w] for t in range(T * block)) for j in range(m) for w in range(n)),
        name="dropoff_leader_from_event"
    )

    # Each nurse is assigned to exactly one event at any time
    model.addConstrs(
        (gp.quicksum(x[j, t, w] for j in range(m)) <= 1 for w in range(n) for t in range(T * block)),
        name="one_event_at_a_time"
    )

    # # each event happens during its feasible time window

    model.addConstrs(
        (g[j, t] <= time_window[j][t] for j in range(m) for t in range(T * block)),
        name="event_time_window"
    )

    # each nurse works minimum hours
    model.addConstrs(
        (gp.quicksum(x[j, t, w] for j in range(m) for t in range(T * block)) >= 30 for w in range(n)),
        name="minimum_hours"
    )

    # add to balance workload
    # each nurse works at least 5 events
    model.addConstrs(
        (gp.quicksum(y[j, t, w] for j in range(m) for t in range(T * block)) >= 5 for w in range(n)),
        name="minimum_events"
    )

    # Each event is assigned to the correct number of nurses
    model.addConstrs(
        (gp.quicksum(y[j, t, w] for w in range(nr) for t in range(T * block)) >= min_nurse[j, 0] for j in range(m)),
        name="min_nurse_RN"
    )
    model.addConstrs(
        (gp.quicksum(y[j, t, w] for w in range(nr,n) for t in range(T * block)) >= min_nurse[j, 1] for j in range(m)),
        name="min_nurse_LVN"
    )

    # Prevent overstaffing:
    model.addConstrs(
        (gp.quicksum(y[j, t, w] for w in range(nr) for t in range(T * block)) <= 2 * min_nurse[j, 0] for j in range(m)),
        name="overstaffing_RN"
    )
    model.addConstrs(
        (gp.quicksum(y[j, t, w] for w in range(nr,n) for t in range(T * block)) <= 2 * min_nurse[j, 1] for j in range(m)),
        name="overstaffing_LVN"
    )

    # Exactly one pick-up/drop-off leader for each event j
    model.addConstrs(
        (gp.quicksum(alpha[j, w] for w in range(n)) == 1 for j in range(m)),
        name="one_pickup_leader"
    )
    model.addConstrs(
        (gp.quicksum(beta[j, w] for w in range(n)) == 1 for j in range(m)),
        name="one_dropoff_leader"
    )

    

    try:
        model.setParam(GRB.Param.TimeLimit, 1800)
        model.setParam(GRB.Param.Method, 2)
        # model.computeIIS()
        # model.write("model_infeasibility.ilp")
        model.optimize()
    except gp.GurobiError:
        print("Optimize failed due to infeasibility")

    
    if model.SolCount > 0:
        
        with open(f"discrete_real_2event.txt", "w") as file:
            file.write(f"Best feasible solution found within {time_limit} seconds:\n")
            # file.write(f"Seed number: {seed_number}\n")
            file.write(f"Number of RNs: {nr}\n")
            file.write(f"Number of LVNs: {nl}\n")
            file.write(f"Objective value: {model.objVal}\n")
            # file.write(f"Total approximate travel cost: {leader_cost + travel_cost}\n")
            file.write("Schedule: \n\n")

            for j in range(m):
                for t in range(T*block):
                    if g[j,t].x == 1:
                        file.write(f"\nEvent {j+1}: {time_slot_to_time(t)}\n")
                        file.write(f"Duaration: {C_dur[j]*30} minutes \n")
                        file.write(f"Minimum number of nurses required: [{min_nurse[j][0]} RN, {min_nurse[j][1]} LVN] \n")
                        for w in range(n):
                            if x[j,t,w].x == 1:
                                if w < nr:
                                    file.write(f"RN {w+1}\n")
                                else:
                                    file.write(f"LVN {w+1}\n")
                                    
                        for w in range(n):
                            if alpha[j,w].x == 1:
                                if w < nr:
                                    file.write(f"Pick up: RN {w+1}\n")
                                else:
                                    file.write(f"Pick up: LVN {w+1}\n")
                                    
                        for w in range(n):
                            if beta[j,w].x == 1:
                                if w < nr:
                                    file.write(f"Drop off: RN {w+1}\n")
                                else:
                                    file.write(f"Drop off: LVN {w+1}\n")
            file.write(f"\n\nNurse schedule:\n")
            for w in range(n):
                file.write(f"\nNurse {w+1}:\n")
                for t in range(T*block):
                    for j in range(m):
                        if y[j,t,w].x == 1:
                            file.write(f"Event {j+1}: {time_slot_to_time(t)}, {C_dur[j]*30} minutes \n")
                            break
        
        with open(f"input_real_2event.txt", "w") as file:

            # file.write(f"Seed number: {seed_number}\n")
            file.write(f"Number of RNs: {nr}\n")
            file.write(f"Number of LVNs: {nl}\n")
            file.write(f"Number of events: {m}\n")
            file.write(f"Objective value: {model.objVal}\n")
            
            # write s_i_d where s_i_d = 1 if event i is scheduled on day d
            for i in range(m):
                for d in range(block):
                    sid = sum(g[i,t].x for t in range(d*T, (d+1)*T))
                    tid = sum((g[i,t].x * t_vector[t]) for t in range(d*T, (d+1)*T))
                    if sid > 0:
                        file.write(f"s_{i}_{d}, {sid}\n")

                        # write leader if event is scheduled on the day
                        for w in range(n):
                            if alpha[i,w].x == 1:
                                file.write(f"alpha_{i}_{d}_{w}, {alpha[i,w].x}\n")
                            if beta[i,w].x == 1:
                                file.write(f"beta_{i}_{d}_{w}, {beta[i,w].x}\n")
                    if tid > 0:
                        file.write(f"t_{i}_{d}, {tid}\n")
        
            for w in range(n):
                # stack the y[j][t][w] into a matrix for each w
                y_matrix = np.zeros((m, T*block))
                for j in range(m):
                    for t in range(T*block):
                        y_matrix[j][t] = y[j,t,w].x
                # stack the p[j][w] into a vector for each w
                p_vector = np.zeros(m)
                for j in range(m):
                    p_vector[j] = alpha[j,w].x
                # stack the d[j][w] into a vector for each w
                d_vector = np.zeros(m)
                for j in range(m):
                    d_vector[j] = beta[j,w].x
                
                for d in range(block):
                    event_1, event_2, x_12d, xhome_depot, xdepot_event, xhome_event, xevent_depot, xdepot_home, xevent_home = get_x_ijd(y_matrix, p_vector, d_vector, d)
                    
                    if xhome_depot == 1:
                        file.write(f"x_-2_-1_{d}_{w}, {xhome_depot}\n")

                    if xdepot_event == 1:
                        file.write(f"x_-1_{event_1}_{d}_{w}, {xdepot_event}\n")

                    if xhome_event == 1:
                        file.write(f"x_-2_{event_1}_{d}_{w}, {xhome_event}\n")
                    
                    if x_12d == 1:
                        file.write(f"x_{event_1}_{event_2}_{d}_{w}, {x_12d}\n")
                    
                    if xevent_depot == 1:
                        # if event 2 is not None
                        if event_2 is not None:
                            file.write(f"x_{event_2}_-1_{d}_{w}, {xevent_depot}\n")
                        else:
                            file.write(f"x_-1_{event_1}_{d}_{w}, {xevent_depot}\n")
                    
                    if xdepot_home == 1:
                        file.write(f"x_-1_-2_{d}_{w}, {xdepot_home}\n")

                    if xevent_home == 1:
                        if event_2 is not None:
                            file.write(f"x_{event_2}_-2_{d}_{w}, {xevent_home}\n")
                        else:
                            file.write(f"x_{event_1}_-2_{d}_{w}, {xevent_home}\n")

            
    # print("No feasible solution found within 20 minutes.")


gurobi_solve()


