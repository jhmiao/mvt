#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

# helper functions
def get_time(index1, index2):
    if index1 == 0 and index2 == 0:
        return "Not scheduled"
    elif index1 == 1 and index2 == 0:
        return "Morning"
    elif index1 == 0 and index2 == 1:
        return "Afternoon"
    else:
        return "Full day"

# Read the Excel file
file_path = 'input_parameters_real_ampm-old.xlsx'

# Read the settings (time_limit, seed_number, nr, nl, m, block)
settings_df = pd.read_excel(file_path, sheet_name='Settings')
settings = settings_df.set_index('Parameter')['Value'].to_dict()

# time_limit = int(settings['time_limit'])
time_limit = 600
seed_number = int(settings['seed_number'])
nr = int(settings['nr'])
nl = int(settings['nl'])
n = nr + nl
m = int(settings['m'])
block = int(settings['block'])

# Read the C_event matrix
C_event_df = pd.read_excel(file_path, sheet_name='C_event')
C_event = C_event_df.values
# take the (61,61) part of C_event
C_event = C_event[:m,:m]

# Read the C_home matrix
C_home_df = pd.read_excel(file_path, sheet_name='C_home')
C_home = C_home_df.values
C_home = C_home[:m]

# Read the C_depot array
# event + home
C_depot_df = pd.read_excel(file_path, sheet_name='C_depot')
C_depot = C_depot_df['Depot_Cost'].values

# Read the C_dur array
C_dur_df = pd.read_excel(file_path, sheet_name='C_dur')
C_dur = C_dur_df['Duration'].values
# # subtract 25 from all the values in C_dur
# C_dur = C_dur - 25
full_day = 240

# Read the time_window matrix
time_window_df = pd.read_excel(file_path, sheet_name='Time_Window')
time_window_flat = time_window_df.values
time_window = time_window_flat.reshape((m, block, 2))
# # randomly replace some [0,0] time windows with [1,1] with p = 0.25
# np.random.seed(seed_number)
# for i in range(m):
#     for d in range(block):
#         if time_window[i][d][0] == 0 and time_window[i][d][1] == 0:
#             if np.random.rand() < 0.25:
#                 time_window[i][d][0] = 1
#                 time_window[i][d][1] = 1

# Read the minimum nurses required matrix
min_nurse_df = pd.read_excel(file_path, sheet_name='Min_Nurse')
min_nurse = min_nurse_df.values


def gurobi_solve():
    iterations = []
    objective_values = []

    model = gp.Model("team_scheduling")
    # M = 800 # A large constant

    # Decision variables

    # x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
    # i, j == -1 for depot, i, j == -2 for home
    x = model.addVars(m+2, m+2, block, n, vtype=GRB.BINARY, name="x") 

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = model.addVars(m, block, vtype=GRB.BINARY, name="s")

    # t_id assigned time slot event i on day d
    # t_id = [0, 0] if not scheduled
    # t_id = [1, 0] if scheduled as a morning event
    # t_id = [0, 1] if scheduled as a afternoon event
    # t_id = [1, 1] if scheduled as a full day event
    t = model.addVars(m, block, 2, vtype=GRB.BINARY, name="t")

    # alpha_idw = 1 if nurse w is the pick-up leader for event i on day d, 0 otherwise
    alpha = model.addVars(m, block, n, vtype=GRB.BINARY, name="alpha")

    # # beta_idw = 1 if nurse w is the drop-off leader for event i on day d, 0 otherwise
    beta = model.addVars(m, block, n, vtype=GRB.BINARY, name="beta")


    # model.update()

    # Objective function

    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(block) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[i,w] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(block)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+1, d, w] )for d in range(block) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[w + m] * gp.quicksum((x[m+1, m, d, w] + x[m, m+1, d, w]) for d in range(block)) for w in range(n)
    )

    travel_cost = event_cost + home_cost + depot_event_cost + depot_home_cost

    objective = travel_cost

    model.setObjective(objective, GRB.MINIMIZE)

    # # set all travels involving depot to 0
    # model.addConstrs((x[i, m+1, d, w] == 0 for i in range(m+1) for d in range(block) for w in range(n)), name=f"no_to_depot")
    # model.addConstrs((x[m+1, i, d, w] == 0 for i in range(m+1) for d in range(block) for w in range(n)), name=f"no_from_depot")

    # Pruning
    # fix all x[i][j][d][w] to 0 if 
    # 1) i == j 
    for i in range(m+2):
        for j in range(m+2):
            if i == j:
                model.addConstrs(
                    (x[i, j, d, w] == 0 for d in range(block) for w in range(n)),
                    name=f"R{i}_{j}_trip_infeasible"
                )

    # 2) time window is infeasible: no overlapping days
    for i in range(m):
        for d in range(block):
            # no between-event trips if it is a full-day event
            if time_window[i][d][0] == 1 and time_window[i][d][1] == 1 and C_dur[i] > full_day:
                model.addConstrs(
                    (x[i, j, d, w] == 0 for j in range(m) for w in range(n)),
                    name=f"R{i}_{d}_between_event_trip_infeasible1"
                )
                # model.addConstrs(
                #     (x[j, i, d, w] == 0 for j in range(m) for w in range(n)),
                #     name=f"R{i}_{d}_between_event_trip_infeasible2"
                # )
            
            # no trip to/from the event if it is not scheduled
            elif time_window[i][d][0] == 0 and time_window[i][d][1] == 0:
                model.addConstr(
                    (s[i, d] == 0),
                    name=f"R{i}_{d}_schedule_infeasible"
                )
                model.addConstrs(x[i, j, d, w] == 0 for j in range(m+2) for w in range(n))
                # model.addConstrs(x[j, i, d, w] == 0 for j in range(m+2) for w in range(n))
                # no leader for the event on that day
                model.addConstrs(alpha[i, d, w] == 0 for w in range(n))
                model.addConstrs(beta[i, d, w] == 0 for w in range(n))

            # no trips to events if it is an evening event
            elif time_window[i][d][0] == 0 and time_window[i][d][1] == 1:
                model.addConstrs(
                    (x[i, j, d, w] == 0 for j in range(m) for w in range(n)),
                    name=f"R{i}_{d}_between_event_trip_infeasible"
                )

            # no trips from events if it is a morning event
            else:
                model.addConstrs(
                    (x[j,i, d, w] == 0 for j in range(m) for w in range(n)),
                    name=f"R{i}_{d}_between_event_trip_infeasible"
                )

            # else:
            #     for j in range(m):
            #         if (i != j) and (time_window[i][d][0] == time_window[j][d][0]) and (time_window[i][d][1] == time_window[j][d][1]):
            #             model.addConstrs(
            #                 (x[i, j, d, w] == 0 for w in range(n)),
            #                 name=f"R{i}_{j}_{d}_day_trip_infeasible"
            #             )
        

    ## Constraints
    # Each event exactly once
    model.addConstrs(
        (gp.quicksum(s[i, d] for d in range(block)) == 1 for i in range(m)),name="schedule_one_day"
    )
    # inflow only exists when event is scheduled
    model.addConstrs(
        (gp.quicksum(x[i,j,d,w] for i in range(m+2)) <= s[j,d] for j in range(m) for d in range(block) for w in range(n)), 
        name="outflow_on_event_day"
    )

    # Maximum 2 events per day for each nurse
    model.addConstrs((gp.quicksum(x[i,j,d,w] for i in range(m) for j in range(m)) <= 1 for d in range(block) for w in range(n)), name=f"max_2_events_per_day") 

    # time window
    model.addConstrs(
        (t[i, d, 0] <= time_window[i][d][0] * s[i,d] for i in range(m) for d in range(block)),
        name="time_window_am"
    )
    model.addConstrs(
        (t[i, d, 1] <= time_window[i][d][1] * s[i,d] for i in range(m) for d in range(block)),
        name="time_window_pm"
    )

    # event duration
    model.addConstrs(
        (gp.quicksum(t[i, d, 0] + t[i, d, 1] for d in range(block)) == np.ceil(C_dur[i]/240) for i in range(m)),
        name="event_duration"
    )

    # Time feasibility for consecutive events
    model.addConstrs(
        (t[i, d, 0] >= x[i, j, d, w] for i in range(m) for j in range(m) for d in range(block) for w in range(n)),
        name="time_feasibility_1"
    )
    model.addConstrs(
        (t[i, d, 1] <= 1-x[i, j, d, w] for i in range(m) for j in range(m) for d in range(block) for w in range(n)),
        name="time_feasibility_2"
    )
    model.addConstrs(
        (t[j, d, 0] <= 1-x[i, j, d, w] for i in range(m) for j in range(m) for d in range(block) for w in range(n)),
        name="time_feasibility_3"
    )

    model.addConstrs(
        (t[j, d, 1] >= x[i, j, d, w] for i in range(m) for j in range(m) for d in range(block) for w in range(n)),
        name="time_feasibility_4"
    )

    # minimum working hours (10 hours per week)
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

    # do not overstaff
    # model.addConstrs(
    #     (gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(block) for w in range(nr)) <= 2*min_nurse[j][0] for j in range(m)),
    #     name="min_RN"
    # )
    # model.addConstrs(
    #     (gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(block) for w in range(nr, nr+nl)) <= 2*min_nurse[j][1] for j in range(m)),
    #     name="min_LVN"
    # )

    # depot
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

    # # fairness: no one takes more than 4 trips to the depot 
    # model.addConstrs(
    #     (gp.quicksum((x[m, m+1, d, w] + x[m+1, m, d, w]) for d in range(block)) <= 4 for w in range(n)),
    #     name="fairness_depot"
    # )
    
    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 2 * x[m, m+1, d, w] for d in range(block) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 2 * x[m+1, m, d, w] for d in range(block) for w in range(n)),
        name="drop_off_leader_depot_home"
    )

    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2)) == gp.quicksum(x[j, i, d, w] for i in range(m+2)) for j in range(m) for d in range(block) for w in range(n)),
        name="network_flow"
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

    
    # # minimum 5 events per week for each nurse
    # for w in range(n):
    #     model.addConstr(sum(
    #         (sum(x[i][j][d][w] for i in range(m) for j in range(m)) + sum(xhome_event[j][d][w] + xdepot_event[j][d][w] for j in range(m))) for d in range(block)) >= 5, name=f"R{w}_min_5_events")

    # print("C_home has shape: ", C_home.shape)
        

    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    # model.setParam(GRB.Param.NodefileStart, 0.5)
    model.Params.Threads = 8  # Use only 8 CPU cores
    model.setParam('LogFile', 'gurobi_output_7p5.txt')
    # model.Params.Method = 4  # Use barrier method
    # model.Params.BarOrder = 0  # Use AMD ordering

    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3
    model.update()

    start_time = time.time()

    model.optimize()
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)  # in seconds, rounded to 2 decimals
    

    if model.SolCount > 0:
        print('The optimal objective is %g' % model.objVal)
        print('The gap is %g' % model.MIPGap)

        file_path1 = f'/Users/jinghongmiao/Code/mvt-code/result-250421/7p5_EVENT_{nr}_{nl}_{m}_seed{seed_number}.txt'
        file_path2 = f'/Users/jinghongmiao/Code/mvt-code/result-250421/7p5_NURSE_{nr}_{nl}_{m}_seed{seed_number}.txt'
        
        with open(file_path1, "w") as file:
            file.write(f"Best feasible solution found within {elapsed_time} seconds:\n")
            file.write(f"Objective value: {model.ObjVal}\n")
            file.write(f"Best bound: {model.ObjBound}\n")
            file.write(f"Gap: {model.MIPGap}\n")

            for d in range(block):
                file.write(f"\n=====================\nDay {d+1} \n=====================\n")
                for i in range(m):
                    if s[i,d].x == 1:
                        file.write(f"\nEvent {i+1}: {get_time(t[i,d,0].x, t[i,d,1].x)} [{min_nurse[i][0]} RN, {min_nurse[i][1]} LVN] \n")
                        for w in range(n):
                            if alpha[i,d,w].x == 1 and beta[i,d,w].x == 1:
                                file.write(f"Nurse {w+1} (pick up & drop off)\n")
                            elif beta[i,d,w].x == 1:
                                file.write(f"Nurse {w+1} (drop off)\n")
                            elif alpha[i,d,w].x == 1:
                                file.write(f"Nurse {w+1} (pick up)\n")
                            elif sum(x[i,j,d,w].x for j in range(m+2)) >= 1:
                                file.write(f"Nurse {w+1}\n")
        
        with open(file_path2, "w") as file:
            file.write(f"Best feasible solution found within {elapsed_time} seconds:\n")
            file.write(f"Objective value: {model.ObjVal}\n")
            file.write(f"Best bound: {model.ObjBound}\n")
            file.write(f"Gap: {model.MIPGap}\n")

            for w in range(n):
                total_worktime = 0
                total_traveltime = 0
                file.write(f"\n=====================\nNurse {w+1} \n=====================\n")
                for d in range(block):
                    day_worktime = 0
                    day_traveltime = 0
                    file.write(f"\nDay {d+1}:\n")

                    # # calculate the total travel time for the day
                    # for i in range(m+2):
                    #     for j in range(m+2):
                    #         if x[i,j,d,w].x == 1:
                    #             if i < m and j < m:
                    #                 day_traveltime += C_event[i,j]
                    #             elif i == m:
                    #                 day_traveltime += C_home[j][w]
                    #             elif j == m:
                    #                 day_traveltime += C_home[i][w]
                                # day_traveltime += (
                                #     C_event[i,j] if i < m and j < m 
                                #     else C_depot[i] if i == m+1 
                                #     else C_home[i][w]
                                #     )
                    # file.write(f"Total travel time: {day_traveltime} minutes\n")

                    if x[m,m+1,d,w].x == 1:
                        file.write("Home -> Depot\n")
                    
                    for i in range(m):
                        if sum(x[i,j,d,w].x for j in range(m+2)) >= 1:
                            # day_worktime += C_dur[i]

                            if t[i,d,0].x == 1:
                                # if alpha[i,d,w].x == 1 and beta[i,d,w].x == 1: 
                                #     file.write(f"Event {i+1} Morning (pick-up & drop-off)\n")
                                # elif alpha[i,d,w].x == 1:
                                #     file.write(f"Event {i+1} Morning (pick-up)\n")
                                # elif beta[i,d,w].x == 1:
                                #     file.write(f"Event {i+1} Morning (drop-off)\n")
                                # else:
                                #     file.write(f"Event {i+1} Morning\n")
                                file.write(f"Event {i+1} Morning\n")

                            if t[i,d,1].x == 1:
                                # if alpha[i,d,w].x == 1 and beta[i,d,w].x == 1: 
                                #     file.write(f"Event {i+1} Afternoon (pick-up & drop-off)\n")
                                # elif alpha[i,d,w].x == 1:
                                #     file.write(f"Event {i+1} Afternoon (pick-up)\n")
                                # elif beta[i,d,w].x == 1:
                                #     file.write(f"Event {i+1} Afternoon (drop-off)\n")
                                # else:
                                #     file.write(f"Event {i+1} Afternoon\n")
                                file.write(f"Event {i+1} Afternoon\n")

                    if x[m+1,m,d,w].x == 1:
                        file.write("Depot -> Home\n")
                #     print(f"Total worktime: {day_worktime} minutes")
                    # total_worktime += day_worktime
                    # total_traveltime += day_traveltime
                    # file.write(f"Day travel time: {day_traveltime} minutes\n")
                    # convert to hours

                # file.write(f"\nTotal travel time: {total_traveltime} minutes / {str(round(total_traveltime / 60, 2))} hours\n")
                # file.write(f"Total work time: {total_worktime} minutes / {total_worktime / 60} hours\n")
                # print(f"Total worktime: {total_worktime} minutes")
        # save to a csv file
        # create a data frame to store the travel time and work time for each nurse
        # column names: nurse, day, travel_time, work_time
        # rows = []
        # for w in range(n):
        #     for d in range(block):
        #         day_worktime = 0
        #         day_traveltime = 0
        #         for i in range(m+2):
        #             for j in range(m+2):
        #                 if x[i,j,d,w].x == 1:
        #                     if i < m and j < m:
        #                         day_traveltime += C_event[i,j]
        #                     elif i == m:
        #                         day_traveltime += C_home[j][w]
        #                     elif j == m:
        #                         day_traveltime += C_home[i][w]
        #         # calculate the total work time for the day
        #         for i in range(m):
        #             if sum(x[i,j,d,w].x for j in range(m+2)) >= 1:
        #                 day_worktime += C_dur[i]
        #         # store row in list
        #         rows.append({
        #             "Nurse": w + 1,
        #             "Day": d + 1,
        #             "Travel Time": day_traveltime,
        #             "Work Time": day_worktime
        #         })
        # result_df = pd.DataFrame(rows)
        # result_df.to_csv("ampm_travel_time_work_time.csv", index=False)

    else:
        print(f"No feasible solution found within {elapsed_time} seconds.")



gurobi_solve()



