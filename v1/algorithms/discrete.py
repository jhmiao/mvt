import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from data_loader import ProblemData
from .helpers import get_timeslot
import pickle

def discrete_algorithm(data: ProblemData, time_limit, seed_number, multiple_tw=None, full_day=300, event_limit=None, pruning=2, min_hour=10):
    """
    Discrete algorithm for the MVT scheduling problem.
    Parameters:
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    max_event (int): The maximum number of events to arrange in a day.
    """

    C_event = data.C_event
    C_home = data.C_home
    C_depot = data.C_depot
    C_dur = data.C_dur
    time_window = data.time_window
    min_nurse = data.min_nurse
    nr = data.nr
    nl = data.nl
    n = data.n
    m = data.m
    day = data.day
    
    model = gp.Model("MVT_scheduling_continuous")

    np.random.seed(seed_number)

    if multiple_tw is not None:
        if multiple_tw > 4:
            raise ValueError("multiple_tw cannot be greater than 4")

    for i in range(m):
        # Find all days with [0, 0] time window
        zero_days = [d for d in range(day) if np.all(time_window[i, d] == 0)]
        
        # If enough zero-days available, randomly select multiple_tw of them
        if len(zero_days) >= multiple_tw:
            chosen_days = np.random.choice(zero_days, size=multiple_tw, replace=False)
        else:
            # Pick as many as available
            chosen_days = zero_days

        # Replace with [1, 1]
        for d in chosen_days:
            time_window[i, d] = [1, 1]

    # Decision variables

    # x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
    # i, j == -1 for depot, i, j == -2 for home
    x = model.addVars(m+2, m+2, day, n, vtype=GRB.BINARY, name="x") 

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = model.addVars(m, day, vtype=GRB.BINARY, name="s")

    # t_id assigned time slot event i on day d
    # t_id = [0, 0] if not scheduled
    # t_id = [1, 0] if scheduled as a morning event
    # t_id = [0, 1] if scheduled as a afternoon event
    # t_id = [1, 1] if scheduled as a full day event
    t = model.addVars(m, day, 2, vtype=GRB.BINARY, name="t")

    # alpha_idw = 1 if nurse w is the pick-up leader for event i on day d, 0 otherwise
    alpha = model.addVars(m, day, n, vtype=GRB.BINARY, name="alpha")

    # # beta_idw = 1 if nurse w is the drop-off leader for event i on day d, 0 otherwise
    beta = model.addVars(m, day, n, vtype=GRB.BINARY, name="beta")


    # model.update()

    # Objective function

    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(day) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[i,w] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(day)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+1, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[w + m] * gp.quicksum((x[m+1, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )

    travel_cost = event_cost + home_cost + depot_event_cost + depot_home_cost

    objective = travel_cost

    model.setObjective(objective, GRB.MINIMIZE)

    # Pruning
    # fix all x[i][j][d][w] to 0 if 
    # 1) i == j 
    if pruning >= 1:
        # fix all x[i][j][d][w] to 0 if 1) i == j
        model.addConstrs(x[i,i,d,w] == 0 for d in range(day) for w in range(n) for i in range(m+2))


    # # 2) time window is infeasible: no overlapping days
    # for i in range(m):
    #     for d in range(day):
    #         # no between-event trips if it is a full-day event
    #         if time_window[i][d][0] == 1 and time_window[i][d][1] == 1 and C_dur[i] > full_day:
    #             model.addConstrs(
    #                 (x[i, j, d, w] == 0 for j in range(m) for w in range(n)),
    #                 name=f"R{i}_{d}_between_event_trip_infeasible1"
    #             )
    #             # model.addConstrs(
    #             #     (x[j, i, d, w] == 0 for j in range(m) for w in range(n)),
    #             #     name=f"R{i}_{d}_between_event_trip_infeasible2"
    #             # )
            
    #         # no trip to/from the event if it is not scheduled
    #         elif time_window[i][d][0] == 0 and time_window[i][d][1] == 0:
    #             model.addConstr(
    #                 (s[i, d] == 0),
    #                 name=f"R{i}_{d}_schedule_infeasible"
    #             )
    #             model.addConstrs(x[i, j, d, w] == 0 for j in range(m+2) for w in range(n))
    #             # model.addConstrs(x[j, i, d, w] == 0 for j in range(m+2) for w in range(n))
    #             # no leader for the event on that day
    #             model.addConstrs(alpha[i, d, w] == 0 for w in range(n))
    #             model.addConstrs(beta[i, d, w] == 0 for w in range(n))

    #         # no trips to events if it is an evening event
    #         elif time_window[i][d][0] == 0 and time_window[i][d][1] == 1:
    #             model.addConstrs(
    #                 (x[i, j, d, w] == 0 for j in range(m) for w in range(n)),
    #                 name=f"R{i}_{d}_between_event_trip_infeasible"
    #             )

    #         # no trips from events if it is a morning event
    #         else:
    #             model.addConstrs(
    #                 (x[j,i, d, w] == 0 for j in range(m) for w in range(n)),
    #                 name=f"R{i}_{d}_between_event_trip_infeasible"
    #             )

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
        (gp.quicksum(s[i, d] for d in range(day)) == 1 for i in range(m)),name="schedule_one_day"
    )
    # inflow only exists when event is scheduled
    model.addConstrs(
        (gp.quicksum(x[i,j,d,w] for i in range(m+2)) <= s[j,d] for j in range(m) for d in range(day) for w in range(n)), 
        name="outflow_on_event_day"
    )

    # Maximum 2 events per day for each nurse
    model.addConstrs((gp.quicksum(x[i,j,d,w] for i in range(m) for j in range(m)) <= 1 for d in range(day) for w in range(n)), name=f"max_2_events_per_day") 

    # time window
    model.addConstrs(
        (t[i, d, 0] <= time_window[i][d][0] * s[i,d] for i in range(m) for d in range(day)),
        name="time_window_am"
    )
    model.addConstrs(
        (t[i, d, 1] <= time_window[i][d][1] * s[i,d] for i in range(m) for d in range(day)),
        name="time_window_pm"
    )

    # event duration
    model.addConstrs(
        (gp.quicksum(t[i, d, 0] + t[i, d, 1] for d in range(day)) == np.ceil(C_dur[i]/full_day) for i in range(m)),
        name="event_duration"
    )

    # Time feasibility for consecutive events
    model.addConstrs(
        (t[i, d, 0] >= x[i, j, d, w] for i in range(m) for j in range(m) for d in range(day) for w in range(n)),
        name="time_feasibility_1"
    )
    model.addConstrs(
        (t[i, d, 1] <= 1-x[i, j, d, w] for i in range(m) for j in range(m) for d in range(day) for w in range(n)),
        name="time_feasibility_2"
    )
    model.addConstrs(
        (t[j, d, 0] <= 1-x[i, j, d, w] for i in range(m) for j in range(m) for d in range(day) for w in range(n)),
        name="time_feasibility_3"
    )

    model.addConstrs(
        (t[j, d, 1] >= x[i, j, d, w] for i in range(m) for j in range(m) for d in range(day) for w in range(n)),
        name="time_feasibility_4"
    )

    # minimum working hours (10 hours per week)
    model.addConstrs(
        (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) >= min_hour * 60 for w in range(n)),
        name="min_working_hours"
    )

    # staffing
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day) for w in range(nr)) >= min_nurse[j][0] for j in range(m)),
        name="min_RN"
    )
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day) for w in range(nr, nr+nl)) >= min_nurse[j][1] for j in range(m)),
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
        (gp.quicksum(alpha[j, d, w] for w in range(n) for d in range(day)) == 1 for j in range(m)),
        name="pick_up_leader"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for w in range(n) for d in range(day)) == 1 for j in range(m)),
        name="drop_off_leader"
    )

    # team leader goes to the event
    model.addConstrs(
        (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+2)) for j in range(m) for d in range(day) for w in range(n)),
        name="pick_up_leader_event"
    )

    model.addConstrs(
        (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+2)) for j in range(m) for d in range(day) for w in range(n)),
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
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 2 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 2 * x[m+1, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_depot_home"
    )

    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2)) == gp.quicksum(x[j, i, d, w] for i in range(m+2)) for j in range(m) for d in range(day) for w in range(n)),
        name="network_flow"
    )

    # outflow from home is at most 1
    model.addConstrs(
        (gp.quicksum(x[m, i, d, w] for i in range(m+2)) <= 1 for d in range(day) for w in range(n)),
        name="home_outflow"
    )

    # depot inflow = outflow
    model.addConstrs(
        (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="morning_depot_flow"
    )

    model.addConstrs(
        (x[m+1, m, d, w] == gp.quicksum(x[j, m+1, d, w] for j in range(m)) for d in range(day) for w in range(n)),
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
    model.setParam('LogFile', 'gurobi_output_da_multipletw.txt')
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
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Best bound: {model.ObjBound}")
        print(f"Gap: {model.MIPGap}")

        # # print the the working hours of each nurse
        # nurse_work_time = {}

        # for w in range(n):  # for each nurse
        #     total_minutes = 0
        #     for j in range(m):  # for each event
        #         for d in range(day):
        #             for i in range(m + 2):  # including home/depot/start/end points
        #                 if x[i, j, d, w].X > 0:
        #                     total_minutes += C_dur[j]
        #                     break  # no need to count more i→j arcs once one is found
        #     nurse_work_time[w] = total_minutes / 60.0  # convert to hours
        # print("Nurse work hours:")
        # for w, hours in nurse_work_time.items():
        #     print(f"Nurse {w + 1}: {hours:.2f} hours")

        file_path = f'/Users/jinghongmiao/Code/mvt-code/result-250501/da_{nr}_{nl}_{m}_tl{time_limit}_fd{full_day}_el{event_limit}_p{pruning}_mh{min_hour}_mt{multiple_tw}_seed{seed_number}.pkl'
        summary = {}

        # 1. Active x[i,j,d,w]
        summary["active_x"] = [
            (i, j, d, w)
            for (i, j, d, w) in x.keys()
            if x[i, j, d, w].x > 0
        ]

        # 2. Active s[i,d]
        summary["active_s"] = [
            (i, d)
            for (i, d) in s.keys()
            if s[i, d].x > 0
        ]

        # 3. Start times t[i,d] (only if event scheduled)
        summary["event_start_times"] = {
            (i, d): t[i, d, k].X
            for (i, d, k) in t.keys()
            if t[i, d, k].X > 0
        }

        # 4. Active alpha[i,d,w]
        summary["active_alpha"] = [
            (i, d, w)
            for (i, d, w) in alpha.keys()
            if alpha[i, d, w].x > 0
        ]

        # 5. Active beta[i,d,w]
        summary["active_beta"] = [
            (i, d, w)
            for (i, d, w) in beta.keys()
            if beta[i, d, w].x > 0
        ]

        # 6. Save objective value, runtime, gap, etc.
        summary["objective_value"] = model.ObjVal
        summary["runtime_sec"] = model.Runtime
        summary["gap"] = model.MIPGap

        # Save as pickle
        with open(file_path, "wb") as f:
            pickle.dump(summary, f)

        # file_path1 = f'/Users/jinghongmiao/Code/mvt-code/result-250429/da_EVENT_{nr}_{nl}_{m}_fd{full_day}_el{event_limit}_p{pruning}_mh{min_hour}_seed{seed_number}.txt'
        # file_path2 = f'/Users/jinghongmiao/Code/mvt-code/result-250429/da_NURSE_{nr}_{nl}_{m}_fd{full_day}_el{event_limit}_p{pruning}_mh{min_hour}_seed{seed_number}.txt'
        
        # with open(file_path1, "a") as file:
        #     file.write(f"Best feasible solution found within {elapsed_time} seconds:\n")
        #     file.write(f"Objective value: {model.ObjVal}\n")
        #     file.write(f"Best bound: {model.ObjBound}\n")
        #     file.write(f"Gap: {model.MIPGap}\n")

        #     for d in range(day):
        #         file.write(f"\n=====================\nDay {d+1} \n=====================\n")
        #         for i in range(m):
        #             if s[i,d].x == 1:
        #                 file.write(f"\nEvent {i+1}: {get_timeslot(t[i,d,0].x, t[i,d,1].x)} [{min_nurse[i][0]} RN, {min_nurse[i][1]} LVN] \n")
        #                 for w in range(n):
        #                     if alpha[i,d,w].x == 1 and beta[i,d,w].x == 1:
        #                         file.write(f"Nurse {w+1} (pick up & drop off)\n")
        #                     elif beta[i,d,w].x == 1:
        #                         file.write(f"Nurse {w+1} (drop off)\n")
        #                     elif alpha[i,d,w].x == 1:
        #                         file.write(f"Nurse {w+1} (pick up)\n")
        #                     elif sum(x[i,j,d,w].x for j in range(m+2)) >= 1:
        #                         file.write(f"Nurse {w+1}\n")
        
        # with open(file_path2, "a") as file:
        #     file.write(f"Best feasible solution found within {elapsed_time} seconds:\n")
        #     file.write(f"Objective value: {model.ObjVal}\n")
        #     file.write(f"Best bound: {model.ObjBound}\n")
        #     file.write(f"Gap: {model.MIPGap}\n")

        #     for w in range(n):
        #         total_worktime = 0
        #         total_traveltime = 0
        #         file.write(f"\n=====================\nNurse {w+1} \n=====================\n")
        #         for d in range(day):
        #             day_worktime = 0
        #             day_traveltime = 0
        #             file.write(f"\nDay {d+1}:\n")

        #             # # calculate the total travel time for the day
        #             # for i in range(m+2):
        #             #     for j in range(m+2):
        #             #         if x[i,j,d,w].x == 1:
        #             #             if i < m and j < m:
        #             #                 day_traveltime += C_event[i,j]
        #             #             elif i == m:
        #             #                 day_traveltime += C_home[j][w]
        #             #             elif j == m:
        #             #                 day_traveltime += C_home[i][w]
        #                         # day_traveltime += (
        #                         #     C_event[i,j] if i < m and j < m 
        #                         #     else C_depot[i] if i == m+1 
        #                         #     else C_home[i][w]
        #                         #     )
        #             # file.write(f"Total travel time: {day_traveltime} minutes\n")

        #             if x[m,m+1,d,w].x == 1:
        #                 file.write("Home -> Depot\n")
                    
        #             for i in range(m):
        #                 if sum(x[i,j,d,w].x for j in range(m+2)) >= 1:
        #                     # day_worktime += C_dur[i]

        #                     if t[i,d,0].x == 1:
        #                         # if alpha[i,d,w].x == 1 and beta[i,d,w].x == 1: 
        #                         #     file.write(f"Event {i+1} Morning (pick-up & drop-off)\n")
        #                         # elif alpha[i,d,w].x == 1:
        #                         #     file.write(f"Event {i+1} Morning (pick-up)\n")
        #                         # elif beta[i,d,w].x == 1:
        #                         #     file.write(f"Event {i+1} Morning (drop-off)\n")
        #                         # else:
        #                         #     file.write(f"Event {i+1} Morning\n")
        #                         file.write(f"Event {i+1} Morning\n")

        #                     if t[i,d,1].x == 1:
        #                         # if alpha[i,d,w].x == 1 and beta[i,d,w].x == 1: 
        #                         #     file.write(f"Event {i+1} Afternoon (pick-up & drop-off)\n")
        #                         # elif alpha[i,d,w].x == 1:
        #                         #     file.write(f"Event {i+1} Afternoon (pick-up)\n")
        #                         # elif beta[i,d,w].x == 1:
        #                         #     file.write(f"Event {i+1} Afternoon (drop-off)\n")
        #                         # else:
        #                         #     file.write(f"Event {i+1} Afternoon\n")
        #                         file.write(f"Event {i+1} Afternoon\n")

        #             if x[m+1,m,d,w].x == 1:
        #                 file.write("Depot -> Home\n")

    else:
        print(f"No feasible solution found within {elapsed_time} seconds.")
    return elapsed_time, model.ObjVal, model.ObjBound, model.MIPGap




def discrete_algorithm_daily (data: ProblemData, time_limit, seed_number, full_day=300,event_limit=None, pruning=False, min_hour=10):
    """
    Discrete algorithm for the MVT scheduling problem broken down by day.
    Parameters:
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    """
    
    C_event = data.C_event
    C_home = data.C_home
    C_depot = data.C_depot
    C_dur = data.C_dur
    time_window = data.time_window
    min_nurse = data.min_nurse
    nr = data.nr
    nl = data.nl
    n = data.n
    m = data.m
    days = data.day

    for d in range(days):
        # filter data for day d
        # get the index of events scheduled on day d: where time_window[:, d, 1] > 0
        events_today = np.where(time_window[:, d, 1] > 0)[0]
        # filter C_event, C_home, C_depot, C_dur, time_window, min_nurse by events_today
        C_event_today = C_event[np.ix_(events_today, events_today)]
        C_home_today = C_home[events_today]
        C_depot_today = np.concatenate([C_depot[events_today], C_depot[m:]])
        C_dur_today = C_dur[events_today]
        time_window_today = time_window[events_today, d, :].reshape((len(events_today), 1, 2))
        min_nurse_today = min_nurse[events_today, :]

        # create a new ProblemData object for day d
        data_today = ProblemData(
            C_event=C_event_today,
            C_home=C_home_today,
            C_depot=C_depot_today,
            C_dur=C_dur_today,
            time_window=time_window_today,
            min_nurse=min_nurse_today,
            nr=nr,
            nl=nl,
            n=n,
            m=len(events_today),
            day=1  # only one day
        )

        print(f"\n\nRunning discrete algorithm for day {d+1} with {len(events_today)} events...")
        print(f"shape of C_event_today: {C_event_today.shape}, C_home_today: {C_home_today.shape}, C_depot_today: {C_depot_today.shape}, C_dur_today: {C_dur_today.shape}, time_window_today: {time_window_today.shape}, min_nurse_today: {min_nurse_today.shape}")
        # Call the discrete_algorithm function with the filtered data
        discrete_algorithm(data_today, time_limit/days, seed_number, full_day, event_limit, pruning, min_hour/days)