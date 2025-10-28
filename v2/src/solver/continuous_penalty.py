# src/solver/continuous.py

# import pandas as pd
from xml.parsers.expat import model
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
# from io.data_loader import load_problem_data
from ..models.problem_data import ProblemData
# from .helpers import get_time
import pickle




def continuous_algorithm_penalty (data: ProblemData, work_limit, seed_number, multiple_tw=None, event_limit=None, pruning=False, min_hour=None, max_hour=None):
    """
    Continuous algorithm for the MVT scheduling problem.
    Parameters:
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    max_event (int): The maximum number of events to arrange in a day.
    max_hour (dict): A dictionary containing the maximum working hours for each nurse.
    """

    C_event = data.C_event
    C_home = data.C_home
    C_depot_e = data.C_depot_e
    C_depot_h = data.C_depot_h
    C_depot = np.concatenate([C_depot_e, C_depot_h])
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
        # for each event, randomly select a [0,0] time window and replace it with [30, 600 - C_dur[i]]
        for i in range(m):
            zero_days = [d for d in range(day) if np.all(time_window[i, d] == 0)]
            
            if zero_days:
                chosen_day = np.random.choice(zero_days)
                time_window[i, chosen_day] = [30, 600 - C_dur[i]]
    
    # Decision variables

    # x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
    # i, j == m for home, i, j == m+1 for depot_am, i, j == m+2 for depot_pm
    x = model.addVars(m+3, m+3, day, n, vtype=GRB.BINARY, name="x") 

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = model.addVars(m, day, vtype=GRB.BINARY, name="s")

    # t_id time when event i starts on day d
    t = model.addVars(m, day, vtype=GRB.INTEGER, name="t")

    # alpha_idw = 1 if nurse w is the pick-up leader for event i on day d, 0 otherwise
    alpha = model.addVars(m, day, n, vtype=GRB.BINARY, name="alpha")

    # # beta_idw = 1 if nurse w is the drop-off leader for event i on day d, 0 otherwise
    beta = model.addVars(m, day, n, vtype=GRB.BINARY, name="beta")

    # Objective function
    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(day) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[w,i] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(day)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+2, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[m+w] * gp.quicksum((x[m+2, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )

    # penalty weight (tune this)
    lam = 10  # make sure it's large relative to travel/time costs

    # violation variables (nonnegative)
    u_pick = model.addVars(day, n, lb=0.0, vtype=GRB.CONTINUOUS, name="u_pick")
    u_drop = model.addVars(day, n, lb=0.0, vtype=GRB.CONTINUOUS, name="u_drop")

    # replace hard constraints with hinge-violation definitions
    model.addConstrs(
        (u_pick[d, w] >= gp.quicksum(alpha[j, d, w] for j in range(m)) - 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_soft"
    )

    model.addConstrs(
        (u_drop[d, w] >= gp.quicksum(beta[j, d, w] for j in range(m)) - 5 * x[m+2, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_soft"
    )

    # add penalty to objective
    penalty = lam * (gp.quicksum(u_pick[d, w] for d in range(day) for w in range(n)) +
                    gp.quicksum(u_drop[d, w] for d in range(day) for w in range(n)))

    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    model.setObjective(objective + penalty, GRB.MINIMIZE)

    # model.setObjective(objective, GRB.MINIMIZE)

    # Pruning
    if pruning >= 1:
        for d in range(day):
            for w in range(n):
                for i in range(m+2):
                    model.addConstr(x[i,i,d,w] == 0, name=f"no_self_loop_i{i}_d{d}_w{w}")
                for i in range(m):
                    model.addConstr(x[i,m+1,d,w] == 0, name=f"no_event_to_depotam_i{i}_d{d}_w{w}")
                    model.addConstr(x[m+2,i,d,w] == 0, name=f"no_depotpm_to_event_i{i}_d{d}_w{w}")
                model.addConstr(x[m+1,m,d,w] == 0, name=f"no_depotam_to_home_d{d}_w{w}")
                model.addConstr(x[m,m+2,d,w] == 0, name=f"no_home_to_depotpm_d{d}_w{w}")

    # elif pruning >= 2:
        # # Each event is scheduled exactly once during its feasible time window
        # # pruning: some events cannot be linked due to time infeaasibility
        # for i in range(m):
        #     for d in range(day):
        #         # set s[i][d], t[i][d] = 0 if time window is [0, 0]
        #         # no flow for events with time window [0, 0]
        #         if time_window[i][d][1] == 0:
        #             model.addConstr(s[i,d] == 0)
        #             model.addConstr(t[i,d] == 0)
        #             model.addConstrs(x[i,j,d,w] == 0 for j in range(m+2) for w in range(n))
        #             model.addConstrs(x[j,i,d,w] == 0 for j in range(m+2) for w in range(n))

    # Constraints
    # Each event happens on one day
    for i in range(m):
        model.addConstr(gp.quicksum(s[i,d] for d in range(day)) == 1, name=f"event_once_i{i}")

    # Each event is scheduled exactly once during its feasible time window
    for i in range(m):
        model.addConstr(gp.quicksum(t[i,d] for d in range(day)) >= 1, name=f"event_time_once_i{i}")
    for d in range(day):
        for i in range(m):
            model.addConstr(t[i,d] >= time_window[i][d][0] * s[i,d], name=f"tw_lb_i{i}_d{d}")
            model.addConstr(t[i,d] <= time_window[i][d][1] * s[i,d], name=f"tw_ub_i{i}_d{d}")
    for i in range(m):
        for d in range(day):
            for w in range(n):
                model.addConstr(sum(x[i,j,d,w] for j in range(m+3)) <= s[i,d], name=f"event_flow_i{i}_d{d}_w{w}")

    # Time feasibility for consecutive events
    # for i in range(m):
    #     for j in range(m):
    #         if i != j:
    #             model.addConstrs(t[j,d] >= t[i,d] + C_dur[i] + C_event[i,j] - M * (1 - x[i,j,d,w]) for w in range(n) for d in range(day))
    for i in range(m):
        for j in range(m):
            for d in range(day):
                for w in range(n):
                    model.addGenConstrIndicator(
                        x[i, j, d, w],
                        True,
                        t[j, d] >= t[i, d] + C_dur[i] + C_event[i, j],
                        name=f"time_ind_i{i}_j{j}_d{d}_w{w}"
                    )

    # Each event happens during its feasible time window
    # model.addConstrs(t[i,d] <= time_window[i][d][1] for i in range(m) for d in range(day))
    # model.addConstrs(gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m))
 
    # Minimum working hours
    if min_hour is not None:
        for w in range(n):
            model.addConstr(
                gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day)) for j in range(m)) >= min_hour[w] * 60,
                name=f"min_working_hours_w{w}"
            )

    # Maximum working hours
    if max_hour is not None:
        for w in range(n):
            model.addConstr(
                gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day)) for j in range(m)) <= max_hour[w] * 60,
                name=f"max_working_hours_w{w}"
            )
    
    # staffing
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day) for w in range(nr)) >= min_nurse[j][0],
            name=f"min_RN_j{j}"
        )
        model.addConstr(
            gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day) for w in range(nr, nr+nl)) >= min_nurse[j][1],
            name=f"min_LVN_j{j}"
        )

    # maximum number of events per day
    if event_limit is not None:
        for d in range(day):
            for w in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j, d, w] for i in range(m) for j in range(m)) <= event_limit-1,
                    name=f"max_events_per_day_d{d}_w{w}"
                )
  
    # network flow
    # event inflow = outflow
    for j in range(m):
        for d in range(day):
            for w in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j, d, w] for i in range(m+3)) == gp.quicksum(x[j, i, d, w] for i in range(m+3)),
                    name=f"event_network_flow_j{j}_d{d}_w{w}"
                )

    # outflow from home is at most 1
    model.addConstrs(
        (gp.quicksum(x[m, i, d, w] for i in range(m+3)) <= 1 for d in range(day) for w in range(n)),
        name="home_outflow"
    )

    # depot inflow = outflow
    model.addConstrs(
        (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="morning_depot_flow"
    )

    model.addConstrs(
        (x[m+2, m, d, w] == gp.quicksum(x[j, m+2, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="evening_depot_flow"
    )
    
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
        (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="pick_up_leader_event"
    )

    model.addConstrs(
        (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="drop_off_leader_event"
    )


    # # pick up leader goes from home to depot to their first event
    # # drop off leader goes from their last event to depot to home
    # model.addConstrs(
    #     (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
    #     name="pick_up_leader_home_depot"
    # )

    # model.addConstrs(
    #     (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+2, m, d, w] for d in range(day) for w in range(n)),
    #     name="drop_off_leader_depot_home"
    # )

    # Set the time limit to 20 minutes (1200 seconds)
    # model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.WorkLimit, work_limit)

    model.Params.Threads = 8
    model.setParam('LogFile', 'gurobi_output_cah_sim.txt')
    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3

    start_time = time.time()
    model.optimize()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)  # in seconds, rounded to 2 decimals

    summary = {}
    work_time_by_nurse = {}

    if model.SolCount > 0:
        objective = model.ObjVal
        print('The optimal objective is %g' % model.objVal)
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Best bound: {model.ObjBound}")
        print(f"Gap: {model.MIPGap}")

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
        summary["active_t"] = {
            (i, d): t[i, d].x
            for (i, d) in t.keys()
            if t[i, d].x > 0
        }

        # # 4. Active alpha[i,d,w]
        # summary["active_alpha"] = [
        #     (i, d, w)
        #     for (i, d, w) in alpha.keys()
        #     if alpha[i, d, w].x > 0
        # ]

        # # 5. Active beta[i,d,w]
        # summary["active_beta"] = [
        #     (i, d, w)
        #     for (i, d, w) in beta.keys()
        #     if beta[i, d, w].x > 0
        # ]

        # 6. Save objective value, runtime, gap, etc.
        summary["objective_value"] = model.ObjVal
        summary["runtime_sec"] = model.Runtime
        summary["gap"] = model.MIPGap

        # also return a dictionary with the working hours of each nurse
        work_time_by_nurse = {}
        for w in range(n):
            total_minutes = 0
            for j in range(m):
                for d in range(day):
                    for i in range(m + 3):
                        if x[i, j, d, w].x > 0:
                            total_minutes += C_dur[j]
                            break
            work_time_by_nurse[w] = total_minutes / 60.0  # convert to hours
        
        summary["work_time_by_nurse"] = work_time_by_nurse

    else:
        objective = None
        print(f"No feasible solution found within {work_limit} work units.")
        # Optionally, set a flag or default value:
        summary["objective_value"] = None
        summary["work_time_by_nurse"] = {}

    return summary


def continuous_algorithm_adaptive_penalty (data: ProblemData, work_limit, seed_number, multiple_tw=None, event_limit=None, pruning=False, min_hour=None, max_hour=None):
    """
    Continuous algorithm for the MVT scheduling problem.
    Parameters:
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    max_event (int): The maximum number of events to arrange in a day.
    max_hour (dict): A dictionary containing the maximum working hours for each nurse.
    """

    C_event = data.C_event
    C_home = data.C_home
    C_depot_e = data.C_depot_e
    C_depot_h = data.C_depot_h
    C_depot = np.concatenate([C_depot_e, C_depot_h])
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
        # for each event, randomly select a [0,0] time window and replace it with [30, 600 - C_dur[i]]
        for i in range(m):
            zero_days = [d for d in range(day) if np.all(time_window[i, d] == 0)]
            
            if zero_days:
                chosen_day = np.random.choice(zero_days)
                time_window[i, chosen_day] = [30, 600 - C_dur[i]]
    
    # Decision variables

    # x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
    # i, j == m for home, i, j == m+1 for depot_am, i, j == m+2 for depot_pm
    x = model.addVars(m+3, m+3, day, n, vtype=GRB.BINARY, name="x") 

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = model.addVars(m, day, vtype=GRB.BINARY, name="s")

    # t_id time when event i starts on day d
    t = model.addVars(m, day, vtype=GRB.INTEGER, name="t")

    # alpha_idw = 1 if nurse w is the pick-up leader for event i on day d, 0 otherwise
    alpha = model.addVars(m, day, n, vtype=GRB.BINARY, name="alpha")

    # # beta_idw = 1 if nurse w is the drop-off leader for event i on day d, 0 otherwise
    beta = model.addVars(m, day, n, vtype=GRB.BINARY, name="beta")

    # violation variables (nonnegative)
    u_pick = model.addVars(day, n, lb=0.0, vtype=GRB.CONTINUOUS, name="u_pick")
    u_drop = model.addVars(day, n, lb=0.0, vtype=GRB.CONTINUOUS, name="u_drop")



    # Objective function
    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(day) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[w,i] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(day)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+2, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[m+w] * gp.quicksum((x[m+2, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )

    # penalty weight (tune this)
    # lam = 1.0  # initial penalty weight
    # lam_pick = {(d, w): lam for d in range(day) for w in range(n)}
    # lam_drop = {(d, w): lam for d in range(day) for w in range(n)}
    # initialize lam based on C_depot_h, uniquely for each nurse
    lams = (C_depot_h / 10.0) ** 3  # scale down to get reasonable penalty weights
    lam_pick = {(d, w): lams[w] for d in range(day) for w in range(n)}
    lam_drop = {(d, w): lams[w] for d in range(day) for w in range(n)}


    # replace hard constraints with hinge-violation definitions
    model.addConstrs(
        (u_pick[d, w] >= gp.quicksum(alpha[j, d, w] for j in range(m)) - 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_soft"
    )

    model.addConstrs(
        (u_drop[d, w] >= gp.quicksum(beta[j, d, w] for j in range(m)) - 5 * x[m+2, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_soft"
    )

    # add penalty to objective
    penalty = gp.quicksum(
        lam_pick[d, w] * u_pick[d, w] + lam_drop[d, w] * u_drop[d, w]
        for d in range(day) for w in range(n)
    )

    main_objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    # model.setObjective(objective + penalty, GRB.MINIMIZE)



    # model.setObjective(objective, GRB.MINIMIZE)

    # Pruning
    if pruning >= 1:
        for d in range(day):
            for w in range(n):
                for i in range(m+2):
                    model.addConstr(x[i,i,d,w] == 0, name=f"no_self_loop_i{i}_d{d}_w{w}")
                for i in range(m):
                    model.addConstr(x[i,m+1,d,w] == 0, name=f"no_event_to_depotam_i{i}_d{d}_w{w}")
                    model.addConstr(x[m+2,i,d,w] == 0, name=f"no_depotpm_to_event_i{i}_d{d}_w{w}")
                model.addConstr(x[m+1,m,d,w] == 0, name=f"no_depotam_to_home_d{d}_w{w}")
                model.addConstr(x[m,m+2,d,w] == 0, name=f"no_home_to_depotpm_d{d}_w{w}")

    # elif pruning >= 2:
        # # Each event is scheduled exactly once during its feasible time window
        # # pruning: some events cannot be linked due to time infeaasibility
        # for i in range(m):
        #     for d in range(day):
        #         # set s[i][d], t[i][d] = 0 if time window is [0, 0]
        #         # no flow for events with time window [0, 0]
        #         if time_window[i][d][1] == 0:
        #             model.addConstr(s[i,d] == 0)
        #             model.addConstr(t[i,d] == 0)
        #             model.addConstrs(x[i,j,d,w] == 0 for j in range(m+2) for w in range(n))
        #             model.addConstrs(x[j,i,d,w] == 0 for j in range(m+2) for w in range(n))

    # Constraints
    # Each event happens on one day
    for i in range(m):
        model.addConstr(gp.quicksum(s[i,d] for d in range(day)) == 1, name=f"event_once_i{i}")

    # Each event is scheduled exactly once during its feasible time window
    for i in range(m):
        model.addConstr(gp.quicksum(t[i,d] for d in range(day)) >= 1, name=f"event_time_once_i{i}")
    for d in range(day):
        for i in range(m):
            model.addConstr(t[i,d] >= time_window[i][d][0] * s[i,d], name=f"tw_lb_i{i}_d{d}")
            model.addConstr(t[i,d] <= time_window[i][d][1] * s[i,d], name=f"tw_ub_i{i}_d{d}")
    for i in range(m):
        for d in range(day):
            for w in range(n):
                model.addConstr(sum(x[i,j,d,w] for j in range(m+3)) <= s[i,d], name=f"event_flow_i{i}_d{d}_w{w}")

    # Time feasibility for consecutive events
    # for i in range(m):
    #     for j in range(m):
    #         if i != j:
    #             model.addConstrs(t[j,d] >= t[i,d] + C_dur[i] + C_event[i,j] - M * (1 - x[i,j,d,w]) for w in range(n) for d in range(day))
    for i in range(m):
        for j in range(m):
            for d in range(day):
                for w in range(n):
                    model.addGenConstrIndicator(
                        x[i, j, d, w],
                        True,
                        t[j, d] >= t[i, d] + C_dur[i] + C_event[i, j],
                        name=f"time_ind_i{i}_j{j}_d{d}_w{w}"
                    )

    # Each event happens during its feasible time window
    # model.addConstrs(t[i,d] <= time_window[i][d][1] for i in range(m) for d in range(day))
    # model.addConstrs(gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m))
 
    # Minimum working hours
    if min_hour is not None:
        for w in range(n):
            model.addConstr(
                gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day)) for j in range(m)) >= min_hour[w] * 60,
                name=f"min_working_hours_w{w}"
            )

    # Maximum working hours
    if max_hour is not None:
        for w in range(n):
            model.addConstr(
                gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day)) for j in range(m)) <= max_hour[w] * 60,
                name=f"max_working_hours_w{w}"
            )
    
    # staffing
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day) for w in range(nr)) >= min_nurse[j][0],
            name=f"min_RN_j{j}"
        )
        model.addConstr(
            gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(day) for w in range(nr, nr+nl)) >= min_nurse[j][1],
            name=f"min_LVN_j{j}"
        )

    # maximum number of events per day
    if event_limit is not None:
        for d in range(day):
            for w in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j, d, w] for i in range(m) for j in range(m)) <= event_limit-1,
                    name=f"max_events_per_day_d{d}_w{w}"
                )
  
    # network flow
    # event inflow = outflow
    for j in range(m):
        for d in range(day):
            for w in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j, d, w] for i in range(m+3)) == gp.quicksum(x[j, i, d, w] for i in range(m+3)),
                    name=f"event_network_flow_j{j}_d{d}_w{w}"
                )

    # outflow from home is at most 1
    model.addConstrs(
        (gp.quicksum(x[m, i, d, w] for i in range(m+3)) <= 1 for d in range(day) for w in range(n)),
        name="home_outflow"
    )

    # depot inflow = outflow
    model.addConstrs(
        (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="morning_depot_flow"
    )

    model.addConstrs(
        (x[m+2, m, d, w] == gp.quicksum(x[j, m+2, d, w] for j in range(m)) for d in range(day) for w in range(n)),
        name="evening_depot_flow"
    )
    
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
        (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="pick_up_leader_event"
    )

    model.addConstrs(
        (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(day) for w in range(n)),
        name="drop_off_leader_event"
    )


    # # pick up leader goes from home to depot to their first event
    # # drop off leader goes from their last event to depot to home
    # model.addConstrs(
    #     (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
    #     name="pick_up_leader_home_depot"
    # )

    # model.addConstrs(
    #     (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+2, m, d, w] for d in range(day) for w in range(n)),
    #     name="drop_off_leader_depot_home"
    # )
    max_iter = 5
    tol = 1e-6
    factor = 10

    model._u_pick = u_pick
    model._u_drop = u_drop
    model._lam_pick = lam_pick
    model._lam_drop = lam_drop
    model._main_obj = main_objective

    model._day = day
    model._n = n
    model._factor = factor
    model._tol = tol

    model._best_main_obj = float('inf')
    model._improved = False

    model.Params.Threads = 8
    model.Params.OutputFlag = 1
    model.Params.WorkLimit = work_limit  # per iteration
    # model.Params.LogFile = "gurobi_output_cah_sim.txt"

    # === 4. Initialize bookkeeping ===
    last_main_obj = float("inf")
    # iteration = 0

    def lagrangian_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            print(f"  -> New incumbent: total objective = {obj:.2f}")


    # === 6. Adaptive loop ===
    for it in range(1, max_iter + 1):
        print(f"\n=== Iteration {it} / {max_iter} ===")

        # Build penalty term dynamically
        penalty = gp.quicksum(
            lam_pick[d, w] * u_pick[d, w] +
            lam_drop[d, w] * u_drop[d, w]
            for d in range(day) for w in range(n)
        )
        model.setObjective(main_objective + penalty, gp.GRB.MINIMIZE)

        # Reset before each iteration so WorkLimit applies freshly
        model.reset()
        model.Params.WorkLimit = work_limit

        # --- Solve ---
        model.optimize(lagrangian_callback)

        # --- Check solver status ---
        if model.Status not in [gp.GRB.Status.OPTIMAL,
                                gp.GRB.Status.INTERRUPTED,
                                gp.GRB.Status.WORK_LIMIT]:
            print(f"⚠️ Iteration {it}: optimization ended with status {model.Status}.")
            break

        # --- Extract results ---
        u_pick_vals = model.getAttr('X', u_pick)
        u_drop_vals = model.getAttr('X', u_drop)

        penalty_val = sum(
            lam_pick[d, w]*u_pick_vals[d, w] +
            lam_drop[d, w]*u_drop_vals[d, w]
            for d in range(day) for w in range(n)
        )
        total_obj = model.ObjVal
        main_obj = total_obj - penalty_val

        print(f"  main_obj={main_obj:.2f}, penalty={penalty_val:.2f}, total={total_obj:.2f}")

        # --- Update lambdas if improvement or work limit reached ---
        update_condition = False

        if main_obj < last_main_obj - 1e-4:
            print("✅ Main objective improved — updating λ and proceeding.")
            update_condition = True
            last_main_obj = main_obj

        elif model.Status == gp.GRB.Status.WORK_LIMIT:
            print(f"⚠️ Work limit ({work_limit}) reached — updating λ and proceeding.")
            update_condition = True

        if update_condition:
            num_viol = 0
            for (d, w), val in u_pick_vals.items():
                if val > tol:
                    lam_pick[d, w] = min(lam_pick[d, w] * factor, 1e4)
                    num_viol += 1
            for (d, w), val in u_drop_vals.items():
                if val > tol:
                    lam_drop[d, w] = min(lam_drop[d, w] * factor, 1e4)
                    num_viol += 1
            # print list of updated lambdas
            print("  Updated λ values:")
            for (d, w), val in lam_pick.items():
                print(f"  λ[{d},{w}] = {val:.2f}")
            for (d, w), val in lam_drop.items():
                print(f"  λ[{d},{w}] = {val:.2f}")
            print(f"⚠️ Number of violations: {num_viol}")

        else:
            print("ℹ️ Neither improvement nor work limit — continuing current setup.")

    print("\n✅ Finished adaptive penalty optimization.")

    # Set the time limit to 20 minutes (1200 seconds)
    # model.setParam(GRB.Param.TimeLimit, time_limit)

    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3

    # start_time = time.time()
    # model.optimize(lagrangian_callback)
    # end_time = time.time()

    # elapsed_time = round(end_time - start_time, 2)  # in seconds, rounded to 2 decimals

    summary = {}
    work_time_by_nurse = {}

    if model.SolCount > 0:
        # objective = model.ObjVal
        print('The optimal objective is %g' % model.objVal)
        # print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Best bound: {model.ObjBound}")
        print(f"Gap: {model.MIPGap}")

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
        summary["active_t"] = {
            (i, d): t[i, d].x
            for (i, d) in t.keys()
            if t[i, d].x > 0
        }

        # # 4. Active alpha[i,d,w]
        # summary["active_alpha"] = [
        #     (i, d, w)
        #     for (i, d, w) in alpha.keys()
        #     if alpha[i, d, w].x > 0
        # ]

        # # 5. Active beta[i,d,w]
        # summary["active_beta"] = [
        #     (i, d, w)
        #     for (i, d, w) in beta.keys()
        #     if beta[i, d, w].x > 0
        # ]

        # 6. Save objective value, runtime, gap, etc.
        summary["objective_value"] = model.ObjVal
        summary["runtime_sec"] = model.Runtime
        summary["gap"] = model.MIPGap

        # also return a dictionary with the working hours of each nurse
        work_time_by_nurse = {}
        for w in range(n):
            total_minutes = 0
            for j in range(m):
                for d in range(day):
                    for i in range(m + 3):
                        if x[i, j, d, w].x > 0:
                            total_minutes += C_dur[j]
                            break
            work_time_by_nurse[w] = total_minutes / 60.0  # convert to hours
        
        summary["work_time_by_nurse"] = work_time_by_nurse

    else:
        objective = None
        print(f"No feasible solution found within {work_limit} work units.")
        # Optionally, set a flag or default value:
        summary["objective_value"] = None
        summary["work_time_by_nurse"] = {}

    return summary


