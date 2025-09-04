import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from data_loader import ProblemData
import pickle


def continuous_algorithm (data: ProblemData, work_limit, seed_number, multiple_tw=None, event_limit=None, pruning=False, min_hour=None, max_hour=None):
    """
    Continuous algorithm for the MVT scheduling problem.
    Parameters:
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    max_event (int): The maximum number of events to arrange in a day.
    max_hour (dict): A dictionary containing the maximum working hours for each nurse.
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

    # --- CARC / Gurobi parameter setup ---
    import os
    try:
        # Treat work_limit as a solver wallclock limit (seconds)
        if 'work_limit' in locals() and locals().get('work_limit') is not None:
            model.Params.TimeLimit = float(locals().get('work_limit'))
        # Solve to optimal unless the time limit is hit
        model.Params.MIPGap = 0.0
        # Use as many CPUs as allocated by Slurm (fallback to all cores)
        threads_env = os.getenv('SLURM_CPUS_PER_TASK')
        if threads_env and threads_env.isdigit():
            model.Params.Threads = int(threads_env)
        else:
            import multiprocessing
            model.Params.Threads = max(1, multiprocessing.cpu_count())
        # Send Gurobi logs to a file if provided via env or local variable
        log_file = locals().get('log_file', None) or os.getenv('CARC_GRB_LOGFILE', None)
        if log_file:
            model.Params.LogFile = str(log_file)
    except Exception as _param_ex:
        # Non-fatal if params can't be set in some code paths
        pass
    # --- end param setup ---

    # --- CARC / Gurobi parameter setup ---
    import os
    try:
        # Use work_limit as a wallclock time limit (seconds).
        if work_limit is not None:
            model.Params.TimeLimit = float(work_limit)
        # Force solve-to-optimal unless time limit is hit.
        model.Params.MIPGap = 0.0
        # Use as many CPUs as Slurm grants or all local cores.
        threads_env = os.getenv("SLURM_CPUS_PER_TASK")
        if threads_env and threads_env.isdigit():
            model.Params.Threads = int(threads_env)
        else:
            import multiprocessing
            model.Params.Threads = max(1, multiprocessing.cpu_count())
        # Set Gurobi log file if specified via env var.
        log_file = os.getenv("CARC_GRB_LOGFILE", None)
        if log_file:
            model.Params.LogFile = str(log_file)
    except Exception as _param_ex:
        pass
    # --- end CARC param setup ---
    M = 600 # A large constant

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
    # i, j == -1 for depot, i, j == -2 for home
    x = model.addVars(m+2, m+2, day, n, vtype=GRB.BINARY, name="x") 

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
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+1, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[m+w] * gp.quicksum((x[m+1, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )

    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    model.setObjective(objective, GRB.MINIMIZE)

    # Pruning
    if pruning >= 1:
        # fix all x[i][j][d][w] to 0 if 1) i == j
        model.addConstrs(x[i,i,d,w] == 0 for d in range(day) for w in range(n) for i in range(m+2))
    
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
    model.addConstrs(gp.quicksum(s[i,d] for d in range(day)) == 1 for i in range(m))

    # Each event is scheduled exactly once during its feasible time window
    model.addConstrs(gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m))
    model.addConstrs(t[i,d] >= time_window[i][d][0] * s[i,d] for d in range(day) for i in range(m))
    model.addConstrs(t[i,d] <= time_window[i][d][1] * s[i,d] for d in range(day) for i in range(m))
    model.addConstrs(sum(x[i,j,d,w] for j in range(m+2)) <= s[i,d] for i in range(m) for d in range(day) for w in range(n))

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
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) >= min_hour[w] * 60 for w in range(n)),
            name="min_working_hours"
        )

    # Maximum working hours
    if max_hour is not None:
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) <= max_hour[w] * 60 for w in range(n)),
            name="max_working_hours"
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

    # maximum number of events per day
    if event_limit is not None:
        model.addConstrs(
            (gp.quicksum(x[i, j, d, w] for i in range(m) for j in range(m)) <= event_limit-1 for d in range(day) for w in range(n)),
            name="max_events_per_day"
        ) 
  
    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2)) == gp.quicksum(x[j, i, d, w] for i in range(m+2)) for j in range(m) for d in range(day) for w in range(n)),
        name="event_network_flow"
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


    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+1, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_depot_home"
    )


    # Set the time limit to 20 minutes (1200 seconds)
    # model.setParam(GRB.Param.TimeLimit, time_limit)
    # model.setParam(GRB.Param.WorkLimit, work_limit)

    # model.Params.Threads = 8
    # model.setParam('LogFile', 'gurobi_output_cah_sim.txt')
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

        # file_path = f'/Users/jinghongmiao/Code/mvt-code/result-250715/ca_{nr}_{nl}_{m}_wl{work_limit}_el{event_limit}_p{pruning}_mh{min_hour}_max25_seed{seed_number}.pkl'
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

        # also return a dictionary with the working hours of each nurse
        work_time_by_nurse = {}
        for w in range(n):
            total_minutes = 0
            for j in range(m):
                for d in range(day):
                    for i in range(m + 2):
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

def continuous_warm_start (data: ProblemData, work_limit, seed_number, multiple_tw=None, event_limit=None, pruning=False, min_hour=None, max_hour=None):
    """
    Continuous algorithm for the MVT scheduling problem.
    Parameters:
    data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
    max_event (int): The maximum number of events to arrange in a day.
    max_hour (dict): A dictionary containing the maximum working hours for each nurse.
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

    # --- CARC / Gurobi parameter setup ---
    import os
    try:
        # Treat work_limit as a solver wallclock limit (seconds)
        if 'work_limit' in locals() and locals().get('work_limit') is not None:
            model.Params.TimeLimit = float(locals().get('work_limit'))
        # Solve to optimal unless the time limit is hit
        model.Params.MIPGap = 0.0
        # Use as many CPUs as allocated by Slurm (fallback to all cores)
        threads_env = os.getenv('SLURM_CPUS_PER_TASK')
        if threads_env and threads_env.isdigit():
            model.Params.Threads = int(threads_env)
        else:
            import multiprocessing
            model.Params.Threads = max(1, multiprocessing.cpu_count())
        # Send Gurobi logs to a file if provided via env or local variable
        log_file = locals().get('log_file', None) or os.getenv('CARC_GRB_LOGFILE', None)
        if log_file:
            model.Params.LogFile = str(log_file)
    except Exception as _param_ex:
        # Non-fatal if params can't be set in some code paths
        pass
    # --- end param setup ---

    # --- CARC / Gurobi parameter setup ---
    import os
    try:
        # Use work_limit as a wallclock time limit (seconds).
        if work_limit is not None:
            model.Params.TimeLimit = float(work_limit)
        # Force solve-to-optimal unless time limit is hit.
        model.Params.MIPGap = 0.0
        # Use as many CPUs as Slurm grants or all local cores.
        threads_env = os.getenv("SLURM_CPUS_PER_TASK")
        if threads_env and threads_env.isdigit():
            model.Params.Threads = int(threads_env)
        else:
            import multiprocessing
            model.Params.Threads = max(1, multiprocessing.cpu_count())
        # Set Gurobi log file if specified via env var.
        log_file = os.getenv("CARC_GRB_LOGFILE", None)
        if log_file:
            model.Params.LogFile = str(log_file)
    except Exception as _param_ex:
        pass
    # --- end CARC param setup ---
    M = 600 # A large constant

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
    # i, j == -1 for depot, i, j == -2 for home
    x = model.addVars(m+2, m+2, day, n, vtype=GRB.BINARY, name="x") 

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
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+1, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[m+w] * gp.quicksum((x[m+1, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )

    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    model.setObjective(objective, GRB.MINIMIZE)

    # Pruning
    if pruning >= 1:
        # fix all x[i][j][d][w] to 0 if 1) i == j
        model.addConstrs((x[i,i,d,w] == 0 for d in range(day) for w in range(n) for i in range(m+2)), name="prune_x_equal")

    # Constraints
    # Each event happens on one day
    model.addConstrs((gp.quicksum(s[i,d] for d in range(day)) == 1 for i in range(m)), name="one_active_s")

    # Each event is scheduled exactly once during its feasible time window
    model.addConstrs((gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m)), name="one_active_t")
    model.addConstrs((t[i,d] >= time_window[i][d][0] * s[i,d] for d in range(day) for i in range(m)), name="tw_start")
    model.addConstrs((t[i,d] <= time_window[i][d][1] * s[i,d] for d in range(day) for i in range(m)), name="tw_end")
    model.addConstrs((sum(x[i,j,d,w] for j in range(m+2)) <= s[i,d] for i in range(m) for d in range(day) for w in range(n)), name="link_x_s")

    # Time feasibility for consecutive events
    # for i in range(m):
    #     for j in range(m):
    #         if i != j:
    #             model.addConstrs((t[j,d] >= t[i,d] + C_dur[i] + C_event[i,j] - M * (1 - x[i,j,d,w]) for w in range(n) for d in range(day)), name="time_feasibility")
    # model.addConstrs(
    #     (t[j, d] >= t[i, d] + C_dur[i] + C_event[i, j] - 600 * (1 - x[i, j, d, w]) for i in range(m) for j in range(m) for d in range(day) for w in range(n)),
    #     name="time_feasibility"
    # )
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

    # Minimum working hours
    if min_hour is not None:
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) >= min_hour[w] * 60 for w in range(n)),
            name="min_working_hours"
        )

    # Maximum working hours
    if max_hour is not None:
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) <= max_hour[w] * 60 for w in range(n)),
            name="max_working_hours"
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

    # maximum number of events per day
    if event_limit is not None:
        model.addConstrs(
            (gp.quicksum(x[i, j, d, w] for i in range(m) for j in range(m)) <= event_limit-1 for d in range(day) for w in range(n)),
            name="max_events_per_day"
        ) 
  
    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2)) == gp.quicksum(x[j, i, d, w] for i in range(m+2)) for j in range(m) for d in range(day) for w in range(n)),
        name="event_network_flow"
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


    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+1, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_depot_home"
    )

    model.update()

    # to_remove = [c for c in model.getConstrs()
    #          if c.ConstrName.startswith("time_feasibility[")]
    # model.remove(to_remove)
    # model.update()

    # # 2. Now re‑add with the correct Big‑M=600 baked in
    # M = 600
    # time_feas = model.addConstrs(
    #     (
    #     t[j, d]
    #         >= t[i, d] + C_dur[i] + C_event[i, j]
    #         - M * (1 - x[i, j, d, w])
    #     for i in range(m)
    #     for j in range(m)
    #     for d in range(day)
    #     for w in range(n)
    #     ),
    #     name="time_feasibility"
    # )
    # model.update()

    # warm start
    # load your list of active indices:
    filepath2 = '/Users/jinghongmiao/Code/mvt-code/result-250722/h_r10_l10_m10_wl1000_max15_17.pkl'
    

    with open(filepath2,'rb') as f:
        warm_start = pickle.load(f)
    
        # turn your saved lists into sets/dicts for fast lookup:
    active_x   = set(warm_start["active_x"])    # e.g. [(i,j,d,w), ...]
    active_s   = set(warm_start["active_s"])    # if you saved these
    active_t = dict(warm_start["active_t"])  # e.g. {(i,d): t0, ...}
    active_alpha = set(warm_start["active_alpha"])
    active_beta  = set(warm_start["active_beta"])

    model.update()

    # 1) fully warm‐start x:
    for key, var in x.items():                 # key is (i,j,d,w)
        var.Start = 1 if key in active_x else 0

    # 2) fully warm‐start s:
    for key, var in s.items():                 # key is (i,d)
        var.Start = 1 if key in active_s else 0

    # 3) fully warm‐start t (use whatever default makes sense; e.g. 0):
    for key, var in t.items():                 # key is (i,d)
        var.Start = active_t.get(key, 0)

    # 4) fully warm‐start alpha and beta:
    for key, var in alpha.items():             # key is (i,d,w)
        var.Start = 1 if key in active_alpha else 0
    for key, var in beta.items():              # key is (i,d,w)
        var.Start = 1 if key in active_beta else 0

    
    # Set the time limit to 20 minutes (1200 seconds)
    # model.setParam(GRB.Param.TimeLimit, time_limit)
    model.setParam(GRB.Param.WorkLimit, work_limit)

    model.Params.Threads = 8
    model.setParam('LogFile', 'gurobi_output_cah_sim.txt')

    start_time = time.time()
    model.optimize()
    end_time = time.time()

    elapsed_time = round(end_time - start_time, 2)  # in seconds, rounded to 2 decimals

    summary = {}
    work_time_by_nurse = {}

    model.update()
    if model.SolCount > 0:
        # c = model.getConstrByName("time_feasibility[2,5,4,0]")

        # # inspect its row
        
        # # t[j, d] >= t[i, d] + C_dur[i] + C_event[i, j] - M * (1 - x[i, j, d, w])
        # # User MIP start violates constraint time_feasibility[2,5,4,0] by 25.000000000
        # i, j, d, w = 2, 5, 4, 0

        # print("t[j,d] =", t[j,d].x)
        # print("t[i,d] =", t[i,d].x)
        # print("x[i,j,d,w] =", x[i,j,d,w].x)
        # print("C_dur[i] + C_event[i,j] =", C_dur[i] + C_event[i,j])
        # print("M * (1 - x) =", M*(1 - x[i,j,d,w].x))
        # print("---")
        # print("Check:  ")
        # print(f"  {t[j,d].x}  >=  {t[i,d].x} + {C_dur[i] + C_event[i,j]}  -  {M}*(1 - {x[i,j,d,w].x})")

        objective = model.ObjVal
        print('The optimal objective is %g' % model.objVal)
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Best bound: {model.ObjBound}")
        print(f"Gap: {model.MIPGap}")

        # file_path = f'/Users/jinghongmiao/Code/mvt-code/result-250715/ca_{nr}_{nl}_{m}_wl{work_limit}_el{event_limit}_p{pruning}_mh{min_hour}_max25_seed{seed_number}.pkl'
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

        # also return a dictionary with the working hours of each nurse
        work_time_by_nurse = {}
        for w in range(n):
            total_minutes = 0
            for j in range(m):
                for d in range(day):
                    for i in range(m + 2):
                        if x[i, j, d, w].x > 0:
                            total_minutes += C_dur[j]
                            break
            work_time_by_nurse[w] = total_minutes / 60.0  # convert to hours
        
        summary["work_time_by_nurse"] = work_time_by_nurse

    else:
        objective = None
        print(f"No feasible solution found within {work_limit} work units.")

    return summary



def continuous_fairness_algorithm (data: ProblemData, work_limit, seed_number, multiple_tw=None, event_limit=None, pruning=False, min_hour=0, max_hour=None, fairness_param = 100):

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

    # --- CARC / Gurobi parameter setup ---
    import os
    try:
        # Treat work_limit as a solver wallclock limit (seconds)
        if 'work_limit' in locals() and locals().get('work_limit') is not None:
            model.Params.TimeLimit = float(locals().get('work_limit'))
        # Solve to optimal unless the time limit is hit
        model.Params.MIPGap = 0.0
        # Use as many CPUs as allocated by Slurm (fallback to all cores)
        threads_env = os.getenv('SLURM_CPUS_PER_TASK')
        if threads_env and threads_env.isdigit():
            model.Params.Threads = int(threads_env)
        else:
            import multiprocessing
            model.Params.Threads = max(1, multiprocessing.cpu_count())
        # Send Gurobi logs to a file if provided via env or local variable
        log_file = locals().get('log_file', None) or os.getenv('CARC_GRB_LOGFILE', None)
        if log_file:
            model.Params.LogFile = str(log_file)
    except Exception as _param_ex:
        # Non-fatal if params can't be set in some code paths
        pass
    # --- end param setup ---

    # --- CARC / Gurobi parameter setup ---
    import os
    try:
        # Use work_limit as a wallclock time limit (seconds).
        if work_limit is not None:
            model.Params.TimeLimit = float(work_limit)
        # Force solve-to-optimal unless time limit is hit.
        model.Params.MIPGap = 0.0
        # Use as many CPUs as Slurm grants or all local cores.
        threads_env = os.getenv("SLURM_CPUS_PER_TASK")
        if threads_env and threads_env.isdigit():
            model.Params.Threads = int(threads_env)
        else:
            import multiprocessing
            model.Params.Threads = max(1, multiprocessing.cpu_count())
        # Set Gurobi log file if specified via env var.
        log_file = os.getenv("CARC_GRB_LOGFILE", None)
        if log_file:
            model.Params.LogFile = str(log_file)
    except Exception as _param_ex:
        pass
    # --- end CARC param setup ---
    M = 600 # A large constant

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
    # i, j == -1 for depot, i, j == -2 for home
    x = model.addVars(m+2, m+2, day, n, vtype=GRB.BINARY, name="x") 

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
        C_depot[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+1, d, w] )for d in range(day) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot[m+w] * gp.quicksum((x[m+1, m, d, w] + x[m, m+1, d, w]) for d in range(day)) for w in range(n)
    )

    # fairness
    work_minutes = model.addVars(n, name="work_minutes", lb=0)
    max_work = model.addVar(name="max_work")
    # min_work = model.addVar(name="min_work")


    travel_objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    fairness_objective = max_work
    # fairness_objective = max_work

    objective = travel_objective + fairness_param * fairness_objective

    model.setObjective(objective, GRB.MINIMIZE)

    # fairness constraints
    for w in range(n):
        model.addConstr(
            work_minutes[w] == gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)),
            name=f"def_work_time_{w}"
        )

    for w in range(n):
        model.addConstr(work_minutes[w] <= max_work, name=f"max_bound_{w}")
        # model.addConstr(work_minutes[w] >= min_work, name=f"min_bound_{w}")

    # Pruning
    if pruning >= 1:
        # fix all x[i][j][d][w] to 0 if 1) i == j
        model.addConstrs(x[i,i,d,w] == 0 for d in range(day) for w in range(n) for i in range(m+2))
    
    # Constraints
    # Each event happens on one day
    model.addConstrs(gp.quicksum(s[i,d] for d in range(day)) == 1 for i in range(m))

    # Each event is scheduled exactly once during its feasible time window
    model.addConstrs(gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m))
    for i in range(m):
        model.addConstrs(t[i,d] >= time_window[i][d][0] * s[i,d] for d in range(day))
        model.addConstrs(t[i,d] <= time_window[i][d][1] * s[i,d] for d in range(day))
    model.addConstrs(sum(x[i,j,d,w] for j in range(m+2)) <= s[i,d] for i in range(m) for d in range(day) for w in range(n))

    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                model.addConstrs(t[j,d] >= t[i,d] + C_dur[i] + C_event[i,j] - M * (1 - x[i,j,d,w]) for w in range(n) for d in range(day))

    # Each event happens during its feasible time window
    model.addConstrs(t[i,d] <= time_window[i][d][1] for i in range(m) for d in range(day))
    model.addConstrs(gp.quicksum(t[i,d] for d in range(day)) >= 1 for i in range(m))
 
    # Minimum working hours
    model.addConstrs(
        (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) >= min_hour * 60 for w in range(n)),
        name="min_working_hours"
    )

    # Maximum working hours
    if max_hour is not None:
        model.addConstrs(
            (gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+2) for d in range(day)) for j in range(m)) <= max_hour[w] * 60 for w in range(n)),
            name="max_working_hours"
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

    # maximum number of events per day
    if event_limit is not None:
        model.addConstrs(
            (gp.quicksum(x[i, j, d, w] for i in range(m) for j in range(m)) <= event_limit-1 for d in range(day) for w in range(n)),
            name="max_events_per_day"
        ) 
  
    # network flow
    # event inflow = outflow
    model.addConstrs(
        (gp.quicksum(x[i, j, d, w] for i in range(m+2)) == gp.quicksum(x[j, i, d, w] for i in range(m+2)) for j in range(m) for d in range(day) for w in range(n)),
        name="event_network_flow"
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


    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(day) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+1, m, d, w] for d in range(day) for w in range(n)),
        name="drop_off_leader_depot_home"
    )


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

    travel_objective_value = travel_objective.getValue()
    print(f"Original objective value (without penalty): {travel_objective_value}")

    elapsed_time = round(end_time - start_time, 2)  # in seconds, rounded to 2 decimals

    summary = {}
    work_time_by_nurse = {}

    if model.SolCount > 0:
        objective = model.ObjVal
        print('The optimal objective is %g' % model.objVal)
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"Best bound: {model.ObjBound}")
        print(f"Gap: {model.MIPGap}")

        file_path = f'/Users/jinghongmiao/Code/mvt-code/result-250624/fair{fairness_param}_{nr}_{nl}_{m}_wl{work_limit}_el{event_limit}_max25_seed{seed_number}.pkl'
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

        # also return a dictionary with the working hours of each nurse
        work_time_by_nurse = {}
        for w in range(n):
            total_minutes = 0
            for j in range(m):
                for d in range(day):
                    for i in range(m + 2):
                        if x[i, j, d, w].x > 0:
                            total_minutes += C_dur[j]
                            break
            work_time_by_nurse[w] = total_minutes / 60.0  # convert to hours

        # Save as pickle
        with open(file_path, "wb") as f:
            pickle.dump(summary, f)

        # print the the working hours of each nurse
        nurse_work_time = {}

        for w in range(n):  # for each nurse
            total_minutes = 0
            for j in range(m):  # for each event
                for d in range(day):
                    for i in range(m + 2):  # including home/depot/start/end points
                        if x[i, j, d, w].X > 0:
                            total_minutes += C_dur[j]
                            break  # no need to count more i→j arcs once one is found
            nurse_work_time[w] = total_minutes / 60.0  # convert to hours
        print("Nurse work hours:")
        for w, hours in nurse_work_time.items():
            print(f"Nurse {w + 1}: {hours:.2f} hours")
  
    else:
        objective = None
        print(f"No feasible solution found within {work_limit} work units.")

    return summary["objective_value"], work_time_by_nurse


# def continuous_algorithm_day (data_d: ProblemData, time_limit, seed_number, event_limit=None, pruning=False, min_hour=0, max_hour=None):
#     """
#     Continuous algorithm for the MVT scheduling problem broken down by day.
#     Parameters:
#     data (ProblemData): The problem data containing costs, time windows, and nurse requirements.
#     hour_limits (dict): A dictionary containing the maximum working hours for each nurse.
#     """
#     return continuous_algorithm(data_d, time_limit/5, seed_number, event_limit, pruning, min_hour=min_hour, max_hour=max_hour)


def continuous_algorithm_daily (data: ProblemData, work_limit, seed_number, event_limit=None, pruning=False, min_hour=0):
    """
    Continuous algorithm for the MVT scheduling problem broken down by day.
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

    current_hours = {w: 0.0 for w in range(n)}
    
    for d in range(days):

        max_hour = {w: 25 - current_hours.get(w, 0.0) for w in range(n)}

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

        print(f"\n\nRunning continuous algorithm for day {d+1} with {len(events_today)} events...")
        print(f"shape of C_event_today: {C_event_today.shape}, C_home_today: {C_home_today.shape}, C_depot_today: {C_depot_today.shape}, C_dur_today: {C_dur_today.shape}, time_window_today: {time_window_today.shape}, min_nurse_today: {min_nurse_today.shape}")
        # Call the continuous_algorithm function with the filtered data

        _, work_time_by_nurse = continuous_algorithm(data_today, work_limit/days, seed_number, event_limit=None, pruning=0, min_hour=0, max_hour=max_hour)

        for w in range(n):
            current_hours[w] += work_time_by_nurse.get(w, 0.0)

    return work_time_by_nurse
