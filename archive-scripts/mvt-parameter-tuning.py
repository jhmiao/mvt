from ortools.sat.python import cp_model
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import time

time_limit = 600
seed_number = 120

random.seed(seed_number)

# Coefficients
nr = 10 # number of RNs 
nl = 20 # number of LVNs
m = 20 # number of events
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
        time_window[i, j] = generate_time_window()

        # generate a partial time window table
        # time_window[i, j] = random.choice([[0,0], generate_time_window()])

# print(np.shape

# minimum number of nurses required for each event
# event1-20 x (RN, LVN)
min_values = np.array([1, 2, 3, 4, 5, 6, 7])
min_probabilities_RN = np.array([0.4, 0.4, 0.2, 0, 0, 0, 0])
min_probabilities_LVN = np.array([0.1, 0.3, 0.3, 0.1, 0.1, 0.05, 0.05])
min_nurse = np.zeros((m, 2), dtype=int)
for i in range(m):
    min_nurse[i][0] = np.random.choice(min_values, p = min_probabilities_RN)
    min_nurse[i][1] = np.random.choice(min_values, p = min_probabilities_LVN)


# pruning: 
# 1） reduce the number of constraints for time feasibility
# 2） 
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
    x = [[[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)] for i in range(m)]

    # x_0jdw = 1 if nurse w goes from home to event j on day d, 0 otherwise
    xhome_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_0{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)]

    # x_i0dw = 1 if nurse w goes from event i to home on day d, 0 otherwise
    xevent_home = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}0{d}{w}") for w in range(nr+nl)] for d in range(block)] for i in range(m)]

    # x_0*dw = 1 if nurse w goes from home to depot on day d, 0 otherwise
    xhome_depot = [[model.addVar(vtype=GRB.BINARY, name=f"x_0*{d}{w}") for w in range(nr)] for d in range(block)]

    # x_*0dw = 1 if nurse w goes from depot to home on day d, 0 otherwise
    xdepot_home = [[model.addVar(vtype=GRB.BINARY, name=f"x_*0{d}{w}") for w in range(nr)] for d in range(block)]

    # x_*jdw = 1 if nurse w goes from depot to event j on day d, 0 otherwise
    xdepot_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_*{j}{d}{w}") for w in range(nr)] for d in range(block)] for j in range(m)]

    # x_i*dw = 1 if nurse w goes from event i to depot on day d, 0 otherwise
    xevent_depot = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}*{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = [[model.addVar(vtype=GRB.BINARY, name=f"s_{i}{d}") for d in range(block)] for i in range(m)] 

    # t_id time when event i starts on day d
    t = [[model.addVar(0, 390, vtype=GRB.INTEGER, name=f"t_{i}{d}") for d in range(block)] for i in range(m)]

    # alpha_idw = 1 if nurse w (RN) is the pick-up leader for event i on day d, 0 otherwise
    alpha = [[[model.addVar(vtype=GRB.BINARY, name=f"alpha_{i}{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]

    # # beta_idw = 1 if nurse w (RN) is the drop-9 leader for event i on day d, 0 otherwise
    # beta = [[[model.addVar(vtype=GRB.BINARY, name=f"beta_{i}{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]


    # Objective function

    cost_between_events = np.sum(np.sum(np.sum(x, axis=3), axis=2) * C_event)
    cost_home_event = np.sum(np.sum(xhome_event, axis=1)* C_home + np.sum(xevent_home, axis=1)*C_home)
    cost_home_depot = np.sum(np.sum(xhome_depot, axis=0)*C_depot[-nr:] + np.sum(xdepot_home, axis=0)*C_depot[-nr:])
    cost_depot_event = np.sum(np.sum(xdepot_event, axis=1) * np.tile(C_depot[:m], (nr, 1)).T 
                            + np.sum(xevent_depot, axis=1) * np.tile(C_depot[:m], (nr, 1)).T)
    # print(np.shape(cost_depot_event))

    objective = cost_between_events + cost_home_event + cost_home_depot + cost_depot_event

    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # fix all x[i][j][d][w] to 0 if 1) i == j or 2) time window is infeasible: no overlapping days (double check)
    for i in range(m):
        for j in range(m):
            if i == j:
                for d in range(block):
                    for w in range(nr+nl):
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
                for w in range(nr+nl):
                    model.addConstr(xevent_home[i][d][w] == 0)
                    model.addConstr(xhome_event[i][d][w] == 0)
                    model.addConstr(xevent_depot[i][d][w] == 0)
                    model.addConstr(xdepot_event[i][d][w] == 0)
                    for j in range(m):
                        model.addConstr(x[i][j][d][w] == 0)
                        model.addConstr(x[j][i][d][w] == 0)
            else: # time window is feasible
                for w in range(nr+nl):
                    if w < nr: # RN
                        model.addConstr(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w] <= s[i][d])
                    else: # LVN
                        model.addConstr(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] <= s[i][d])

    # Each event happens 
    for i in range(m):
        model.addConstr(sum(t[i][d] for d in range(block)) >= 1)
        for d in range(block):
            if time_window[i][d][1] > 0:
                model.addConstr(t[i][d] >= time_window[i][d][0] * s[i][d])
                model.addConstr(t[i][d] <= time_window[i][d][1] * s[i][d])

    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                for d in range(block):
                    # only add this constraint when 1) event i is scheduled and 2) the latest time event j can happen is feasible for travel
                    if time_window[i][d][1] > 0 and time_window[j][d][1] >= time_window[i][d][0] + C_dur[i] + C_event[i][j]:
                        for w in range(nr+nl):
                            model.addConstr(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j] - M * (1 - x[i][j][d][w]))
                        # model.addConstr(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j]).OnlyEnforceIf(x[i][j][d][w])

    # Minimum working hours (20 hours per week)
    for w in range(nr+nl):
        if w < nr: # RN
            model.addConstr(sum(C_dur[j] * 
                        sum(
                            (sum(x[i][j][d][w] for i in range(m)) 
                            + xhome_event[j][d][w]  
                            + xdepot_event[j][d][w]
                            ) for d in range(block)
                            ) for j in range(m)) >= 60*3) 
        else: # LVN
            model.addConstr(sum(C_dur[j] * 
                        sum(
                            (sum(x[i][j][d][w] for i in range(m)) 
                            + xhome_event[j][d][w]
                            ) for d in range(block)
                            ) for j in range(m)) >= 60*3) 
            
    # Each event is assigned to the correct number of nurses
    for j in range(m):
        # RN
        model.addConstr(sum((sum(x[i][j][d][w] for i in range(m))
                    + xhome_event[j][d][w] + xdepot_event[j][d][w])
                    for d in range(block) for w in range(nr)) >= min_nurse[j][0])
        # LVN
        model.addConstr(sum((sum(x[i][j][d][w] for i in range(m)) 
                    + xhome_event[j][d][w])
                    for d in range(block) for w in range(nr, nr+nl)) >= min_nurse[j][1])

    # Network flow constraints
    # inflow = outflow for each event （does NOT involve depots) 
    for j in range(m):
        for d in range(block):
            # only add constraints when the event is feasible on day d
            if time_window[j][d][1] > 0:
                for w in range(nr): # RN
                    model.addConstr(sum(x[i][j][d][w] for i in range(m))
                            + xhome_event[j][d][w] 
                            + xdepot_event[j][d][w]
                            == sum(x[j][i][d][w] for i in range(m)) 
                                + xevent_home[j][d][w]
                                + xevent_depot[j][d][w])
                for w in range(nr, nr+nl): # LVN
                    model.addConstr(sum(x[i][j][d][w] for i in range(m))
                            + xhome_event[j][d][w] 
                            == sum(x[j][i][d][w] for i in range(m))
                                + xevent_home[j][d][w])

    # home -> event, event -> home for LVN
    for d in range(block):
        for w in range(nr, nr+nl):
            model.addConstr(sum(xhome_event[j][d][w] for j in range(m)) <= 1)
        for w in range (nr):
            model.addConstr(xhome_depot[d][w] + sum(xhome_event[j][d][w] for j in range(m)) <= 1)
            # network flow for depots
            model.addConstr(xhome_depot[d][w] == sum(xdepot_event[j][d][w] for j in range(m)))
            model.addConstr(xdepot_home[d][w] == sum(xevent_depot[j][d][w] for j in range(m)))

    # team leader constraints
    # one team leader for each event
    for i in range(m):
        for d in range(block):
            # set alpha[i][d][w] = 0 if the event is not scheduled
            if time_window[i][d][1] == 0:
                for w in range(nr):
                    model.addConstr(alpha[i][d][w] == 0)
            else:
                for w in range(nr):
                    # team leader is a RN who goes to the event
                    model.addConstr(alpha[i][d][w] <= sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w])
        # exactly one RN is the team leader
        model.addConstr(sum(alpha[i][d][w] for w in range(nr) for d in range(block)) == 1)
                
    # team leader goes from home to depot to their first event, and goes from their last event to depot to home
    for w in range(nr):
        for d in range(block):
            model.addConstr(xhome_depot[d][w] == xdepot_home[d][w])
            # team leader goes from home to depot to their first event if they are the team leader for some event on that day
            for i in range(m):
                model.addConstr(sum(alpha[i][d][w] for i in range(m)) <= 10 * xhome_depot[d][w])
            
    # try:
    #     model.optimize()
    # except gp.GurobiError:
    #     print("Optimize failed due to infeasibility")

    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3
    model.optimize()

    if model.SolCount > 0:
        # print('The optimal objective is %g' % model.objVal)
        
        with open(f"optimization_results_{nr}_{nl}_{m}_seed{seed_number}.txt", "w") as file:
            file.write(f"Best feasible solution found within {time_limit} seconds:\n")
            for d in range(block):
                file.write(f"\n=====================\nDay {d+1} \n=====================\n")
                for i in range(m):
                    if s[i][d].x == 1:
                        file.write(f"\nEvent {i+1}: {get_time(t[i][d].x)}-{get_time(t[i][d].x + C_dur[i])} [{min_nurse[i][0]} RN, {min_nurse[i][1]} LVN] \n")
                        for w in range(nr):
                            if alpha[i][d][w].x == 1:
                                file.write(f"RN {w+1} (team leader)\n")
                            else:
                                if sum(x[j][i][d][w].x for j in range(m)) + xhome_event[i][d][w].x + xdepot_event[i][d][w].x >= 1:
                                    file.write(f"RN {w+1}\n")
                        for w in range(nr, nr+nl):
                            if sum(x[j][i][d][w].x for j in range(m)) + xhome_event[i][d][w].x >= 1:
                                    file.write(f"LVN {w+1}\n")
            # Write the objective value to the file
            file.write(f"\nObjective value: {model.ObjVal}\n")
    else:
        print("No feasible solution found within 20 minutes.")

def gurobi_solve1():
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
    x = [[[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)] for i in range(m)]

    # x_0jdw = 1 if nurse w goes from home to event j on day d, 0 otherwise
    xhome_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_0{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)]

    # x_i0dw = 1 if nurse w goes from event i to home on day d, 0 otherwise
    xevent_home = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}0{d}{w}") for w in range(nr+nl)] for d in range(block)] for i in range(m)]

    # x_0*dw = 1 if nurse w goes from home to depot on day d, 0 otherwise
    xhome_depot = [[model.addVar(vtype=GRB.BINARY, name=f"x_0*{d}{w}") for w in range(nr)] for d in range(block)]

    # x_*0dw = 1 if nurse w goes from depot to home on day d, 0 otherwise
    xdepot_home = [[model.addVar(vtype=GRB.BINARY, name=f"x_*0{d}{w}") for w in range(nr)] for d in range(block)]

    # x_*jdw = 1 if nurse w goes from depot to event j on day d, 0 otherwise
    xdepot_event = [[[model.addVar(vtype=GRB.BINARY, name=f"x_*{j}{d}{w}") for w in range(nr)] for d in range(block)] for j in range(m)]

    # x_i*dw = 1 if nurse w goes from event i to depot on day d, 0 otherwise
    xevent_depot = [[[model.addVar(vtype=GRB.BINARY, name=f"x_{i}*{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]

    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = [[model.addVar(vtype=GRB.BINARY, name=f"s_{i}{d}") for d in range(block)] for i in range(m)] 

    # t_id time when event i starts on day d
    t = [[model.addVar(0, 390, vtype=GRB.INTEGER, name=f"t_{i}{d}") for d in range(block)] for i in range(m)]

    # alpha_idw = 1 if nurse w (RN) is the team leader for event i on day d, 0 otherwise
    alpha = [[[model.addVar(vtype=GRB.BINARY, name=f"alpha_{i}{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]


    # Objective function

    cost_between_events = np.sum(np.sum(np.sum(x, axis=3), axis=2) * C_event)
    cost_home_event = np.sum(np.sum(xhome_event, axis=1)* C_home + np.sum(xevent_home, axis=1)*C_home)
    cost_home_depot = np.sum(np.sum(xhome_depot, axis=0)*C_depot[-nr:] + np.sum(xdepot_home, axis=0)*C_depot[-nr:])
    cost_depot_event = np.sum(np.sum(xdepot_event, axis=1) * np.tile(C_depot[:m], (nr, 1)).T 
                            + np.sum(xevent_depot, axis=1) * np.tile(C_depot[:m], (nr, 1)).T)
    # print(np.shape(cost_depot_event))

    objective = cost_between_events + cost_home_event + cost_home_depot + cost_depot_event

    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints
    # fix all x[i][j][d][w] to 0 if 1) i == j or 2) time window is infeasible: no overlapping days (double check)
    for i in range(m):
        for j in range(m):
            if i == j:
                for d in range(block):
                    for w in range(nr+nl):
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
                for w in range(nr+nl):
                    model.addConstr(xevent_home[i][d][w] == 0)
                    model.addConstr(xhome_event[i][d][w] == 0)
                    model.addConstr(xevent_depot[i][d][w] == 0)
                    model.addConstr(xdepot_event[i][d][w] == 0)
                    for j in range(m):
                        model.addConstr(x[i][j][d][w] == 0)
                        model.addConstr(x[j][i][d][w] == 0)
            else: # time window is feasible
                for w in range(nr+nl):
                    if w < nr: # RN
                        model.addConstr(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w] <= s[i][d])
                    else: # LVN
                        model.addConstr(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] <= s[i][d])

    # Each event happens 
    for i in range(m):
        model.addConstr(sum(t[i][d] for d in range(block)) >= 1)
        for d in range(block):
            if time_window[i][d][1] > 0:
                model.addConstr(t[i][d] >= time_window[i][d][0] * s[i][d])
                model.addConstr(t[i][d] <= time_window[i][d][1] * s[i][d])

    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                for d in range(block):
                    # only add this constraint when 1) event i is scheduled and 2) the latest time event j can happen is feasible for travel
                    if time_window[i][d][1] > 0 and time_window[j][d][1] >= time_window[i][d][0] + C_dur[i] + C_event[i][j]:
                        for w in range(nr+nl):
                            model.addConstr(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j] - M * (1 - x[i][j][d][w]))
                        # model.addConstr(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j]).OnlyEnforceIf(x[i][j][d][w])

    # Minimum working hours (20 hours per week)
    for w in range(nr+nl):
        if w < nr: # RN
            model.addConstr(sum(C_dur[j] * 
                        sum(
                            (sum(x[i][j][d][w] for i in range(m)) 
                            + xhome_event[j][d][w]  
                            + xdepot_event[j][d][w]
                            ) for d in range(block)
                            ) for j in range(m)) >= 60*3) 
        else: # LVN
            model.addConstr(sum(C_dur[j] * 
                        sum(
                            (sum(x[i][j][d][w] for i in range(m)) 
                            + xhome_event[j][d][w]
                            ) for d in range(block)
                            ) for j in range(m)) >= 60*3) 
            
    # Each event is assigned to the correct number of nurses
    for j in range(m):
        # RN
        model.addConstr(sum((sum(x[i][j][d][w] for i in range(m))
                    + xhome_event[j][d][w] + xdepot_event[j][d][w])
                    for d in range(block) for w in range(nr)) >= min_nurse[j][0])
        # LVN
        model.addConstr(sum((sum(x[i][j][d][w] for i in range(m)) 
                    + xhome_event[j][d][w])
                    for d in range(block) for w in range(nr, nr+nl)) >= min_nurse[j][1])

    # Network flow constraints
    # inflow = outflow for each event （does NOT involve depots) 
    for j in range(m):
        for d in range(block):
            # only add constraints when the event is feasible on day d
            if time_window[j][d][1] > 0:
                for w in range(nr): # RN
                    model.addConstr(sum(x[i][j][d][w] for i in range(m))
                            + xhome_event[j][d][w] 
                            + xdepot_event[j][d][w]
                            == sum(x[j][i][d][w] for i in range(m)) 
                                + xevent_home[j][d][w]
                                + xevent_depot[j][d][w])
                for w in range(nr, nr+nl): # LVN
                    model.addConstr(sum(x[i][j][d][w] for i in range(m))
                            + xhome_event[j][d][w] 
                            == sum(x[j][i][d][w] for i in range(m))
                                + xevent_home[j][d][w])

    # home -> event, event -> home for LVN
    for d in range(block):
        for w in range(nr, nr+nl):
            model.addConstr(sum(xhome_event[j][d][w] for j in range(m)) <= 1)
        for w in range (nr):
            model.addConstr(xhome_depot[d][w] + sum(xhome_event[j][d][w] for j in range(m)) <= 1)
            # network flow for depots
            model.addConstr(xhome_depot[d][w] == sum(xdepot_event[j][d][w] for j in range(m)))
            model.addConstr(xdepot_home[d][w] == sum(xevent_depot[j][d][w] for j in range(m)))

    # team leader constraints
    # one team leader for each event
    for i in range(m):
        for d in range(block):
            # set alpha[i][d][w] = 0 if the event is not scheduled
            if time_window[i][d][1] == 0:
                for w in range(nr):
                    model.addConstr(alpha[i][d][w] == 0)
            else:
                for w in range(nr):
                    # team leader is a RN who goes to the event
                    model.addConstr(alpha[i][d][w] <= sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w])
        # exactly one RN is the team leader
        model.addConstr(sum(alpha[i][d][w] for w in range(nr) for d in range(block)) == 1)
                
    # team leader goes from home to depot to their first event, and goes from their last event to depot to home
    for w in range(nr):
        for d in range(block):
            model.addConstr(xhome_depot[d][w] == xdepot_home[d][w])
            # team leader goes from home to depot to their first event if they are the team leader for some event on that day
            for i in range(m):
                model.addConstr(sum(alpha[i][d][w] for i in range(m)) <= 10 * xhome_depot[d][w])
            
    # try:
    #     model.optimize()
    # except gp.GurobiError:
    #     print("Optimize failed due to infeasibility")

    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    # model.setParam(GRB.Param.MIPFocus, 1)
    # model.setParam(GRB.Param.Heuristics, 0.5)
    # model.setParam(GRB.Param.Cuts, 1)
    # model.setParam(GRB.Param.Presolve, 2)
    # model.Params.PoolSearchMode = 1
    # model.Params.PoolSolutions = 3
    model.optimize()

    if model.SolCount > 0:
        # print('The optimal objective is %g' % model.objVal)
        
        with open(f"network_{nr}_{nl}_{m}_seed{seed_number}.txt", "w") as file:
            file.write(f"Best feasible solution found within {time_limit} seconds:\n")
            for d in range(block):
                file.write(f"\n=====================\nDay {d+1} \n=====================\n")
                for i in range(m):
                    if s[i][d].x == 1:
                        file.write(f"\nEvent {i+1}: {get_time(t[i][d].x)}-{get_time(t[i][d].x + C_dur[i])} [{min_nurse[i][0]} RN, {min_nurse[i][1]} LVN] \n")
                        for w in range(nr):
                            if alpha[i][d][w].x == 1:
                                file.write(f"RN {w+1} (team leader)\n")
                            else:
                                if sum(x[j][i][d][w].x for j in range(m)) + xhome_event[i][d][w].x + xdepot_event[i][d][w].x >= 1:
                                    file.write(f"RN {w+1}\n")
                        for w in range(nr, nr+nl):
                            if sum(x[j][i][d][w].x for j in range(m)) + xhome_event[i][d][w].x >= 1:
                                    file.write(f"LVN {w+1}\n")
            # Write the objective value to the file
            file.write(f"\nObjective value: {model.ObjVal}\n")
    else:
        print("No feasible solution found.")

# cp_solve()
# gurobi_solve()
# gurobi_solve1()
# gurobi_solve()
gurobi_solve1()
