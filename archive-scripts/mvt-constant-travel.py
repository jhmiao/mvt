from ortools.sat.python import cp_model
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import time

time_limit = 600
seed_number = 21

np.random.seed(seed_number)

# Coefficients
nr = 15 # number of RNs 
nl = 35 # number of LVNs
m = 75 # number of events
block = 5 # number of days
T = 12 # number of 30-minute time slots in a day

# helper functions
# translate the time slot to the actual time
def time_slot_to_time(time_slot):
    day = time_slot // T + 1
    hour = int((time_slot % T) // 2 + 9)
    minute = int((time_slot % T) % 2 * 30)
    if minute < 10:
        minute = "0" + str(minute)
    return f"day {day} at {hour}:{minute}"


# event cost: event1-20

# Define the array of possible values
dur_values = np.array([1,2,3,4,6,8])
# Define the probability distribution
dur_probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Adjust probabilities as needed
# Generate the array
C_dur = np.random.choice(dur_values, size=m, p = dur_probabilities)

# feasible time window: set entry to 1 if it is a feasible starting time
# Initialize the matrix
time_window = np.zeros((m, T*block), dtype=int)

for i in range(m):
    for j in range(T*block):
        # only possibly set entry to 1 when when the finishing time does not exceed daily upper bound
        if j % T <= T - C_dur[i]:
            time_window[i, j] = np.random.choice([1,0],p=[0.5,0.5])

# minimum number of nurses required for each event
# event1-20 x (RN, LVN)
min_values = np.array([1, 2, 3, 4, 5, 6, 7])
min_probabilities_RN = np.array([0.4, 0.4, 0.2, 0, 0, 0, 0])
min_probabilities_LVN = np.array([0.1, 0.3, 0.3, 0.1, 0.1, 0.05, 0.05])
min_nurse = np.zeros((m, 2), dtype=int)
for i in range(m):
    min_nurse[i][0] = np.random.choice(min_values, p = min_probabilities_RN)
    min_nurse[i][1] = np.random.choice(min_values, p = min_probabilities_LVN)


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

    # Objective function

    objective = 1

    model.setObjective(objective, GRB.MINIMIZE)

    # Constraints

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

    # try:
    #     model.optimize()
    # except gp.GurobiError:
    #     print("Optimize failed due to infeasibility")

    # Set the time limit to 20 minutes (1200 seconds)
    model.setParam(GRB.Param.TimeLimit, time_limit)
    model.optimize()

    if model.SolCount > 0:
        # print('The optimal objective is %g' % model.objVal)
        
        with open(f"discrete_{nr}_{nl}_{m}_seed{seed_number}.txt", "w") as file:
            file.write(f"Best feasible solution found within {time_limit} seconds:\n")
            file.write(f"Seed number: {seed_number}\n")
            file.write(f"Number of RNs: {nr}\n")
            file.write(f"Number of LVNs: {nl}\n")
            file.write(f"Objective value: {model.objVal}\n")
            file.write("Iterations: ")
            file.write(str(iterations))
            file.write("\n")
            file.write("Schedule: \n\n")

            # for t in range(T*block):
            #     day = t // T
            #     for day in range(5):
            #         file.write(f"Day {day+1}\n")
            #         for j in range(m):
            #             if f[j][t].x == 1:
            #                 file.write(f"Event {j+1}: {time_slot_to_time(t)}\n")
            #                 for w in range(nr+nl):
            #                     if x[j][t][w].x == 1:
            #                         if w < nr:
            #                             file.write(f"RN {w+1}\n")
            #                         else:
            #                             file.write(f"LVN {w+1}\n")

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
            
            file.write(f"\n\nNurse schedule:\n")
            for w in range(nr+nl):
                file.write(f"\nNurse {w+1}:\n")
                for t in range(T*block):
                    for j in range(m):
                        if y[j][t][w].x == 1:
                            file.write(f"Event {j+1}: {time_slot_to_time(t)}, {C_dur[j]*30} minutes \n")
                            break
    # print("No feasible solution found within 20 minutes.")


gurobi_solve()
