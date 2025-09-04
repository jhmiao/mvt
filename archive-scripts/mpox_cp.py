from ortools.sat.python import cp_model
import numpy as np

print('hello')

# Coefficients

nr = 2 # number of RNs 
nl = 3 # number of LVNs
m = 5 # number of events
block = 5 # number of days/blocks

# travel cost: event1-5 x event1-5
C_event = np.array([[0,15,25,5,28],
                    [15,0,25,22,15],
                    [25,25,0,20,16],
                    [5,22,20,0,24],
                    [28,15,16,24,0]])

# travel cost: event1-5 x (home RN1-2, LVN1-3)
C_home = np.array([[11,12,11,12,10],
                   [13,14,12,14,18],
                   [15,16,13,16,14],
                   [17,18,14,18,16],
                   [19,10,15,10,12]])

# travel cost: (event1-5, RN1-2) x depot
C_depot = np.array([8,9,10,11,12,15,10])

# event cost: event1-5
C_dur = np.array([30, 30, 120, 90, 60])

# feasible time window (earliest, latest) starting time by minutes past 9:00am 
# event1-5 x 5 days
# time_window = np.array([[[240, 330],[300,330],[0,0],[0,0],[0,0]],
#                         [[0,0],[0,0],[120,240],[240,360],[0,0]],
#                         [[0,0],[0,0],[0,0],[0,0],[60,180]],
#                         [[0,0],[0,0],[270,360],[0,0],[0,0]],
#                         [[60,360],[0,0],[0,0],[0,0],[0,0]]])
time_window = np.array([[[240, 360],[300,360],[0,0],[0,0],[0,0]],
                        [[0,0],[0,0],[120,240],[240,360],[0,0]],
                        [[0,0],[0,0],[0,0],[0,0],[60,180]],
                        [[0,0],[0,0],[270,360],[0,0],[0,0]],
                        [[60,360],[0,0],[0,0],[0,0],[0,0]]])

# print(np.shape(time_window))

# minimum number of nurses required for each event
# event1-5 x (RN, LVN)
min_nurse = np.array([[1,1],
                      [1,1],
                      [2,2],
                      [1,2],
                      [2,2]])

model = cp_model.CpModel()

# Decision variables

# x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
x = [[[[model.NewBoolVar(f"x_{i}{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)] for i in range(m)]

# x_0jdw = 1 if nurse w goes from home to event j on day d, 0 otherwise
xhome_event = [[[model.NewBoolVar(f"x_0{j}{d}{w}") for w in range(nr+nl)] for d in range(block)] for j in range(m)]

# x_i0dw = 1 if nurse w goes from event i to home on day d, 0 otherwise
xevent_home = [[[model.NewBoolVar(f"x_{i}0{d}{w}") for w in range(nr+nl)] for d in range(block)] for i in range(m)]

# x_0*dw = 1 if nurse w goes from home to depot on day d, 0 otherwise
xhome_depot = [[model.NewBoolVar(f"x_0*{d}{w}") for w in range(nr)] for d in range(block)]

# x_*0dw = 1 if nurse w goes from depot to home on day d, 0 otherwise
xdepot_home = [[model.NewBoolVar(f"x_*0{d}{w}") for w in range(nr)] for d in range(block)]

# x_*jdw = 1 if nurse w goes from depot to event j on day d, 0 otherwise
xdepot_event = [[[model.NewBoolVar(f"x_*{j}{d}{w}") for w in range(nr)] for d in range(block)] for j in range(m)]

# x_i*dw = 1 if nurse w goes from event i to depot on day d, 0 otherwise
xevent_depot = [[[model.NewBoolVar(f"x_{i}*{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]

# s_id = 1 if event i is scheduled on day d, 0 otherwise
s = [[model.NewBoolVar(f"s_{i}{d}") for d in range(block)] for i in range(m)] 

# t_id time when event i starts on day d
t = [[model.NewIntVar(0, 390, f"t_{i}{d}") for d in range(block)] for i in range(m)]

# delta_idw = 1 if nurse w (RN) is the team leader for event i on day d, 0 otherwise
delta = [[[model.NewBoolVar(f"delta_{i}{d}{w}") for w in range(nr)] for d in range(block)] for i in range(m)]

# Objective function

# cost_between_events = np.sum(np.sum(np.sum(x, axis=3), axis=2) * C_event)
# cost_home_event = np.sum(np.sum(xhome_event, axis=1)* C_home + np.sum(xevent_home, axis=1)*C_home)
# cost_home_depot = np.sum(np.sum(xhome_depot, axis=0)*C_depot[-2:] + np.sum(xdepot_home, axis=0)*C_depot[-2:])
# cost_depot_event = np.sum(np.sum(xdepot_event, axis=1) * np.stack((C_depot[:5],C_depot[:5])).T + np.sum(xevent_depot, axis=1) * np.stack((C_depot[:5],C_depot[:5])).T)
# # print(np.shape(cost_depot_event))

# objective = cost_between_events + cost_home_event + cost_home_depot + cost_depot_event



cost_between_events = np.sum(np.sum(np.sum(x, axis=3), axis=2) * C_event)
cost_home_event = np.sum(np.sum(xhome_event, axis=1)* C_home + np.sum(xevent_home, axis=1)*C_home)
cost_home_depot = np.sum(np.sum(xhome_depot, axis=0)*C_depot[-nr:] + np.sum(xdepot_home, axis=0)*C_depot[-nr:])
cost_depot_event = np.sum(np.sum(xdepot_event, axis=1) * np.tile(C_depot[:m], (nr, 1)).T 
                          + np.sum(xevent_depot, axis=1) * np.tile(C_depot[:m], (nr, 1)).T)
# print(np.shape(cost_depot_event))

objective = cost_between_events + cost_home_event + cost_home_depot + cost_depot_event

# Constraints

# Each event is scheduled exactly once
for i in range(m):
    model.Add(sum(s[i][d] for d in range(block)) == 1)
    for d in range(block):
        for w in range(nr+nl):
            if w < nr: # RN
                model.Add(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w] <= s[i][d])
            else: # LVN
                model.Add(sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] <= s[i][d])

# Each event happens during its feasible time window
for i in range(m):
    model.Add(sum(t[i][d] for d in range(block)) > 0)
    for d in range(block):
        model.Add(t[i][d] >= time_window[i][d][0] * s[i][d])
        model.Add(t[i][d] <= time_window[i][d][1] * s[i][d])

# Time feasibility for consecutive events
for i in range(m):
    for j in range(m):
        for d in range(block):
            for w in range(nr+nl):
                # travel time considered
                model.Add(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j]).OnlyEnforceIf(x[i][j][d][w])

                # no travel time considered
                # model.Add(t[j][d] >= t[i][d] + C_dur[i]).OnlyEnforceIf(x[i][j][d][w])

# Minimum working hours (20 hours per week)
for w in range(nr+nl):
    if w < nr: # RN
        model.Add(sum(C_dur[j] * 
                      sum(
                          (sum(x[i][j][d][w] for i in range(m)) 
                           + xhome_event[j][d][w]  
                           + xdepot_event[j][d][w]
                           ) for d in range(block)
                           ) for j in range(m)) >= 60) 
    else: # LVN
        model.Add(sum(C_dur[j] * 
                      sum(
                          (sum(x[i][j][d][w] for i in range(m)) 
                           + xhome_event[j][d][w]
                           ) for d in range(block)
                           ) for j in range(m)) >= 60) 
        
# Each event is assigned to the correct number of nurses
for j in range(m):
    # RN
    model.Add(sum((sum(x[i][j][d][w] for i in range(m))
                   + xhome_event[j][d][w] + xdepot_event[j][d][w])
                   for d in range(block) for w in range(nr)) >= min_nurse[j][0])
    # LVN
    model.Add(sum((sum(x[i][j][d][w] for i in range(m)) 
                   + xhome_event[j][d][w])
                   for d in range(block) for w in range(nr, nr+nl)) >= min_nurse[j][1])

# Network flow constraints
# inflow = outflow for each event （does NOT involve depots) 
for j in range(m):
    for d in range(block):
        for w in range(nr): # RN
            model.Add(sum(x[i][j][d][w] for i in range(m))
                      + xhome_event[j][d][w] 
                      + xdepot_event[j][d][w]
                      == sum(x[j][i][d][w] for i in range(m)) 
                        + xevent_home[j][d][w]
                        + xevent_depot[j][d][w])
# #             # at most 1 unit of flow for each nurse
#             model.Add(sum(x[i][j][d][w] for i in range(m))
#                       + xhome_event[j][d][w] 
#                       + xdepot_event[j][d][w] <= 1)
        for w in range(nr, nr+nl): # LVN
            model.Add(sum(x[i][j][d][w] for i in range(m))
                      + xhome_event[j][d][w] 
                      == sum(x[j][i][d][w] for i in range(m))
                        + xevent_home[j][d][w])
            # # at most 1 unit of flow for each nurse
            # model.Add(sum(x[i][j][d][w] for i in range(m))
            #           + xhome_event[j][d][w] <= 1)

# home -> event, event -> home for LVN
for d in range(block):
    for w in range(nr, nr+nl):
        model.Add(sum(xhome_event[j][d][w] for j in range(m)) <= 1)
        # model.Add(sum(xevent_home[i][d][w] for i in range(m)) <= 1)
        # model.Add(sum(xhome_event[j][d][w] for j in range(m)) == sum(xevent_home[i][d][w] for i in range(m)))
    for w in range (nr):
        model.Add(xhome_depot[d][w] + sum(xhome_event[j][d][w] for j in range(m)) <= 1)
        # network flow for depots
        model.Add(xhome_depot[d][w] == sum(xdepot_event[j][d][w] for j in range(m)))
        model.Add(xdepot_home[d][w] == sum(xevent_depot[j][d][w] for j in range(m)))

# team leader constraints
# one team leader for each event
for i in range(m):
    # exactly one RN is the team leader
    model.Add(sum(delta[i][d][w] for w in range(nr) for d in range(block)) == 1)
    for d in range(block):
        for w in range(nr):
            # team leader happens only if the event is scheduled
            # model.Add(delta[i][d][w] <= s[i][d])
            # team leader is the RN who goes to the event
            model.Add(delta[i][d][w] <= sum(x[i][j][d][w] for j in range(m)) + xevent_home[i][d][w] + xevent_depot[i][d][w])

# team leader goes from home to depot to their first event, and goes from their last event to depot to home
for w in range(nr):
    for d in range(block):
        model.Add(xhome_depot[d][w] == xdepot_home[d][w])
        # team leader goes from home to depot to their first event if they are the team leader for some event on that day
        for i in range(m):
            model.Add(xhome_depot[d][w] == 1).OnlyEnforceIf(delta[i][d][w])
        # model.Add(sum(delta[i][d][w] for i in range(m)) <= 10*xhome_depot[d][w])
        # model.Add(xhome_depot[d][w] <= sum(delta[i][d][w] for i in range(m)))
        # model.Add(1-sum(delta[i][d][w] for i in range(m)) <= sum(xhome_event[j][d][w] for j in range(m)))
        # model.Add(10*sum(xhome_event[j][d][w] for j in range(m)) <= 10-sum(delta[i][d][w] for i in range(m)))

# helper functions
def get_time(minute):
    hour = minute // 60 + 9
    minute = minute % 60
    if minute < 10:
        minute = "0" + str(minute)
    return f"{hour}:{minute}"

model.Minimize(objective)

solver = cp_model.CpSolver()
status = solver.solve(model)

if status == cp_model.OPTIMAL:
    print(f"Minimum of objective function: {solver.objective_value}\n")
    for i in range(m):
        for d in range(block):
            if solver.value(s[i][d]) == 1:
                print(f"\n Event {i+1} is scheduled on day {d+1} at {get_time(solver.value(t[i][d]))}.")
                for w in range(nr+nl):
                    if w < nr: # RN
                        if solver.value(delta[i][d][w]) == 1:
                            print(f"Nurse {w+1} is the team leader for Event {i+1} on day {d+1}.")
                        if solver.value(xhome_depot[d][w]) == 1:
                            print(f"Nurse {w+1} picks up at Depot on day {d+1}.")
                        if solver.value(xdepot_home[d][w]) == 1:
                            print(f"Nurse {w+1} drops off at Depot on day {d+1}.")
                    # LVN
                    if solver.value(xhome_event[i][d][w]) == 1:
                        print(f"Nurse {w+1} goes from home to Event {i+1} on day {d+1}.")
                    for j in range(m):
                        if solver.value(x[j][i][d][w]) == 1:
                            print(f"Nurse {w+1} goes from Event {j+1} to {i+1} on day {d+1}.")
else:
    print("No solution found.")


# time_window = np.array([[[210, 330],[270,330],[-1,-1],[-1,-1],[-1,-1]],
#                         [[-1,-1],[-1,-1],[90,210],[210,330],[0,0]],
#                         [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[30,150]],
#                         [[-1,-1],[-1,-1],[240,330],[0,0],[0,0]],
#                         [[-1,-1],[-1,-1],[-1,-1],[-1,-1],[30,330]]])

# class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
#     """Print intermediate solutions."""

#     def __init__(self, variables: list[cp_model.IntVar]):
#         cp_model.CpSolverSolutionCallback.__init__(self)
#         self.__variables = variables
#         self.__solution_count = 0

#     def on_solution_callback(self) -> None:
#         self.__solution_count += 1
#         for v in self.__variables:
#             print(f"{v}={self.value(v)}", end=" ")
#         print()

#     @property
#     def solution_count(self) -> int:
#         return self.__solution_count


# def search_for_all_solutions_sample_sat():
#     """Showcases calling the solver to search for all solutions."""
#     # Creates the model.
#     model = cp_model.CpModel()

#     # Creates the variables.
#     num_vals = 3
#     x = model.new_int_var(0, num_vals - 1, "x")
#     y = model.new_int_var(0, num_vals - 1, "y")
#     z = model.new_int_var(0, num_vals - 1, "z")

#     # Create the constraints.
#     model.add(x != y)

#     # Create a solver and solve.
#     solver = cp_model.CpSolver()
#     solution_printer = VarArraySolutionPrinter([x, y, z])
#     # Enumerate all solutions.
#     solver.parameters.enumerate_all_solutions = True
#     # Solve.
#     status = solver.solve(model, solution_printer)

#     print(f"Status = {solver.status_name(status)}")
#     print(f"Number of solutions found: {solution_printer.solution_count}")

# search_for_all_solutions_sample_sat()