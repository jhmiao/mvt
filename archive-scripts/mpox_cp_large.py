from ortools.sat.python import cp_model
import numpy as np
import random

random.seed(10)

print('hello')

# Coefficients
nr = 15 # number of RNs 
nl = 35 # number of LVNs
m = 50 # number of events
block = 5 # number of days/blocks

# travel cost: event1-m x event1-m
C_event = np.random.randint(5, 26, size=(m, m))
# Set the diagonal elements to 0
np.fill_diagonal(C_event, 0)

# travel cost: event1-m x (home RN1-nr, LVN1-nl)
C_home = np.random.randint(5, 21, size=(m, nr+nl))

# travel cost: (event1-m, RN1-nr) x depot
C_depot = np.random.randint(5, 21, size=(m+nr))

# event cost: event1-m
# Define the array of possible values
dur_values = np.array([30, 45, 60, 90, 120, 150])

# Define the probability distribution
dur_probabilities = np.array([0.3, 0.1, 0.2, 0.2, 0.1, 0.1])  # Adjust probabilities as needed

# Generate the array
C_dur = np.random.choice(dur_values, size=m, p = dur_probabilities)

# feasible time window (earliest, latest) by minutes past 9:00am 
# event1-m x 5 days

# Function to generate a random time window
def generate_time_window():
    start = np.random.randint(0, 11) * 30  # Start time: 0 to 300 by 30s
    end = np.random.randint(start // 30 + 1, 12) * 30  # End time: start + 30 to 330 by 30s
    return [start, end]

# Initialize the matrix
time_window = np.zeros((m, block, 2), dtype=int)

# Populate the matrix with feasible time windows
for i in range(m):
    for j in range(block):
        time_window[i, j] = generate_time_window()

# print(np.shape(time_window))

# minimum number of nurses required for each event
# event1-20 x (RN, LVN)
min_values = np.array([1, 2, 3, 4, 5, 6, 7])
min_probabilities_RN = np.array([0.4, 0.4, 0.2, 0, 0, 0, 0])
min_probabilities_LVN = np.array([0.1, 0.3, 0.3, 0.1, 0.1, 0.05, 0.05])
min_nurse = np.zeros((m, 2), dtype=int)
for i in range(m):
    min_nurse[i][0] = np.random.choice(min_values, p = min_probabilities_RN)
    min_nurse[i][1] = np.random.choice(min_values, p = min_probabilities_LVN)

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
                # model.Add(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j] - 390 * (1 - x[i][j][d][w]))
                model.Add(t[j][d] >= t[i][d] + C_dur[i] + C_event[i][j]).OnlyEnforceIf(x[i][j][d][w])

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

# Sets a time limit of 10 seconds.
solver.parameters.max_time_in_seconds = 60.0

status = solver.solve(model)

# if status == cp_model.OPTIMAL:
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Minimum of objective function: {solver.objective_value}\n")
    # for i in range(m):
    #     for d in range(block):
    #         if solver.value(s[i][d]) == 1:
    #             print(f"\n Event {i+1} is scheduled on day {d+1} at {get_time(solver.value(t[i][d]))}.")
    #             for w in range(nr+nl):
    #                 if w < nr: # RN
    #                     if solver.value(delta[i][d][w]) == 1:
    #                         print(f"Nurse {w+1} is the team leader for Event {i+1} on day {d+1}.")
    #                     if solver.value(xhome_depot[d][w]) == 1:
    #                         print(f"Nurse {w+1} picks up at Depot on day {d+1}.")
    #                     if solver.value(xdepot_home[d][w]) == 1:
    #                         print(f"Nurse {w+1} drops off at Depot on day {d+1}.")
    #                 # LVN
    #                 if solver.value(xhome_event[i][d][w]) == 1:
    #                     print(f"Nurse {w+1} goes from home to Event {i+1} on day {d+1}.")
    #                 for j in range(m):
    #                     if solver.value(x[j][i][d][w]) == 1:
    #                         print(f"Nurse {w+1} goes from Event {j+1} to {i+1} on day {d+1}.")
else:
    print("No solution found.")


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