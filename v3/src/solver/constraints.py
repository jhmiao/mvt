import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.structures.problem_data import ProblemData

def add_base_constraints(model: gp.Model, problem_data: ProblemData, 
                         x, s, t, alpha, beta):
    """
    Add base constraints to the Gurobi model.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.
    x: Decision variable for nurse routes. x[i,j,d,w] = 1 if nurse w goes from event i to j on day d. i = m for home, i = m+1 for depot_am, i = m+2 for depot_pm.
    s: Decision variable for event scheduling. s[i,d] = 1 if event i is scheduled on day d.
    t: Decision variable for event start times. t[i,d] = start time of event i on day d.
    alpha: Decision variable for pick-up leaders. alpha[i,d,w] = 1 if nurse w is the pick-up leader for event i on day d.
    beta: Decision variable for drop-off leaders. beta[i,d,w] = 1 if nurse w is the drop-off leader for event i on day d.

    Returns:
    None
    """
    C_event = problem_data.event_event_costs

    C_dur = problem_data.event_durations
    time_window = problem_data.time_windows
    min_nurse = problem_data.min_nurses
    nr = problem_data.total_rn
    nl = problem_data.total_lvn
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    # prune invalid routes
    for d in range(days):
        for w in range(n):
            for i in range(m+3):
                model.addConstr(x[i,i,d,w] == 0, name=f"no_self_loop_i{i}_d{d}_w{w}")
            for i in range(m):
                model.addConstr(x[i,m+1,d,w] == 0, name=f"no_event_to_depotam_i{i}_d{d}_w{w}")
                model.addConstr(x[m+2,i,d,w] == 0, name=f"no_depotpm_to_event_i{i}_d{d}_w{w}")
            model.addConstr(x[m+1,m,d,w] == 0, name=f"no_depotam_to_home_d{d}_w{w}")
            model.addConstr(x[m,m+2,d,w] == 0, name=f"no_home_to_depotpm_d{d}_w{w}")
    
    # Each event happens on one day
    for i in range(m):
        model.addConstr(gp.quicksum(s[i,d] for d in range(days)) == 1, name=f"event_once_i{i}")

    # Each event is scheduled exactly once during its feasible time window
    for i in range(m):
        model.addConstr(gp.quicksum(t[i,d] for d in range(days)) >= 1, name=f"event_time_once_i{i}")
    for d in range(days):
        for i in range(m):
            model.addConstr(t[i,d] >= time_window[i][d][0] * s[i,d], name=f"tw_lb_i{i}_d{d}")
            model.addConstr(t[i,d] <= time_window[i][d][1] * s[i,d], name=f"tw_ub_i{i}_d{d}")
    for i in range(m):
        for d in range(days):
            for w in range(n):
                model.addConstr(sum(x[i,j,d,w] for j in range(m+3)) <= s[i,d], name=f"event_flow_i{i}_d{d}_w{w}")

    M = 1440  # large constant for time constraints
    # Time feasibility for consecutive events
    for i in range(m):
        for j in range(m):
            if i != j:
                model.addConstrs(t[j,d] >= t[i,d] + C_dur[i] + C_event[i,j] - M * (1 - x[i,j,d,w]) for w in range(n) for d in range(days))
    
    # staffing
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days) for w in range(nr)) >= min_nurse[j][0],
            name=f"min_RN_j{j}"
        )
        model.addConstr(
            gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days) for w in range(nr, nr+nl)) >= min_nurse[j][1],
            name=f"min_LVN_j{j}"
        )
  
    # network flow
    # event inflow = outflow
    for j in range(m):
        for d in range(days):
            for w in range(n):
                model.addConstr(
                    gp.quicksum(x[i, j, d, w] for i in range(m+3)) == gp.quicksum(x[j, i, d, w] for i in range(m+3)),
                    name=f"event_network_flow_j{j}_d{d}_w{w}"
                )

    # outflow from home is at most 1
    model.addConstrs(
        (gp.quicksum(x[m, i, d, w] for i in range(m+3)) <= 1 for d in range(days) for w in range(n)),
        name="home_outflow"
    )

    # depot inflow = outflow
    model.addConstrs(
        (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(days) for w in range(n)),
        name="morning_depot_flow"
    )

    model.addConstrs(
        (x[m+2, m, d, w] == gp.quicksum(x[j, m+2, d, w] for j in range(m)) for d in range(days) for w in range(n)),
        name="evening_depot_flow"
    )
    
    # team leader: exactly one pick up and one drop off leader for each event
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for w in range(n) for d in range(days)) == 1 for j in range(m)),
        name="pick_up_leader"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for w in range(n) for d in range(days)) == 1 for j in range(m)),
        name="drop_off_leader"
    )

    # team leader goes to the event
    model.addConstrs(
        (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(days) for w in range(n)),
        name="pick_up_leader_event"
    )

    model.addConstrs(
        (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(days) for w in range(n)),
        name="drop_off_leader_event"
    )

    # pick up leader goes from home to depot to their first event
    # drop off leader goes from their last event to depot to home
    model.addConstrs(
        (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(days) for w in range(n)),
        name="pick_up_leader_home_depot"
    )

    model.addConstrs(
        (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+2, m, d, w] for d in range(days) for w in range(n)),
        name="drop_off_leader_depot_home"
    )


def add_discrete_time_constraints(model: gp.Model, problem_data: ProblemData, t):
    """
    Add discrete time constraints to the Gurobi model.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.
    t: Decision variable for event start times. t[i,d] = start time of event i on day d.

    Returns:
    None
    """
    # t[i,d] must be in {0, 30, 60, ..., 1440}
    m = problem_data.total_event
    days = problem_data.total_day

    slots = model.addVars(m, days, vtype=GRB.INTEGER, lb=0, ub=48, name="t_slots")
    for i in range(m):
        for d in range(days):
            model.addConstr(t[i, d] == 30 * slots[i, d], name=f"discrete_time_i{i}_d{d}")

def add_max_hour_constraints(model: gp.Model, problem_data: ProblemData, x):
    """
    Add maximum working hour constraints to the Gurobi model.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.
    x: Decision variable for nurse routes. x[i,j,d,w] = 1 if nurse w goes from event i to j on day d. i = m for home, i = m+1 for depot_am, i = m+2 for depot_pm.
    
    Returns:
    None
    """

    C_dur = problem_data.event_durations
    max_hours = problem_data.max_hours

    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    for w in range(n):
        model.addConstr(
            gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days)) for j in range(m)) <= max_hours[w] * 60,
            name=f"max_working_hours_w{w}"
        )
