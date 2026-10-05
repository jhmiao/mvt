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
            model.addConstr(s[i,d] <= time_window[i][d][0], name=f"tw_lb_s_i{i}_d{d}")
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
    
    # set all x[i,j,d,w]=0 for i,j where C_dur[i]+C_dur[i]+C_event[i,j] > 8 * 60
    model.addConstrs(
        (
            x[i, j, d, w] == 0
            for i in range(m)
            for j in range(m)
            for d in range(days)
            for w in range(n)
            if C_dur[i] + C_dur[j] + C_event[i, j] > 8 * 60
        ),
        name="prune_long_route",
    )



    
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

    # # depot inflow = outflow
    # model.addConstrs(
    #     (x[m, m+1, d, w] == gp.quicksum(x[m+1, j, d, w] for j in range(m)) for d in range(days) for w in range(n)),
    #     name="morning_depot_flow"
    # )

    # model.addConstrs(
    #     (x[m+2, m, d, w] == gp.quicksum(x[j, m+2, d, w] for j in range(m)) for d in range(days) for w in range(n)),
    #     name="evening_depot_flow"
    # )
    
    # # team leader: exactly one pick up and one drop off leader for each event
    # model.addConstrs(
    #     (gp.quicksum(alpha[j, d, w] for w in range(n) for d in range(days)) == 1 for j in range(m)),
    #     name="pick_up_leader"
    # )

    # model.addConstrs(
    #     (gp.quicksum(beta[j, d, w] for w in range(n) for d in range(days)) == 1 for j in range(m)),
    #     name="drop_off_leader"
    # )

    # # make leader the same person: alpha[j,d,w] = beta[j,d,w] for all j,d,w
    # model.addConstrs(
    #     (alpha[j, d, w] == beta[j, d, w] for j in range(m) for d in range(days) for w in range(n)),
    #     name="same_leader_alpha_beta"
    # )

    # # team leader goes to the event
    # model.addConstrs(
    #     (alpha[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(days) for w in range(n)),
    #     name="pick_up_leader_event"
    # )

    # model.addConstrs(
    #     (beta[j, d, w] <= gp.quicksum(x[i, j, d, w] for i in range(m+3)) for j in range(m) for d in range(days) for w in range(n)),
    #     name="drop_off_leader_event"
    # )

    # # pick up leader goes from home to depot to their first event
    # # drop off leader goes from their last event to depot to home
    # model.addConstrs(
    #     (gp.quicksum(alpha[j, d, w] for j in range(m)) <= 5 * x[m, m+1, d, w] for d in range(days) for w in range(n)),
    #     name="pick_up_leader_home_depot"
    # )

    # model.addConstrs(
    #     (gp.quicksum(beta[j, d, w] for j in range(m)) <= 5 * x[m+2, m, d, w] for d in range(days) for w in range(n)),
    #     name="drop_off_leader_depot_home"
    # )

def add_depot_constraints(model: gp.Model, problem_data: ProblemData, x, alpha, beta):
    m = problem_data.total_event
    n = problem_data.total_nurse
    days = problem_data.total_day
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

    # make leader the same person: alpha[j,d,w] = beta[j,d,w] for all j,d,w
    model.addConstrs(
        (alpha[j, d, w] == beta[j, d, w] for j in range(m) for d in range(days) for w in range(n)),
        name="same_leader_alpha_beta"
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


def add_leader_irredundancy_constraints(model: gp.Model, problem_data: ProblemData, x, s):
    """Require a private physical-coverage witness for every depot-day trip.

    Call after add_base_constraints and add_depot_constraints: event flow is
    binary and bounded by s, and depot flow links the home/depot arcs to events.
    Reuse those flows as u[i,d,w] and g[d,w], rather than creating duplicate
    indicators. Existing alpha/beta constraints already ensure valid leaders.

    Return a dictionary of u/g expressions and ell/q/p tupledicts for inspection.
    ell and p can be continuous: binary u/g force ell to be integral; one-hot
    q and p <= ell then force the unique witness to be integral as well.
    No big-M constants are used here.
    """
    m = problem_data.total_event
    n = problem_data.total_nurse
    days = problem_data.total_day
    home, depot_am, depot_pm = m, m + 1, m + 2

    u = gp.tupledict({
        (i, d, w): gp.quicksum(x[i, j, d, w] for j in range(m + 3) if j != i)
        for i in range(m) for d in range(days) for w in range(n)
    })
    g = gp.tupledict({
        (d, w): x[home, depot_am, d, w]
        for d in range(days) for w in range(n)
    })
    ell = model.addVars(m, days, n, lb=0.0, ub=1.0,
                        vtype=GRB.CONTINUOUS, name="depot_cover")
    q = model.addVars(m, days, n + 1, vtype=GRB.BINARY,
                      name="depot_cover_count")
    p = model.addVars(m, days, n, lb=0.0, ub=1.0,
                      vtype=GRB.CONTINUOUS, name="depot_witness")

    for d in range(days):
        for w in range(n):
            # The existing depot flow equations handle event-facing arcs.
            model.addConstr(g[d, w] == x[depot_pm, home, d, w],
                            name=f"irredundancy_depot_pair_d{d}_w{w}")
            for a, b in ((depot_am, depot_pm), (depot_pm, depot_am)):
                model.addConstr(x[a, b, d, w] == 0,
                                name=f"irredundancy_no_depot_cross_i{a}_j{b}_d{d}_w{w}")
            model.addConstr(g[d, w] <= gp.quicksum(p[i, d, w] for i in range(m)),
                            name=f"irredundant_depot_d{d}_w{w}")

    for i in range(m):
        for d in range(days):
            for w in range(n):
                model.addConstr(ell[i, d, w] <= u[i, d, w],
                                name=f"depot_cover_visit_i{i}_d{d}_w{w}")
                model.addConstr(ell[i, d, w] <= g[d, w],
                                name=f"depot_cover_depot_i{i}_d{d}_w{w}")
                model.addConstr(ell[i, d, w] >= u[i, d, w] + g[d, w] - 1,
                                name=f"depot_cover_and_i{i}_d{d}_w{w}")
                model.addConstr(p[i, d, w] <= ell[i, d, w],
                                name=f"witness_cover_i{i}_d{d}_w{w}")
            model.addConstr(gp.quicksum(q[i, d, k] for k in range(n + 1)) == 1,
                            name=f"depot_count_onehot_i{i}_d{d}")
            model.addConstr(
                gp.quicksum(ell[i, d, w] for w in range(n))
                == gp.quicksum(k * q[i, d, k] for k in range(n + 1)),
                name=f"depot_count_value_i{i}_d{d}")
            # Base flow forbids unscheduled attendance; scheduled events need
            # physical depot coverage. This also handles the zero-nurse case.
            model.addConstr(q[i, d, 0] == 1 - s[i, d],
                            name=f"depot_count_schedule_i{i}_d{d}")
            model.addConstr(gp.quicksum(p[i, d, w] for w in range(n))
                            == (q[i, d, 1] if n else 0),
                            name=f"unique_witness_i{i}_d{d}")

    return {"u": u, "g": g, "ell": ell, "q": q, "p": p}


def add_no_depot_constraints(model: gp.Model, problem_data: ProblemData, x, alpha, beta):
    """Disable all depot arcs and leader variables for a no-depot solve."""
    m = problem_data.total_event
    n = problem_data.total_nurse
    days = problem_data.total_day
    depot_nodes = (m + 1, m + 2)

    model.addConstrs(
        (
            x[i, j, d, w] == 0
            for i in range(m + 3)
            for j in depot_nodes
            for d in range(days)
            for w in range(n)
        ),
        name="no_depot_inflow",
    )
    model.addConstrs(
        (
            x[i, j, d, w] == 0
            for i in depot_nodes
            for j in range(m + 3)
            for d in range(days)
            for w in range(n)
        ),
        name="no_depot_outflow",
    )
    model.addConstrs(
        (alpha[i, d, w] == 0 for i in range(m) for d in range(days) for w in range(n)),
        name="no_depot_alpha",
    )
    model.addConstrs(
        (beta[i, d, w] == 0 for i in range(m) for d in range(days) for w in range(n)),
        name="no_depot_beta",
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

def add_hour_balance_constraints(model: gp.Model, problem_data: ProblemData, x):
    """
    Add working hour balance constraints to the Gurobi model.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.
    x: Decision variable for nurse routes. x[i,j,d,w] = 1 if nurse w goes from event i to j on day d. i = m for home, i = m+1 for depot_am, i = m+2 for depot_pm.
    avg_hours: Average working hours per nurse.

    Returns:
    None
    """

    C_dur = problem_data.event_durations
    min_nurses = problem_data.min_nurses
    nr = problem_data.total_rn
    nl = problem_data.total_lvn
    # calculate average hours for RNs and LVNs separately
    total_RN_minutes = np.sum(C_dur * min_nurses[:, 0])
    total_LVN_minutes = np.sum(C_dur * min_nurses[:, 1])

    # Compute average working hours
    avg_RN_minutes = total_RN_minutes / nr if nr > 0 else 0
    avg_LVN_minutes = total_LVN_minutes / nl if nl > 0 else 0

    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    for w in range(nr):
        model.addConstr(
            gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days)) for j in range(m)) >= avg_RN_minutes * 0.8,
            name=f"min_balanced_hours_RN_w{w}"
        )
        model.addConstr(
            gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days)) for j in range(m)) <= avg_RN_minutes * 1.5,
            name=f"max_balanced_hours_RN_w{w}"
        )
    
    for w in range(nr, n):
        model.addConstr(
            gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days)) for j in range(m)) >= avg_LVN_minutes * 0.8,
            name=f"min_balanced_hours_LVN_w{w}"
        )
        model.addConstr(
            gp.quicksum(C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m+3) for d in range(days)) for j in range(m)) <= avg_LVN_minutes * 1.5,
            name=f"max_balanced_hours_LVN_w{w}"
        )
