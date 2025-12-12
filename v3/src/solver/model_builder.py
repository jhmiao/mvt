import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.structures.problem_data import ProblemData
from src.solver.config import SolverConfig
from .constraints import add_base_constraints


def build_model(problem_data: ProblemData, config: SolverConfig) -> gp.Model:
    """
    Build and return a Gurobi optimization model based on the provided problem data and configuration.

    Parameters:
    problem_data (ProblemData): Travel costs, event durations, time windows, nurse requirements, etc.
    config (SolverConfig): Configuration settings for the solver.

    Returns:
    gp.Model: A Gurobi optimization model.
    """
    model = gp.Model("Nurse_Scheduling_Routing_Problem")

    C_event = problem_data.event_event_costs
    C_home = problem_data.home_event_costs
    C_depot_e = problem_data.event_depot_costs
    C_depot_h = problem_data.home_depot_costs
    C_depot = np.concatenate([C_depot_e, C_depot_h])
    C_dur = problem_data.event_durations
    time_windows = problem_data.time_windows
    min_nurses = problem_data.min_nurses
    nr = problem_data.total_rn
    nl = problem_data.total_lvn
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    # Variables

    # x_ijdw = 1 if nurse w goes from event i to j on day d, 0 otherwise
    # i, j == m for home, i, j == m+1 for depot_am, i, j == m+2 for depot_pm
    x = model.addVars(m+3, m+3, days, n, vtype=GRB.BINARY, name="x") 
    # s_id = 1 if event i is scheduled on day d, 0 otherwise
    s = model.addVars(m, days, vtype=GRB.BINARY, name="s")
    # t_id time when event i starts on day d
    t = model.addVars(m, days, vtype=GRB.INTEGER, name="t")
    # alpha_idw = 1 if nurse w is the pick-up leader for event i on day d, 0 otherwise
    # beta_idw = 1 if nurse w is the drop-off leader for event i on day d, 0 otherwise
    alpha = model.addVars(m, days, n, vtype=GRB.BINARY, name="alpha")
    beta = model.addVars(m, days, n, vtype=GRB.BINARY, name="beta")

    # Constraints
    add_base_constraints(model, problem_data, x, s, t, alpha, beta)

    if config.half_hour_starts:
        # Add discrete time constraints
        from .constraints import add_discrete_time_constraints
        add_discrete_time_constraints(model, problem_data, t)

    if config.enforce_max_hours:
        from .constraints import add_max_hour_constraints
        add_max_hour_constraints(model, problem_data, x)
    
    # Objective
    if config.fairness_objective:
        from .objectives import add_fairness_objective
        add_fairness_objective(model, problem_data)
    else:
        from .objectives import add_baseline_objectives
        add_baseline_objectives(model, problem_data, x, s, t, alpha, beta)

    # Set solver parameters from config
    model.Params.WorkLimit = config.work_limit
    model.Params.Seed = config.seed

    return model
