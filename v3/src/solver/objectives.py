import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.structures.problem_data import ProblemData

def add_baseline_objectives(model: gp.Model, problem_data: ProblemData, x, s, t, alpha, beta):
    """
    Add baseline objectives to the Gurobi model.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.

    Returns:
    None
    """
    C_event = problem_data.event_event_costs
    C_home = problem_data.home_event_costs
    C_depot_e = problem_data.event_depot_costs
    C_depot_h = problem_data.home_depot_costs
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(days) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[w,i] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(days)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot_e[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+2, d, w] )for d in range(days) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot_h[w] * gp.quicksum((x[m+2, m, d, w] + x[m, m+1, d, w]) for d in range(days)) for w in range(n)
    )

    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    model.setObjective(objective, GRB.MINIMIZE)


def add_fairness_objective(model: gp.Model, problem_data: ProblemData):
    """
    Add fairness objective to the Gurobi model.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.

    Returns:
    None
    """
    # Example: Minimize the maximum workload among nurses
    # This is a placeholder; actual implementation will depend on the problem specifics

    # Assuming we have variables workload[i] representing workload of nurse i
    # max_workload = model.addVar(name="max_workload")
    # model.addConstrs((workload[i] <= max_workload for i in range(problem_data.total_nurse)), name="Fairness_Constraints")
    # model.setObjective(max_workload, GRB.MINIMIZE)
    pass