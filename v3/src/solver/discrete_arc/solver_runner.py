from __future__ import annotations

from gurobipy import GRB

from src.structures.problem_data import ProblemData
from src.solver.config import SolverConfig
from .model_builder_disc import build_model


def solve(problem_data: ProblemData, solver_config: SolverConfig):
    model = build_model(problem_data, solver_config)
    model.optimize()

    if model.SolCount > 0:
        print(f"Objective value: {model.ObjVal}")
        return model

    status = model.Status
    # if model.Status == GRB.INFEASIBLE:
        # model.computeIIS()
        # model.write("infeasible.ilp")   # or .iis
    status_name = {
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }.get(status, str(status))
    print(f"Objective value: unavailable (status={status_name})")
    return model
