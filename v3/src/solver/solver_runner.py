from __future__ import annotations
from src.structures.problem_data import ProblemData
from src.solver.config import SolverConfig
from src.solver.model_builder import build_model
# from src.solver.solution_utils import extract_solution, merge_day_solutions

def solve(problem_data, solver_config):
    if solver_config.solve_by_day:
        return solve_by_day(problem_data, solver_config)
    else:
        return solve_full(problem_data, solver_config)


def solve_full(problem_data: ProblemData, solver_config: SolverConfig):
    model = build_model(problem_data, solver_config)
    model.optimize()
    return model.ObjVal

def solve_by_day(problem_data: ProblemData, solver_config: SolverConfig):
    day_instances = problem_data.split_by_day()
    day_solutions = []

    for day_instance in day_instances:
        pass