from __future__ import annotations
import gurobipy as gp
from src.structures.problem_data import ProblemData
from src.solver.config import SolverConfig
from src.solver.model_builder import build_model
from src.solutions.solution_utils import extract_solution, merge_day_solutions

def optimize_model(model: gp.Model) -> gp.Model:
    model.optimize()
    return model

def solve(problem_data, solver_config):
    if solver_config.solve_by_day:
        return solve_by_day(problem_data, solver_config)
    else:
        return solve_full(problem_data, solver_config)


def solve_full(problem_data: ProblemData, solver_config: SolverConfig):
    model = build_model(problem_data, solver_config)
    optimize_model(model)
    solution = extract_solution(model, problem_data)
    return solution

def solve_by_day(problem_data: ProblemData, solver_config: SolverConfig):
    day_instances = problem_data.split_by_day()
    day_solutions = []

    for day_instance in day_instances:
        model = build_model(day_instance, solver_config)
        optimize_model(model)
        day_solution = extract_solution(model, day_instance)
        day_solutions.append(day_solution)
    
    solution = merge_day_solutions(day_solutions, full_problem_data=problem_data)
    
    return solution
