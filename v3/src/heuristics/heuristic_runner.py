from src.heuristics.construction import generate_schedules
from src.heuristics.assignment import assign_nurses
from src.solutions.solutions import Solution, MergedSolution
from src.solutions.solution_utils import merge_day_solutions

def run_heuristic(problem, config) -> MergedSolution:
    schedules = generate_schedules(problem, config)
    print(f"Generated {len(schedules)} schedules.")
    # sols = [assign_nurses(s, problem, config) for s in schedules]

    # for i in range(config.improvement_rounds):
    #     sols = [improve(sol, problem, config) for sol in sols]
    #     sols = select_candidates(sols, config)

    return 0 
