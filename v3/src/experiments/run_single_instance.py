import sys
from pathlib import Path

# Ensure project root is on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.io.data_loader import load_problem_data
from src.solver.solver_runner import solve
from src.solver.config import SolverConfig
from src.solutions.io import save_solution_pickle, save_solution_json

def main():
    instance = 'c101'
    event_type = 'Even'  # 'Odd' or 'Even'
    data_path = PROJECT_ROOT / "data" / "cleaned" / f"{instance}_{event_type}.xlsx"
    problem = load_problem_data(data_path)

    config = SolverConfig(
        solve_by_day=True,     # or True
        fairness_objective=False,
        use_warmstart=False,
        half_hour_starts=True,
        gurobi_outputflag=0,
        work_limit=50,        # e.g., 600 Gurobi work units
    )

    solution = solve(problem, config)
    save_solution_pickle(solution, PROJECT_ROOT / "outputs" / f"{instance}_{event_type}.pkl")
    save_solution_json(solution, PROJECT_ROOT / "outputs" / f"{instance}_{event_type}.json")


if __name__ == "__main__":
    main()
