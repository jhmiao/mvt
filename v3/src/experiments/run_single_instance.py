import sys
from pathlib import Path

# Ensure project root is on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.io.data_loader import load_problem_data
from src.solver.solver_runner import solve
from src.solver.config import SolverConfig

def main():
    data_path = PROJECT_ROOT / "data" / "cleaned" / "c101_Even.xlsx"
    problem = load_problem_data(data_path)

    config = SolverConfig(
        solve_by_day=False,     # or True
        fairness_objective=False,
        use_warmstart=False,
        half_hour_starts=True,
    )

    solution = solve(problem, config)


if __name__ == "__main__":
    main()
