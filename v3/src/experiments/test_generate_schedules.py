from pathlib import Path
import sys

# Ensure project root (v3/) is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.heuristics.construction import generate_schedules
from src.heuristics.assignment import assign_nurses_with_depot
from src.heuristics.config import HeuristicConfig
from src.io.data_loader import load_problem_data


def main():
    instance = "c101"
    event_type = "Even"
    data_path = PROJECT_ROOT / "data" / "cleaned" / f"{instance}_{event_type}.xlsx"
    problem = load_problem_data(data_path)

    config = HeuristicConfig(
        num_samples=10,
        seed=0,
        gurobi_outputflag=0
    )

    schedules = generate_schedules(problem, config)

    for d in range(problem.total_day):
        print(f"Day {d}:")
        for idx, day_schedule in enumerate(schedules[d]):
            solution = assign_nurses_with_depot(problem, day_schedule, d, config)
            print(f"    Solution: {solution}")
        break

if __name__ == "__main__":
    main()
