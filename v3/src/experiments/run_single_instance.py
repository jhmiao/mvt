from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a single instance.")
    default_project = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=default_project, help="Project root path")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory (default: <project-root>/outputs)")
    parser.add_argument("--instance", type=str, default="c101", help="Instance name (e.g., c101, c201, r101, r201)")
    parser.add_argument("--event-type", type=str, default="Even", help="Event type (Even, Skewed1, Skewed2, Random1, Random2)")
    parser.add_argument("--work-limit", type=float, default=None, help="Gurobi WorkLimit (None to disable)")
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit in seconds (None to disable)")
    parser.add_argument("--gurobi-output", type=int, default=1, help="Gurobi OutputFlag (0=quiet, 1=verbose)")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or (project_root / "outputs")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Delayed imports so project_root is on sys.path
    from src.io.data_loader import load_problem_data  # noqa: E402
    from src.solver.config import SolverConfig  # noqa: E402
    from src.solver.solver_runner import solve  # noqa: E402
    from src.solutions.io import save_solution_json, save_solution_pickle  # noqa: E402

    data_path = project_root / "data" / "cleaned" / f"{args.instance}_{args.event_type}.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    problem = load_problem_data(data_path, sample_k=10, sample_seed=45)


    config = SolverConfig(
        solve_by_day=False,
        fairness_objective=False,
        enforce_hour_balance=False,
        use_warmstart=False,
        half_hour_starts=True,
        gurobi_outputflag=args.gurobi_output,
        work_limit=args.work_limit,
        time_limit=args.time_limit,
    )

    solution = solve(problem, config)
    base = output_root / f"{args.instance}_{args.event_type}"
    save_solution_pickle(solution, base.with_suffix(".pkl"))
    save_solution_json(solution, base.with_suffix(".json"))
    print(f"Solved {args.instance}_{args.event_type} -> {base.with_suffix('.pkl')} / .json")


if __name__ == "__main__":
    main()

# Example command to run:
# python v3/src/experiments/run_single_instance.py --instance c101 --event-type Random1
