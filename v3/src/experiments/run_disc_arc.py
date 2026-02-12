from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discrete-arc model on a single instance.")
    default_project = Path(__file__).resolve().parents[2]
    default_instance = default_project / "data" / "cleaned" / "c101_Random2.xlsx"
    parser.add_argument("--project-root", type=Path, default=default_project, help="Project root path")
    parser.add_argument("--instance-path", type=Path, default=default_instance, help="Path to instance .xlsx")
    parser.add_argument("--work-limit", type=float, default=None, help="Gurobi WorkLimit (None to disable)")
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit in seconds (None to disable)")
    parser.add_argument("--gurobi-output", type=int, default=1, help="Gurobi OutputFlag (0=quiet, 1=verbose)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Delayed imports so project_root is on sys.path
    from src.io.data_loader import load_problem_data  # noqa: E402
    from src.solver.config import SolverConfig  # noqa: E402
    from src.solver.discrete_arc.solver_runner import solve  # noqa: E402
    from src.solutions.extract_discrete_arc_result import (  # noqa: E402
        build_discrete_arc_output_path,
        extract_discrete_arc_result,
        save_discrete_arc_json,
    )

    data_path = args.instance_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    problem = load_problem_data(data_path, sample_k=10, sample_seed=45)

    config = SolverConfig(
        solve_by_day=False,
        fairness_objective=True,
        enforce_hour_balance=False,
        use_warmstart=False,
        half_hour_starts=False,
        gurobi_outputflag=args.gurobi_output,
        work_limit=args.work_limit,
        time_limit=args.time_limit,
    )

    model = solve(problem, config)
    if model is None:
        return

    result = extract_discrete_arc_result(model)
    output_dir = project_root / "outputs" / "discrete_arc"
    output_path = build_discrete_arc_output_path(data_path, output_dir)
    save_discrete_arc_json(result, output_path)
    print("Saved discrete-arc solution to", output_path)


if __name__ == "__main__":
    main()

# Example command to run:
# python v3/src/experiments/run_disc_arc.py --instance-path v3/data/cleaned/c101_Random2.xlsx --time-limit 600 --gurobi-output 1
