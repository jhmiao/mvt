from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the route-based RMP on a single instance.")
    default_project = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=default_project, help="Project root path")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to a cleaned .xlsx file under v3/data/cleaned.",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit in seconds (None to disable)")
    parser.add_argument("--work-limit", type=float, default=None, help="Gurobi WorkLimit (None to disable)")
    parser.add_argument("--gurobi-output", type=int, default=1, help="Gurobi OutputFlag (0=quiet, 1=verbose)")
    parser.add_argument("--seed", type=int, default=0, help="Gurobi Seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.io.data_loader import load_problem_data  # noqa: E402
    from src.solver.route.rmp_runner import solve_rmp_routes  # noqa: E402
    from src.solver.route.pool_builder import RoutePoolConfig  # noqa: E402

    data_path = Path(args.file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    problem = load_problem_data(str(data_path))

    pool_cfg = RoutePoolConfig()

    result = solve_rmp_routes(
        problem=problem,
        pool_cfg=pool_cfg,
        time_limit=args.time_limit,
        work_limit=args.work_limit,
        seed=args.seed,
        outputflag=args.gurobi_output,
    )

    print(
        "RMP status=", result.status,
        "runtime=", result.runtime,
        "obj=", result.obj_val,
        "bound=", result.obj_bound,
        "gap=", result.mip_gap,
        "nodes=", result.node_count,
    )


if __name__ == "__main__":
    main()

# run with:
# python v3/src/experiments/run_rmp_routes.py --file v3/data/cleaned/c101_Random1.xlsx
