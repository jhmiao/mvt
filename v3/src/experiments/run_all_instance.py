from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple


INSTANCES = ["c101", "c201", "r101", "rc101"]
EVENT_TYPES = ["Even", "Skewed1", "Skewed2", "Random1", "Random2"]


def iter_instances(project_root: Path) -> Iterable[Tuple[str, str, Path]]:
    for inst in INSTANCES:
        for ev_type in EVENT_TYPES:
            yield (
                inst,
                ev_type,
                project_root / "data" / "cleaned" / f"{inst}_{ev_type}.xlsx",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve all configured instances.")
    default_project = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project,
        help="Root of the v3 project",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write outputs (default: <project-root>/outputs)",
    )
    parser.add_argument(
        "--work-limit",
        type=float,
        default=50,
        help="Gurobi WorkLimit (set None to disable)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Gurobi TimeLimit in seconds (set None to disable)",
    )
    parser.add_argument(
        "--gurobi-output",
        type=int,
        default=0,
        help="Gurobi OutputFlag (0=quiet, 1=verbose)",
    )
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

    config = SolverConfig(
        solve_by_day=False,
        fairness_objective=True,
        use_warmstart=False,
        half_hour_starts=True,
        gurobi_outputflag=args.gurobi_output,
        work_limit=args.work_limit,
        time_limit=args.time_limit,
    )

    for inst, ev_type, path in iter_instances(project_root):
        print(f"Solving {inst}_{ev_type}...")
        if not path.exists():
            print(f"  Skipping (missing file): {path}")
            continue

        problem = load_problem_data(path)
        solution = solve(problem, config)

        base = output_root / f"{inst}_{ev_type}"
        save_solution_pickle(solution, base.with_suffix(".pkl"))
        save_solution_json(solution, base.with_suffix(".json"))
        print(f"  Saved to {base.with_suffix('.pkl')} / {base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
