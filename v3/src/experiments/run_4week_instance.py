from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve four weekly instances sequentially with cumulative state.")
    default_project = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=default_project, help="Project root path")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory (default: <project-root>/outputs/weeks)")
    parser.add_argument(
        "--instances",
        nargs=4,
        required=True,
        help="Four weekly instance file names or stems, in chronological order.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="4week_run",
        help="Prefix used for cumulative state and output files.",
    )
    parser.add_argument("--work-limit", type=float, default=None, help="Gurobi WorkLimit (None to disable)")
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit in seconds (None to disable)")
    parser.add_argument("--gurobi-output", type=int, default=1, help="Gurobi OutputFlag (0=quiet, 1=verbose)")
    parser.add_argument(
        "--include-running-fairness-penalty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable running fairness penalty.",
    )
    parser.add_argument(
        "--include-fairness-penalty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable myopic fairness penalty.",
    )
    parser.add_argument(
        "--fairness-penalty-weight",
        type=float,
        default=1.0,
        help="Weight for fairness penalty term.",
    )
    return parser.parse_args()


def _resolve_instance_path(project_root: Path, instance_name: str) -> Path:
    candidate = Path(instance_name)
    if candidate.suffix != ".xlsx":
        candidate = candidate.with_suffix(".xlsx")
    if candidate.is_absolute():
        return candidate
    return project_root / "data" / "cleaned" / "weeks" / candidate.name


def _build_run_output_dir(output_root: Path, run_name: str, include_running: bool, include_fairness: bool) -> Path:
    folder_parts = [run_name]
    if include_running:
        folder_parts.append("running")
    if include_fairness:
        folder_parts.append("myopic")
    return output_root / "_".join(folder_parts)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or (project_root / "outputs" / "weeks")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_output_dir = _build_run_output_dir(
        output_root,
        args.run_name,
        args.include_running_fairness_penalty,
        args.include_fairness_penalty,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.io.cumulative_state import (  # noqa: E402
        initialize_cumulative_state,
        load_cumulative_state,
        save_cumulative_state,
        update_cumulative_state_from_metrics,
    )
    from src.io.data_loader import load_problem_data  # noqa: E402
    from src.solver.config import SolverConfig  # noqa: E402
    from src.solver.solver_runner import solve  # noqa: E402
    from src.solutions.io import save_solution_synopsis_json  # noqa: E402

    instance_paths = [_resolve_instance_path(project_root, name) for name in args.instances]
    for instance_path in instance_paths:
        if not instance_path.exists():
            raise FileNotFoundError(f"Data file not found: {instance_path}")

    first_problem = load_problem_data(instance_paths[0])
    cumulative_path = run_output_dir / f"{args.run_name}_cumulative.json"
    initialize_cumulative_state(cumulative_path, first_problem.total_nurse)

    for week_idx, instance_path in enumerate(instance_paths, start=1):
        problem = first_problem if week_idx == 1 else load_problem_data(instance_path)
        if problem.total_nurse != first_problem.total_nurse:
            raise ValueError(
                f"Nurse count changed at week {week_idx}: expected {first_problem.total_nurse}, found {problem.total_nurse}"
            )

        config = SolverConfig(
            solve_by_day=False,
            include_weekly_fairness_penalty_hours=args.include_fairness_penalty,
            include_weekly_fairness_penalty_leaders=args.include_fairness_penalty,
            include_running_fairness_penalty=args.include_running_fairness_penalty,
            fairness_penalty_weight=args.fairness_penalty_weight,
            enforce_hour_balance=False,
            use_warmstart=False,
            half_hour_starts=True,
            gurobi_outputflag=args.gurobi_output,
            work_limit=args.work_limit,
            time_limit=args.time_limit,
            cumulative_state_path=cumulative_path,
        )

        solution = solve(problem, config)
        base_name = f"{args.run_name}_week{week_idx}_{instance_path.stem}_penalty{args.fairness_penalty_weight}"
        save_solution_synopsis_json(solution, run_output_dir / f"{base_name}.json")

        state = load_cumulative_state(cumulative_path, nurse_count=problem.total_nurse)
        updated_state = update_cumulative_state_from_metrics(state, solution.metrics)
        save_cumulative_state(cumulative_path, updated_state)

        print(
            f"Week {week_idx}: solved {instance_path.name} -> "
            f"{run_output_dir / f'{base_name}_synopsis.json'}; cumulative state updated at {cumulative_path}"
        )


if __name__ == "__main__":
    main()


# python v3/src/experiments/run_4week_instance.py \
#   --instances c201_Even_5p0std_seed42 c201_Even_5p0std_seed43 c201_Even_5p0std_seed44 c201_Even_5p0std_seed45 \
#   --run-name c201_even_4week \
#   --time-limit 120 \
#   --fairness-penalty-weight 1
#   --include-fairness-penalty
#   --include-running-fairness-penalty
