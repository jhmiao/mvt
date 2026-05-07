from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve four weekly instances without depot, then add depot trips by heuristic."
    )
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
        default="4week_add_depot",
        help="Prefix used for cumulative state and output files.",
    )
    parser.add_argument("--sample-k", type=int, default=None, help="Optional number of events to sample from each instance.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional random seed for instance sampling.")
    parser.add_argument("--work-limit", type=float, default=None, help="Gurobi WorkLimit for the first optimization")
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit in seconds for the first optimization")
    parser.add_argument("--gurobi-output", type=int, default=1, help="Gurobi OutputFlag (0=quiet, 1=verbose)")
    parser.add_argument(
        "--depot-gurobi-output",
        type=int,
        default=0,
        help="Gurobi OutputFlag for the depot insertion model.",
    )
    parser.add_argument(
        "--include-weekly-fairness-penalty-hours",
        action="store_true",
        default=False,
        help="Enable weekly fairness penalty on hours.",
    )
    parser.add_argument(
        "--include-weekly-fairness-penalty-leaders",
        action="store_true",
        default=False,
        help="Enable weekly fairness penalty on leaders.",
    )
    parser.add_argument(
        "--include-running-fairness-penalty",
        action="store_true",
        default=False,
        help="Enable running fairness penalty.",
    )
    parser.add_argument(
        "--workload-penalty-weight",
        type=float,
        default=1.0,
        help="Weight for workload fairness penalty term.",
    )
    parser.add_argument(
        "--leaders-penalty-weight",
        type=float,
        default=10.0,
        help="Weight for leadership fairness penalty term.",
    )

    return parser.parse_args()


def _resolve_instance_path(project_root: Path, instance_name: str) -> Path:
    candidate = Path(instance_name)
    if candidate.suffix != ".xlsx":
        candidate = candidate.with_suffix(".xlsx")
    if candidate.is_absolute():
        return candidate
    return project_root / "data" / "cleaned" / "weeks" / candidate.name


def _add_depot_run_name(run_name: str) -> str:
    if "add_depot" in run_name.lower():
        return run_name
    return f"{run_name}_add_depot"


def _build_run_output_dir(
    output_root: Path,
    run_name: str,
    include_weekly_hours: bool,
    include_weekly_leaders: bool,
    include_running: bool,
) -> Path:
    folder_parts = [_add_depot_run_name(run_name)]
    if include_weekly_hours:
        folder_parts.append("weekly_hours")
    if include_weekly_leaders:
        folder_parts.append("weekly_leaders")
    if include_running:
        folder_parts.append("running")
    return output_root / "_".join(folder_parts)


def _load_week_problem(instance_path: Path, sample_k: int | None, sample_seed: int | None):
    from src.io.data_loader import load_problem_data

    problem = load_problem_data(instance_path, sample_k=sample_k, sample_seed=sample_seed)
    if sample_k is not None:
        print(f"Loaded {instance_path.name} with sampled events {problem.original_event_ids}")
    return problem


def _save_solved_variables(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sol_count = int(getattr(model, "SolCount", 0))
    payload = {
        "status": int(getattr(model, "Status", -1)),
        "objective_value": None,
        "sol_count": sol_count,
        "variables": {},
    }

    if sol_count > 0:
        try:
            payload["objective_value"] = float(model.ObjVal)
        except Exception:
            payload["objective_value"] = None

        decision_prefixes = ("x[", "s[", "t[", "alpha[", "beta[")
        payload["variables"] = {
            var.VarName: float(var.X)
            for var in model.getVars()
            if var.VarName.startswith(decision_prefixes) and abs(float(var.X)) > 1e-6
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or (project_root / "outputs" / "weeks")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_output_dir = _build_run_output_dir(
        output_root,
        args.run_name,
        args.include_weekly_fairness_penalty_hours,
        args.include_weekly_fairness_penalty_leaders,
        args.include_running_fairness_penalty,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.heuristics.add_depot import add_depot  # noqa: E402
    from src.io.cumulative_state import (  # noqa: E402
        initialize_cumulative_state,
        load_cumulative_state,
        save_cumulative_state,
        update_cumulative_state_from_metrics,
    )
    from src.solver.config import SolverConfig  # noqa: E402
    from src.solver.model_builder import build_model  # noqa: E402
    from src.solver.solver_runner import optimize_model  # noqa: E402
    from src.solutions.io import (  # noqa: E402
        save_solution_json,
        save_solution_pickle,
        save_solution_synopsis_json,
    )
    from src.solutions.solution_utils import extract_solution  # noqa: E402

    instance_paths = [_resolve_instance_path(project_root, name) for name in args.instances]
    for instance_path in instance_paths:
        if not instance_path.exists():
            raise FileNotFoundError(f"Data file not found: {instance_path}")

    first_problem = _load_week_problem(instance_paths[0], args.sample_k, args.sample_seed)
    add_depot_run_name = _add_depot_run_name(args.run_name)
    cumulative_path = run_output_dir / f"{add_depot_run_name}_cumulative.json"
    initialize_cumulative_state(cumulative_path, first_problem.total_nurse)

    for week_idx, instance_path in enumerate(instance_paths, start=1):
        problem = first_problem if week_idx == 1 else _load_week_problem(instance_path, args.sample_k, args.sample_seed)
        if problem.total_nurse != first_problem.total_nurse:
            raise ValueError(
                f"Nurse count changed at week {week_idx}: expected {first_problem.total_nurse}, found {problem.total_nurse}"
            )

        config = SolverConfig(
            solve_by_day=False,
            include_depot=False,
            include_weekly_fairness_penalty_hours=args.include_weekly_fairness_penalty_hours,
            include_weekly_fairness_penalty_leaders=False,
            include_running_fairness_penalty=False,
            workload_penalty_weight=args.workload_penalty_weight,
            leaders_penalty_weight=args.leaders_penalty_weight,
            enforce_hour_balance=False,
            use_warmstart=False,
            half_hour_starts=True,
            gurobi_outputflag=args.gurobi_output,
            work_limit=args.work_limit,
            time_limit=args.time_limit,
            cumulative_state_path=cumulative_path,
        )

        base_name = f"{add_depot_run_name}_week{week_idx}_{instance_path.stem}"
        model = build_model(problem, config)
        optimize_model(model)

        nodepot_solution = extract_solution(model, problem)
        variables_path = run_output_dir / f"{base_name}_nodepot_variables.json"
        _save_solved_variables(model, variables_path)

        depot_config = SolverConfig(
            include_weekly_fairness_penalty_leaders=args.include_weekly_fairness_penalty_leaders,
            include_running_fairness_penalty=args.include_running_fairness_penalty,
            leaders_penalty_weight=args.leaders_penalty_weight,
            gurobi_outputflag=args.depot_gurobi_output,
            seed=config.seed,
            cumulative_state_path=cumulative_path,
        )
        solution = add_depot(problem, nodepot_solution, solver_config=depot_config)
        save_solution_pickle(solution, run_output_dir / f"{base_name}.pkl")
        save_solution_json(solution, run_output_dir / f"{base_name}.json")
        save_solution_synopsis_json(solution, run_output_dir / f"{base_name}.json")

        if solution.metrics is None:
            raise RuntimeError(
                f"Week {week_idx} did not produce solution metrics; "
                f"saved synopsis to {run_output_dir}, but cumulative state was not updated."
            )

        state = load_cumulative_state(cumulative_path, nurse_count=problem.total_nurse)
        updated_state = update_cumulative_state_from_metrics(state, solution.metrics)
        save_cumulative_state(cumulative_path, updated_state)

        print(
            f"Week {week_idx}: solved {instance_path.name} without depot, added depot -> "
            f"{run_output_dir / f'{base_name}_synopsis.json'}; "
            f"nodepot variables saved at {variables_path}; cumulative state updated at {cumulative_path}"
        )


if __name__ == "__main__":
    main()


# python v3/src/experiments/run_4week_add_depot.py \
#   --instances c101_Even_5p0std_seed42 c101_Even_5p0std_seed43 c101_Even_5p0std_seed44 c101_Even_5p0std_seed45 \
#   --run-name c101_even_4week \
#   --time-limit 60
