from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LP-relaxation pick-day experiment.")
    default_project = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=default_project, help="Project root path")
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory (default: <project-root>/outputs/pick_day)")
    parser.add_argument("--instance", type=str, default="c101", help="Instance name (e.g., c101, c201, r101, r201)")
    parser.add_argument("--event-type", type=str, default="Even", help="Event type (Even, Skewed1, Skewed2, Random1, Random2)")
    parser.add_argument("--sample-k", type=int, default=None, help="Optional number of sampled events")
    parser.add_argument("--sample-seed", type=int, default=425, help="Random seed for sampling events")
    parser.add_argument("--work-limit", type=float, default=None, help="Gurobi WorkLimit (None to disable)")
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi TimeLimit in seconds (None to disable)")
    parser.add_argument("--gurobi-output", type=int, default=1, help="Gurobi OutputFlag (0=quiet, 1=verbose)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or (project_root / "outputs" / "pick_day")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.io.data_loader import load_problem_data  # noqa: E402
    from src.solver.config import SolverConfig  # noqa: E402
    from src.solver.pick_day import iterative_pick_day_by_lp_relaxation  # noqa: E402
    from src.solver.solver_runner import solve  # noqa: E402

    data_path = project_root / "data" / "cleaned" / f"{args.instance}_{args.event_type}.xlsx"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    problem = load_problem_data(data_path, sample_k=20, sample_seed=45)
    print(f"Loaded problem with events {problem.original_event_ids}")

    config = SolverConfig(
        solve_by_day=False,
        include_weekly_fairness_penalty_hours=False,
        include_weekly_fairness_penalty_leaders=False,
        enforce_hour_balance=False,
        use_warmstart=False,
        half_hour_starts=True,
        gurobi_outputflag=args.gurobi_output,
        work_limit=args.work_limit,
        time_limit=args.time_limit,
    )

    result = iterative_pick_day_by_lp_relaxation(problem, config, print_s_lp_each_iteration=True)

    tag = f"{args.instance}_{args.event_type}"
    if args.sample_k is not None:
        tag += f"_k{args.sample_k}_seed{args.sample_seed}"

    s_lp_path = output_root / f"{tag}_final_s_lp.npy"
    day_csv_path = output_root / f"{tag}_final_fixed_s_id.csv"
    history_path = output_root / f"{tag}_s_lp_history.npy"

    np.save(s_lp_path, result.final_s_lp)
    np.save(history_path, np.array(result.s_lp_history))

    with day_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "fixed_day", "reason", "final_score"])
        for i in range(result.fixed_day_by_event.shape[0]):
            d = int(result.fixed_day_by_event[i])
            score = float(result.final_s_lp[i, d]) if d >= 0 else 0.0
            writer.writerow([i, d, result.fixed_reason_by_event[i], score])

    print(f"Status: {result.status}")
    print(f"Objective (LP): {result.objective_value}")
    print(f"Iterations: {result.iterations}")
    print(f"Saved final LP s matrix: {s_lp_path}")
    print(f"Saved s_lp history: {history_path}")
    print(f"Saved final fixed table: {day_csv_path}")

    # Final solve on original MIP with heuristic-fixed s_id.
    mip_config = SolverConfig(
        solve_by_day=False,
        include_weekly_fairness_penalty_hours=False,
        include_weekly_fairness_penalty_leaders=False,
        enforce_hour_balance=False,
        use_warmstart=False,
        half_hour_starts=True,
        gurobi_outputflag=args.gurobi_output,
        work_limit=args.work_limit,
        time_limit=args.time_limit,
        fixed_event_days=result.fixed_day_by_event.tolist(),
    )
    mip_solution = solve(problem, mip_config)
    mip_summary_path = output_root / f"{tag}_final_mip_summary.csv"
    with mip_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "objective_value", "lower_bound"])
        writer.writerow([mip_solution.status, mip_solution.objective_value, mip_solution.lower_bound])
    print(f"Final MIP status: {mip_solution.status}")
    print(f"Final MIP objective: {mip_solution.objective_value}")
    print(f"Saved final MIP summary: {mip_summary_path}")


if __name__ == "__main__":
    main()

# python v3/src/experiments/run_pick_day_experiment.py \
#   --instance c101 \
#   --event-type Even \
#   --sample-k 10 \
#   --gurobi-output 0 \
#   --time-limit 30
