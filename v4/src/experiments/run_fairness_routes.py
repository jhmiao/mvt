"""Run alternating weekly route-pool workload and leader-day fairness repairs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alternating weekly workload and leader-day route fairness repairs.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--file", required=True, help="Path to a cleaned weekly .xlsx instance.")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--workload-penalty-coefficient", type=float, default=1.0)
    parser.add_argument("--leader-day-penalty-coefficient", type=float, default=10.0)
    parser.add_argument("--top-k-nurses", type=int, default=2)
    parser.add_argument("--top-k-leaders", type=int, default=None, help="Defaults to --top-k-nurses.")
    parser.add_argument("--time-limit", type=float, default=None, help="Per-day RMP TimeLimit.")
    parser.add_argument("--work-limit", type=float, default=None, help="Per-day RMP WorkLimit.")
    parser.add_argument("--gurobi-output", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the weekly JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.fairness_heuristics_route import RouteFairnessConfig, RouteWeekHeuristic
    from src.io.data_loader import load_problem_data
    from src.solver.route.pool_builder import RoutePoolConfig

    data_path = Path(args.file).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    problem = load_problem_data(str(data_path))
    pool_config = RoutePoolConfig(
        seed=args.seed,
        sample_two_event_pairs=False,
        max_two_per_day=None,
        include_three_event_routes=True,
        sample_three_event_triples=False,
        max_three_per_day=None,
        max_routes_per_nurse_day=None,
    )
    heuristic = RouteWeekHeuristic(
        problem,
        pool_config,
        RouteFairnessConfig(
            max_iterations=args.max_iterations,
            workload_penalty_coefficient=args.workload_penalty_coefficient,
            leader_day_penalty_coefficient=args.leader_day_penalty_coefficient,
            top_k_nurses=args.top_k_nurses,
            top_k_leaders=args.top_k_leaders,
            time_limit=args.time_limit,
            work_limit=args.work_limit,
            seed=args.seed,
            gurobi_outputflag=args.gurobi_output,
        ),
    )
    result = heuristic.run()
    output = args.output or (project_root / "outputs" / f"{data_path.stem}_route_fairness.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_path": str(data_path),
        "evaluation": asdict(result.evaluation),
        "history": [asdict(record) for record in result.history],
        "daily_results": [day.to_dict() for day in result.daily_results],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    accepted = sum(record.accepted for record in result.history)
    print(
        f"Finished route fairness: {accepted}/{len(result.history)} accepted; "
        f"objective={result.evaluation.objective:.3f}, travel={result.evaluation.travel_cost:.3f}, "
        f"workload_penalty={result.evaluation.workload_penalty:.3f}, "
        f"leader_day_objective={result.evaluation.leader_day_objective:.3f}, "
        f"leader_day_penalty={result.evaluation.leader_day_penalty:.3f}"
    )
    print(f"Saved result to {output}")


if __name__ == "__main__":
    main()


# Example:
# PYTHONPATH=v3 python v3/src/experiments/run_fairness_routes.py \\
#   --file v3/data/cleaned/weeks/c101_Random1_5p0std_seed42.xlsx --max-iterations 20
