"""Run fast travel-only route repairs with weekly fairness SA acceptance."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fast roulette-based route-pool fairness ALNS with travel-only repairs."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--file", required=True, help="Path to a cleaned weekly .xlsx instance.")
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument("--wall-time-limit", type=float, default=None, help="Search wall-clock limit in seconds.")
    parser.add_argument("--workload-penalty-coefficient", type=float, default=1.0)
    parser.add_argument("--leader-day-penalty-coefficient", type=float, default=400.0)
    parser.add_argument("--top-k-nurses", type=int, default=3)
    parser.add_argument("--top-k-leaders", type=int, default=None, help="Defaults to --top-k-nurses.")
    parser.add_argument("--uniform-selection", action="store_true", help="Select uniformly within top-k instead of rank bias.")
    parser.add_argument(
        "--workload-route-restriction-factor", type=float, default=1.0,
        help="Forbid workload routes at or above this fraction of the incumbent route workload.",
    )
    parser.add_argument("--repair-time-limit", type=float, default=0.25, help="Strict per-repair Gurobi TimeLimit.")
    parser.add_argument("--repair-work-limit", type=float, default=None, help="Strict per-repair Gurobi WorkLimit.")
    parser.add_argument("--repair-solution-limit", type=int, default=1, help="Use 1 to stop at first feasible repair.")
    parser.add_argument("--initial-time-limit", type=float, default=None, help="Optional limit for initial daily travel solves.")
    parser.add_argument("--initial-work-limit", type=float, default=None)
    parser.add_argument("--initial-solution-limit", type=int, default=None)
    parser.add_argument("--initial-temperature", type=float, default=0.05, help="SA temperature for relative deterioration.")
    parser.add_argument("--cooling-rate", type=float, default=0.995)
    parser.add_argument("--gurobi-output", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.fairness_heuristics_route import FastRouteFairnessConfig, FastRouteWeekHeuristic
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
        workload_fairness=False,
    )
    result = FastRouteWeekHeuristic(
        problem,
        pool_config,
        FastRouteFairnessConfig(
            max_iterations=args.max_iterations,
            wall_time_limit=args.wall_time_limit,
            workload_penalty_coefficient=args.workload_penalty_coefficient,
            leader_day_penalty_coefficient=args.leader_day_penalty_coefficient,
            top_k_nurses=args.top_k_nurses,
            top_k_leaders=args.top_k_leaders,
            rank_biased_selection=not args.uniform_selection,
            workload_route_restriction_factor=args.workload_route_restriction_factor,
            repair_time_limit=args.repair_time_limit,
            repair_work_limit=args.repair_work_limit,
            repair_solution_limit=args.repair_solution_limit,
            initial_time_limit=args.initial_time_limit,
            initial_work_limit=args.initial_work_limit,
            initial_solution_limit=args.initial_solution_limit,
            initial_temperature=args.initial_temperature,
            cooling_rate=args.cooling_rate,
            seed=args.seed,
            gurobi_outputflag=args.gurobi_output,
        ),
    ).run()

    output = args.output or (project_root / "outputs" / f"{data_path.stem}_fast_route_fairness.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_path": str(data_path),
        "best_evaluation": asdict(result.evaluation),
        "current_evaluation": asdict(result.current_evaluation),
        "history": [asdict(record) for record in result.history],
        "operator_stats": {
            name: {**asdict(stats), "average_repair_seconds": stats.average_repair_seconds}
            for name, stats in result.operator_stats.items()
        },
        "daily_results": [day.to_dict() for day in result.daily_results],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    accepted = sum(record.accepted for record in result.history)
    print(
        f"Finished fast route fairness: {accepted}/{len(result.history)} accepted; "
        f"best_Z={result.evaluation.objective:.3f}, travel={result.evaluation.travel_cost:.3f}, "
        f"workload_penalty={result.evaluation.workload_penalty:.3f}, "
        f"leader_day_penalty={result.evaluation.leader_day_penalty:.3f}"
    )
    print(f"Saved result to {output}")


if __name__ == "__main__":
    main()
