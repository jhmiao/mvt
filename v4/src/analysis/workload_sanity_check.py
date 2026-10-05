from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


X_VAR_RE = re.compile(r"^x\[(\d+),(\d+),(\d+),(\d+)\]$")


@dataclass
class BaselineCost:
    event_cost: float = 0.0
    home_cost: float = 0.0
    depot_event_cost: float = 0.0
    depot_home_cost: float = 0.0
    ignored_x_count: int = 0
    active_x_count: int = 0
    objective_value: Optional[float] = None

    @property
    def baseline_cost(self) -> float:
        return self.event_cost + self.home_cost + self.depot_event_cost + self.depot_home_cost

    @property
    def saved_minus_baseline(self) -> Optional[float]:
        if self.objective_value is None:
            return None
        return self.objective_value - self.baseline_cost

    def to_dict(self) -> Dict[str, Optional[float] | int]:
        result = asdict(self)
        result["baseline_cost"] = self.baseline_cost
        result["saved_minus_baseline"] = self.saved_minus_baseline
        return result


def parse_x_name(name: str) -> Optional[Tuple[int, int, int, int]]:
    """Return (i, j, d, w) for an x variable name, or None for non-x names."""
    match = X_VAR_RE.match(name)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def iter_active_x_variables(
    variables: Mapping[str, Any],
    tolerance: float,
) -> Iterable[Tuple[str, Tuple[int, int, int, int], float]]:
    for name, raw_value in variables.items():
        indices = parse_x_name(name)
        if indices is None:
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Variable {name!r} has non-numeric value {raw_value!r}") from exc

        if abs(value) > tolerance:
            yield name, indices, value


def compute_baseline_cost(
    problem_data,
    variables: Mapping[str, Any],
    *,
    objective_value: Optional[float] = None,
    tolerance: float = 1e-6,
    strict: bool = True,
) -> BaselineCost:
    """
    Calculate the baseline travel objective from saved x variables.

    This mirrors src.solver.objectives.add_baseline_objectives:
        objective = event_cost + home_cost + depot_event_cost + depot_home_cost
    """
    C_event = problem_data.event_event_costs
    C_home = problem_data.home_event_costs
    C_depot_e = problem_data.event_depot_costs
    C_depot_h = problem_data.home_depot_costs
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    result = BaselineCost(objective_value=objective_value)
    unknown_arcs = []

    for name, (i, j, d, w), value in iter_active_x_variables(variables, tolerance):
        if not (0 <= w < n):
            message = f"{name}: nurse index {w} outside [0, {n})"
            if strict:
                raise ValueError(message)
            unknown_arcs.append(message)
            result.ignored_x_count += 1
            continue
        if not (0 <= d < days):
            message = f"{name}: day index {d} outside [0, {days})"
            if strict:
                raise ValueError(message)
            unknown_arcs.append(message)
            result.ignored_x_count += 1
            continue

        result.active_x_count += 1

        if i < m and j < m:
            result.event_cost += float(C_event[i, j]) * value
        elif i == m and j < m:
            result.home_cost += float(C_home[w, j]) * value
        elif i < m and j == m:
            result.home_cost += float(C_home[w, i]) * value
        elif i == m + 1 and j < m:
            result.depot_event_cost += float(C_depot_e[j]) * value
        elif i < m and j == m + 2:
            result.depot_event_cost += float(C_depot_e[i]) * value
        elif i == m and j == m + 1:
            result.depot_home_cost += float(C_depot_h[w]) * value
        elif i == m + 2 and j == m:
            result.depot_home_cost += float(C_depot_h[w]) * value
        else:
            message = f"{name}: unrecognized arc for total_event={m}"
            if strict:
                raise ValueError(message)
            unknown_arcs.append(message)
            result.ignored_x_count += 1

    if unknown_arcs:
        print(
            f"Warning: ignored {len(unknown_arcs)} x variables; first issue: {unknown_arcs[0]}",
            file=sys.stderr,
        )

    return result


def load_variables_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    variables = payload.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(f"{path} does not contain a variables object")
    return payload


def _resolve_instance_path(project_root: Path, instance: str) -> Path:
    candidate = Path(instance)
    if candidate.suffix != ".xlsx":
        candidate = candidate.with_suffix(".xlsx")
    if candidate.is_absolute():
        return candidate

    search_paths = [
        project_root / candidate,
        project_root / "data" / "cleaned" / "weeks" / candidate.name,
        project_root / "data" / "cleaned" / candidate.name,
    ]
    for path in search_paths:
        if path.exists():
            return path
    return search_paths[1]


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Calculate the baseline travel cost from a saved *_variables.json file."
    )
    parser.add_argument("variables_json", type=Path, help="Saved JSON containing a variables object.")
    parser.add_argument(
        "--instance",
        required=True,
        help="Instance .xlsx path or stem used to create the variables file.",
    )
    parser.add_argument("--project-root", type=Path, default=default_project, help="v3 project root path.")
    parser.add_argument("--sample-k", type=int, default=None, help="Optional event sample size used for the solve.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional event sample seed used for the solve.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Ignore x values with abs(value) <= tolerance.")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Ignore x variables with invalid indices or unrecognized arcs instead of failing.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write the cost breakdown JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.io.data_loader import load_problem_data  # noqa: E402

    variables_path = args.variables_json.resolve()
    instance_path = _resolve_instance_path(project_root, args.instance)
    if not instance_path.exists():
        raise FileNotFoundError(f"Data file not found: {instance_path}")

    payload = load_variables_payload(variables_path)
    saved_objective = payload.get("objective_value")
    if saved_objective is not None:
        saved_objective = float(saved_objective)

    problem_data = load_problem_data(instance_path, sample_k=args.sample_k, sample_seed=args.sample_seed)
    result = compute_baseline_cost(
        problem_data,
        payload["variables"],
        objective_value=saved_objective,
        tolerance=args.tolerance,
        strict=not args.non_strict,
    )
    result_dict = result.to_dict()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
