from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from src.solutions.solutions import MergedSolution, Solution


def _normalize_assignments(assignments: Dict[Any, Any]) -> Dict[int, List[int]]:
    # Ensure event ids are ints and nurse lists are plain lists
    normalized: Dict[int, List[int]] = {}
    for k, v in assignments.items():
        try:
            evt = int(k)
        except Exception:
            evt = k
        nurses = list(v) if isinstance(v, (list, tuple, set)) else [v]
        normalized[evt] = nurses
    return normalized


def _solution_to_dict(solution: MergedSolution) -> Dict[str, Any]:
    daily = []
    for ds in solution.daily_solutions:
        daily.append(
            {
                "day": ds.day,
                "assignments": _normalize_assignments(ds.assignments),
                "leaders": {int(k): v for k, v in ds.leaders.items()},
                "start_times": {int(k): v for k, v in ds.start_times.items()},
                "objective_value": ds.objective_value,
                "lower_bound": ds.lower_bound,
                "status": ds.status,
                "extra": _serialize_extra(ds.extra),
            }
        )
    return {
        "daily_solutions": daily,
        "objective_value": solution.objective_value,
        "lower_bound": solution.lower_bound,
        "status": solution.status,
        "metrics": solution.metrics,
    }


def _serialize_extra(extra: Any) -> Any:
    if extra is None:
        return None
    if isinstance(extra, dict):
        serialized = {}
        for k, v in extra.items():
            if isinstance(v, set):
                serialized[k] = sorted(v)
            elif isinstance(v, dict):
                serialized[k] = _serialize_extra(v)
            elif isinstance(v, list):
                serialized[k] = [_serialize_extra(i) for i in v]
            elif isinstance(v, tuple):
                serialized[k] = [_serialize_extra(i) for i in v]
            else:
                serialized[k] = v
        return serialized
    return extra


def _dict_to_solution(data: Dict[str, Any]) -> MergedSolution:
    daily_solutions: List[Solution] = []
    for ds in data.get("daily_solutions", []):
        assignments = {int(k): list(v) for k, v in ds.get("assignments", {}).items()}
        leaders = {int(k): v for k, v in ds.get("leaders", {}).items()}
        start_times = {int(k): v for k, v in ds.get("start_times", {}).items()}
        extra = ds.get("extra")
        daily_solutions.append(
            Solution(
                day=ds.get("day"),
                assignments=assignments,
                leaders=leaders,
                start_times=start_times,
                objective_value=ds.get("objective_value"),
                lower_bound=ds.get("lower_bound"),
                status=ds.get("status") or "",
                extra=extra,
            )
        )

    return MergedSolution(
        daily_solutions=daily_solutions,
        objective_value=data.get("objective_value"),
        lower_bound=data.get("lower_bound"),
        status=data.get("status") or "",
        metrics=data.get("metrics"),
    )


def save_solution_pickle(solution: MergedSolution, path: Path) -> None:
    """Save solution to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(solution, f)


def load_solution_pickle(path: Path) -> MergedSolution:
    """Load solution from a pickle file."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_solution_json(solution: MergedSolution, path: Path) -> None:
    """Save solution to a JSON file (serializable structure)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _solution_to_dict(solution)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_solution_json(path: Path) -> MergedSolution:
    """Load solution from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _dict_to_solution(data)
