from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.solutions.io import load_solution_pickle  # noqa: E402


def load_solution(instance: str, event_type: str, outputs_dir: Optional[Path] = None):
    """
    Load a solution pickle for a given instance and type (e.g., 'c101', 'Even').
    """
    if outputs_dir is None:
        outputs_dir = PROJECT_ROOT / "outputs"
    path = outputs_dir / f"{instance}_{event_type}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Solution file not found: {path}")
    return load_solution_pickle(path)


def main():
    instance = sys.argv[1] if len(sys.argv) > 1 else "c101"
    event_type = sys.argv[2] if len(sys.argv) > 2 else "Even"

    sol = load_solution(instance, event_type)
    print(f"Loaded solution for {instance}_{event_type}:")
    print(f"  status: {sol.status}")
    print(f"  objective: {sol.objective_value}")
    print(f"  lower bound: {sol.lower_bound}")
    print(f"  days: {len(sol.daily_solutions)}")


if __name__ == "__main__":
    main()
