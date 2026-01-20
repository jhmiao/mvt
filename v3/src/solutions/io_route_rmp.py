from __future__ import annotations

import json
from pathlib import Path

from src.solutions.route_rmp_result import RouteRmpResult


def build_route_rmp_output_path(input_path: Path, output_dir: Path) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    return output_dir / f"{input_path.stem}_route_rmp.json"


def save_route_rmp_json(result: RouteRmpResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
