from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.solver.discrete_arc.model_builder_disc import BLOCKS_PER_DAY


def _status_to_str(status_code: int) -> str:
    status_lookup = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return status_lookup.get(status_code, str(status_code))


def _parse_index(token: str) -> Any:
    token = token.strip()
    if token.lstrip("-").isdigit():
        return int(token)
    return token


def _parse_x_name(var_name: str) -> Tuple[int, Any, Any, int]:
    inside = var_name[var_name.find("[") + 1 : var_name.find("]")]
    parts = [p.strip() for p in inside.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Unexpected x var format: {var_name}")
    w = _parse_index(parts[0])
    i = _parse_index(parts[1])
    j = _parse_index(parts[2])
    b = _parse_index(parts[3])
    return w, i, j, b


def extract_discrete_arc_result(model: gp.Model) -> Dict[str, Any]:
    """
    Extract solution for discrete-arc model.

    Output format:
      {
        "status": "...",
        "objective": <float or None>,
        "assignments": {
          "<nurse_id>": [
            {"from": i, "to": j, "block": b, "day": d}
          ]
        }
      }
    """
    status_code = model.Status
    status = _status_to_str(status_code)

    objective = None
    try:
        objective = float(model.ObjVal)
    except gp.GurobiError:
        objective = None

    sol_count = getattr(model, "SolCount", 0)
    if sol_count == 0 or status_code in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}:
        return {"status": status, "objective": objective, "assignments": {}}

    assignments: Dict[str, List[Dict[str, Any]]] = {}

    for var in model.getVars():
        if not var.VarName.startswith("x["):
            continue
        if var.X <= 0.5:
            continue
        w, i, j, b = _parse_x_name(var.VarName)
        day = b // BLOCKS_PER_DAY
        nurse_key = str(w)
        assignments.setdefault(nurse_key, []).append(
            {"from": i, "to": j, "block": b, "day": day}
        )

    for nurse_key in assignments:
        assignments[nurse_key].sort(key=lambda item: (item["day"], item["block"], str(item["to"])))

    return {"status": status, "objective": objective, "assignments": assignments}


def build_discrete_arc_output_path(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_disc_arc.json"


def save_discrete_arc_json(result: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
