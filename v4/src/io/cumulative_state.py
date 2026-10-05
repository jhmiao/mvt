from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def initialize_cumulative_state(path: Path, nurse_count: int) -> dict[str, list[float]]:
    state = {
        "nurse_id": list(range(nurse_count)),
        "cumu_hours": [0.0] * nurse_count,
        "cumu_leaders": [0.0] * nurse_count,
        "cumu_leader_days": [0.0] * nurse_count,
    }
    save_cumulative_state(path, state)
    return state


def load_cumulative_state(path: Path, nurse_count: int | None = None) -> dict[str, list[float]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        state = _normalize_state(payload)
    elif suffix in {".xlsx", ".xls"}:
        state = _load_cumulative_state_from_excel(path)
    else:
        raise ValueError(f"Unsupported cumulative state file: {path}")

    if nurse_count is not None:
        state = _validate_nurse_count(state, nurse_count, path)
    return state


def save_cumulative_state(path: Path, state: dict[str, list[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    normalized = _normalize_state(state)

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)
        return

    if suffix in {".xlsx", ".xls"}:
        df = pd.DataFrame(normalized)
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, index=False, sheet_name="cumulative_state")
        return

    raise ValueError(f"Unsupported cumulative state file: {path}")


def update_cumulative_state_from_metrics(
    state: dict[str, list[float]],
    metrics: dict[str, Any] | None,
) -> dict[str, list[float]]:
    if metrics is None:
        raise ValueError("Solution metrics are missing; cannot update cumulative state.")

    hours = [float(v) for v in metrics.get("working_hours_by_nurse", [])]
    leaders = [float(v) for v in metrics.get("leader_count_by_nurse", [])]
    leader_days = [float(v) for v in metrics.get("leader_days_by_nurse", [])]
    nurse_count = len(state["nurse_id"])
    if len(hours) != nurse_count:
        raise ValueError(
            f"working_hours_by_nurse length {len(hours)} does not match cumulative nurse count {nurse_count}"
        )
    if len(leaders) != nurse_count:
        raise ValueError(
            f"leader_count_by_nurse length {len(leaders)} does not match cumulative nurse count {nurse_count}"
        )
    if len(leader_days) != nurse_count:
        raise ValueError(
            f"leader_days_by_nurse length {len(leader_days)} does not match cumulative nurse count {nurse_count}"
        )

    return {
        "nurse_id": list(state["nurse_id"]),
        "cumu_hours": [float(prev) + hours[idx] for idx, prev in enumerate(state["cumu_hours"])],
        "cumu_leaders": [float(prev) + leaders[idx] for idx, prev in enumerate(state["cumu_leaders"])],
        "cumu_leader_days": [
            float(prev) + leader_days[idx] for idx, prev in enumerate(state["cumu_leader_days"])
        ],
    }


def _normalize_state(payload: dict[str, Any]) -> dict[str, list[float]]:
    nurse_ids = [int(v) for v in payload.get("nurse_id", [])]
    cumu_hours = [float(v) for v in payload.get("cumu_hours", [])]
    cumu_leaders = [float(v) for v in payload.get("cumu_leaders", [])]
    cumu_leader_days = [float(v) for v in payload.get("cumu_leader_days", [0.0] * len(nurse_ids))]

    if not (len(nurse_ids) == len(cumu_hours) == len(cumu_leaders) == len(cumu_leader_days)):
        raise ValueError("cumulative state arrays must have identical lengths")

    return {
        "nurse_id": nurse_ids,
        "cumu_hours": cumu_hours,
        "cumu_leaders": cumu_leaders,
        "cumu_leader_days": cumu_leader_days,
    }


def _validate_nurse_count(
    state: dict[str, list[float]],
    nurse_count: int,
    path: Path,
) -> dict[str, list[float]]:
    actual = len(state["nurse_id"])
    if actual != nurse_count:
        raise ValueError(
            f"Cumulative state nurse count mismatch for {path}: expected {nurse_count}, found {actual}"
        )
    return state


def _load_cumulative_state_from_excel(path: Path) -> dict[str, list[float]]:
    try:
        df = pd.read_excel(path, sheet_name="cumulative_state")
        if {"nurse_id", "cumu_hours", "cumu_leaders"}.issubset(df.columns):
            return _normalize_state(df.to_dict(orient="list"))
    except ValueError:
        pass

    hours_df = pd.read_excel(path, sheet_name="working_hours_by_nurse")
    leaders_df = pd.read_excel(path, sheet_name="leaders_by_nurse")
    if "Index" not in hours_df.columns or "Cumulative" not in hours_df.columns:
        raise ValueError(f"Missing expected columns in {path} sheet working_hours_by_nurse")
    if "Index" not in leaders_df.columns or "Cumulative" not in leaders_df.columns:
        raise ValueError(f"Missing expected columns in {path} sheet leaders_by_nurse")

    merged = hours_df[["Index", "Cumulative"]].rename(
        columns={"Index": "nurse_id", "Cumulative": "cumu_hours"}
    )
    merged["cumu_leaders"] = leaders_df["Cumulative"]
    try:
        leader_days_df = pd.read_excel(path, sheet_name="leader_days_by_nurse")
        if "Cumulative" not in leader_days_df.columns:
            raise ValueError(f"Missing expected columns in {path} sheet leader_days_by_nurse")
        merged["cumu_leader_days"] = leader_days_df["Cumulative"]
    except ValueError:
        merged["cumu_leader_days"] = 0.0
    return _normalize_state(merged.to_dict(orient="list"))
