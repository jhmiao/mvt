

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


CUMULATIVE_FIELDS = ["cumu_hours", "cumu_leaders", "cumu_leader_days"]
SUMMARY_STATS = ["max", "min", "mean", "median", "std"]
PENALTY_VALUE_RE = r"\d+(?:\.\d+)?"
PENALTY_FOLDER_RE = re.compile(
    rf"^weeks_(?P<hour_penalty>{PENALTY_VALUE_RE})_(?P<leader_penalty>{PENALTY_VALUE_RE})(?:_|$)"
)


def _get_outputs_folder(folder_name: str, outputs_dir: str | Path | None = None) -> Path:
    """Return v3/outputs/<folder_name>, or outputs_dir/<folder_name> if supplied."""
    if outputs_dir is None:
        outputs_dir = Path(__file__).resolve().parent / "outputs"
    else:
        outputs_dir = Path(outputs_dir)

    folder = outputs_dir / folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Output folder not found: {folder}")

    return folder


def _parse_penalties_from_folder_name(folder_name: str) -> tuple[float, float]:
    """Extract hour and leader penalties from folder names like weeks_25_1000 or weeks_0.1_0.5."""
    match = PENALTY_FOLDER_RE.match(folder_name)
    if not match:
        raise ValueError(
            "folder_name must start with 'weeks_<hour_penalty>_<leader_penalty>', "
            f"got: {folder_name}"
        )

    return float(match["hour_penalty"]), float(match["leader_penalty"])


def _read_cumulative_records(json_path: Path) -> list[dict]:
    """Read one *_cumulative.json file and return a list of nurse records."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Support both a raw mapping of nurse_id -> values and a wrapper dict.
        for key in ["records", "nurses", "data", "cumulative"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        if all(isinstance(v, dict) for v in data.values()):
            records = []
            for nurse_id, values in data.items():
                record = dict(values)
                record.setdefault("nurse_id", nurse_id)
                records.append(record)
            return records

        # Handle columnar format: {"nurse_id": [...], "cumu_leaders": [...], ...}
        if "nurse_id" in data and isinstance(data["nurse_id"], list):
            records = []
            num_nurses = len(data["nurse_id"])
            for i in range(num_nurses):
                record = {}
                for field, values in data.items():
                    if isinstance(values, list) and len(values) == num_nurses:
                        record[field] = values[i]
                    elif not isinstance(values, list):
                        record[field] = values
                records.append(record)
            return records

        if all(field in data for field in CUMULATIVE_FIELDS):
            return [data]

    raise ValueError(f"Unsupported JSON structure in {json_path}")


def _summarize_records(records: Iterable[dict]) -> dict[str, float]:
    """Compute max, min, mean, and median for cumulative nurse fields."""
    df = pd.DataFrame(records)

    # Only use fields that are actually present in the data
    available_fields = [field for field in CUMULATIVE_FIELDS if field in df.columns]
    if not available_fields:
        raise ValueError(f"No cumulative fields found. Expected one of: {CUMULATIVE_FIELDS}")

    summary: dict[str, float] = {}
    for field in available_fields:
        values = pd.to_numeric(df[field], errors="coerce").dropna()
        for stat in SUMMARY_STATS:
            summary[f"{field}_{stat}"] = getattr(values, stat)()
        summary[f"{field}_diff"] = summary[f"{field}_max"] - summary[f"{field}_min"]

    return summary


def parse_cumulative_outputs(
    folder_name: str,
    outputs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Parse all outputs/<folder_name>/<subfolder>/*_cumulative.json files.

    Returns a DataFrame with one row per cumulative file. The first column is
    the instance name, defined as the subfolder name in outputs/<folder_name>/.
    """
    folder = _get_outputs_folder(folder_name, outputs_dir)
    json_files = sorted(folder.glob("*/*_cumulative.json"))

    rows = []
    all_fields = set()
    for json_path in json_files:
        instance = json_path.parent.name
        records = _read_cumulative_records(json_path)
        row = {"instance": instance}
        summary = _summarize_records(records)
        row.update(summary)
        rows.append(row)
        all_fields.update(summary.keys())

    # Build columns dynamically based on fields found
    columns = ["instance"] + sorted(all_fields)
    return pd.DataFrame(rows, columns=columns)



def _read_synopsis_file(json_path: Path) -> float | None:
    """Read a synopsis JSON file and extract travel_cost_total from metrics."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "metrics" in data:
            metrics = data["metrics"]
            if isinstance(metrics, dict) and "travel_cost_total" in metrics:
                return float(metrics["travel_cost_total"])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    
    return None


def parse_synopsis_outputs(
    folder_name: str,
    outputs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Parse all outputs/<folder_name>/<subfolder>/*_synopsis.json files.
    
    For each subfolder, sums the travel_cost_total values from all synopsis files
    (skipping files with 'lpen10p0' in the name if there are 4 or more files).
    
    Returns a DataFrame with 'instance' (subfolder name) and 'total_travel' columns.
    """
    folder = _get_outputs_folder(folder_name, outputs_dir)
    
    rows = []
    for subfolder in sorted(folder.iterdir()):
        if not subfolder.is_dir():
            continue
        
        synopsis_files = sorted(subfolder.glob("*_synopsis.json"))
        
        if not synopsis_files:
            continue
        
        # If a folder has multiple leader-penalty variants, prefer the
        # non-lpen100 files. Some baseline folders only have lpen100 files,
        # so keep them instead of filtering the folder down to nothing.
        non_lpen100_files = [f for f in synopsis_files if "lpen100" not in f.name]
        if len(synopsis_files) >= 4 and non_lpen100_files:
            synopsis_files = non_lpen100_files
        
        total_travel = 0.0
        count = 0
        for json_path in synopsis_files:
            travel_cost = _read_synopsis_file(json_path)
            if travel_cost is not None:
                total_travel += travel_cost
                count += 1
        
        if count > 0:
            rows.append({
                "instance": subfolder.name,
                "total_travel": total_travel
            })
    
    return pd.DataFrame(rows)


def main(folder_name: str = "weeks_10_1000") -> None:
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    
    # Parse cumulative outputs
    cumulative_df = parse_cumulative_outputs(folder_name, outputs_dir)
    
    # Parse synopsis outputs
    synopsis_df = parse_synopsis_outputs(folder_name, outputs_dir)

    summary_csv = outputs_dir / f"{folder_name}_summary.csv"
    summary_df = cumulative_df.merge(synopsis_df, on="instance", how="left")
    hour_penalty, leader_penalty = _parse_penalties_from_folder_name(folder_name)
    summary_df.insert(1, "hour_penalty", hour_penalty)
    summary_df.insert(2, "leader_penalty", leader_penalty)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved {len(summary_df)} rows to {summary_csv}")


if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else "weeks_0.1_0.5"
    main(folder_name)
