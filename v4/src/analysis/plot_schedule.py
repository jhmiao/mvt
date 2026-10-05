"""Plot a daily nurse schedule from a solution JSON and its problem-data workbook.

Example:
    python v4/src/analysis/plot_schedule.py \
        --solution-json v4/outputs/v4_exact_weekly_fairness_w10_l100_mad/r101_Random2.json \
        --day 0

The event bars use the event durations from the ``C_dur`` sheet in the matching
problem-data workbook.  Pickup and dropoff bars identify the leader assigned to
each depot trip; their visual width is configurable because trip durations are
not represented in the solution JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DEPOT_COLOR = "#9E9E9E"
PICKUP_START = 9 * 60
DROPOFF_START = 17 * 60 + 30


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_solution_path() -> Path:
    return (
        _project_root()
        / "outputs"
        / "v4_exact_weekly_fairness_w10_l100_mad"
        / "r101_Random2.json"
    )


def _infer_problem_data_path(solution_path: Path) -> Path:
    """Infer ``data/cleaned/<instance>.xlsx`` from ``<instance>.json``."""
    return _project_root() / "data" / "cleaned" / f"{solution_path.stem}.xlsx"


def _load_event_durations(problem_data_path: Path) -> List[float]:
    """Load C_dur from the project problem-data workbook."""
    durations = pd.read_excel(problem_data_path, sheet_name="C_dur", index_col=0)
    return [float(value) for value in durations.to_numpy().flatten()]


def _load_daily_solution(solution_path: Path, day: int) -> Mapping[str, Any]:
    with solution_path.open(encoding="utf-8") as handle:
        solution = json.load(handle)

    for daily_solution in solution.get("daily_solutions", []):
        if int(daily_solution["day"]) == day:
            return daily_solution
    raise ValueError(f"Day {day} is not present in {solution_path}")


def _as_int_keyed(mapping: Mapping[str, Any]) -> Dict[int, Any]:
    return {int(key): value for key, value in mapping.items()}


def _map_original_event_ids(
    daily_solution: Mapping[str, Any], full_event_count: int
) -> Mapping[str, Any]:
    """Map sampled event and route node IDs to the original workbook indices.

    Home and the two depots follow the events in both index spaces.  The
    returned copy leaves the loaded solution unchanged.
    """
    extra = daily_solution.get("extra") or {}
    original_ids = extra.get("original_event_ids")
    if original_ids is None:
        return daily_solution

    local_count = int(extra.get("local_event_count", len(original_ids)))
    if local_count != len(original_ids):
        raise ValueError("original_event_ids must contain one ID per local event")

    def map_node(node: Any) -> int:
        node = int(node)
        if 0 <= node < local_count:
            return int(original_ids[node])
        if local_count <= node <= local_count + 2:
            return full_event_count + node - local_count
        return node

    mapped = dict(daily_solution)
    for field in ("assignments", "leaders", "start_times"):
        mapped[field] = {
            map_node(event): value
            for event, value in daily_solution.get(field, {}).items()
        }
    mapped_extra = dict(extra)
    mapped_extra["routes"] = {
        nurse: [[map_node(start), map_node(end)] for start, end in arcs]
        for nurse, arcs in extra.get("routes", {}).items()
    }
    if "scheduled_events" in extra:
        mapped_extra["scheduled_events"] = [
            map_node(event) for event in extra["scheduled_events"]
        ]
    # The copy now uses workbook indices; avoid applying the mapping twice.
    mapped_extra["original_event_ids"] = None
    mapped["extra"] = mapped_extra
    return mapped


def _clock_ticks(start: float, end: float) -> tuple[List[int], List[str]]:
    first_hour = int(start // 60)
    last_hour = int(-(-end // 60))  # ceiling division for floats
    ticks = list(range(first_hour * 60, (last_hour + 1) * 60, 60))
    return ticks, [str(tick // 60) for tick in ticks]


def plot_schedule(
    daily_solution: Mapping[str, Any],
    event_durations: Sequence[float],
    title: str,
    output_path: Path,
    depot_block_minutes: float = 30,
) -> None:
    """Draw one lane per assigned nurse and save a daily schedule figure."""
    daily_solution = _map_original_event_ids(daily_solution, len(event_durations))
    assignments = _as_int_keyed(daily_solution.get("assignments", {}))
    leaders = _as_int_keyed(daily_solution.get("leaders", {}))
    start_times = _as_int_keyed(daily_solution.get("start_times", {}))

    scheduled_events = sorted(set(assignments) & set(start_times))
    if not scheduled_events:
        raise ValueError("The selected day has no events with assignments and start times")

    missing_durations = [event for event in scheduled_events if event >= len(event_durations)]
    if missing_durations:
        raise ValueError(f"C_dur has no duration for event(s): {missing_durations}")

    nurse_ids = sorted(
        {
            int(nurse)
            for event in scheduled_events
            for nurse in assignments[event]
        }
        | {
            int(nurse)
            for leader in leaders.values()
            for nurse in leader.values()
            if nurse is not None
        }
    )
    nurse_y = {nurse: index for index, nurse in enumerate(nurse_ids)}

    event_colors = {
        event: plt.colormaps["tab20"].colors[index % 20]
        for index, event in enumerate(scheduled_events)
    }
    event_start = min(PICKUP_START, min(float(start_times[event]) for event in scheduled_events))
    event_end = max(
        DROPOFF_START + depot_block_minutes,
        max(float(start_times[event]) + float(event_durations[event]) for event in scheduled_events),
    )
    x_start = event_start - 15
    x_end = event_end + 15

    figure_height = max(5, 0.42 * len(nurse_ids) + 2.2)
    _, axis = plt.subplots(figsize=(16, figure_height))

    for event in scheduled_events:
        start = float(start_times[event])
        duration = float(event_durations[event])
        leader = leaders.get(event, {})
        leader_nurses = {
            int(nurse) for nurse in (leader.get("pickup"), leader.get("dropoff")) if nurse is not None
        }

        for nurse in assignments[event]:
            nurse = int(nurse)
            y = nurse_y[nurse]
            is_leader = nurse in leader_nurses
            axis.barh(
                y,
                duration,
                left=start,
                height=0.68,
                color=event_colors[event],
                edgecolor="black" if is_leader else "white",
                linewidth=1.8 if is_leader else 0.8,
            )
            axis.text(start + duration / 2, y, f"E{event}", ha="center", va="center", fontsize=8, color="white")

        pickup_nurse = leader.get("pickup")
        if pickup_nurse is not None:
            pickup_nurse = int(pickup_nurse)
            y = nurse_y[pickup_nurse]
            axis.barh(
                y,
                depot_block_minutes,
                left=PICKUP_START,
                height=0.68,
                color=DEPOT_COLOR,
                edgecolor="white",
            )
            axis.text(
                PICKUP_START + depot_block_minutes / 2,
                y,
                "Pick up",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
            )

        dropoff_nurse = leader.get("dropoff")
        if dropoff_nurse is not None:
            dropoff_nurse = int(dropoff_nurse)
            y = nurse_y[dropoff_nurse]
            axis.barh(
                y,
                depot_block_minutes,
                left=DROPOFF_START,
                height=0.68,
                color=DEPOT_COLOR,
                edgecolor="white",
            )
            axis.text(
                DROPOFF_START + depot_block_minutes / 2,
                y,
                "Drop off",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
            )

    ticks, tick_labels = _clock_ticks(x_start, x_end)
    axis.set_xlim(x_start, x_end)
    axis.set_xticks(ticks, tick_labels)
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xlabel("Time of day")
    axis.set_yticks(range(len(nurse_ids)), [f"Nurse {nurse}" for nurse in nurse_ids])
    axis.invert_yaxis()
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_title(title, fontsize=15, fontweight="bold", pad=28)
    axis.legend(
        handles=[
            Patch(color=DEPOT_COLOR, label="Depot pick up / drop off"),
            Patch(facecolor="white", edgecolor="black", linewidth=1.8, label="Leader event"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
    )
    figure = axis.get_figure()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a daily nurse-event schedule.")
    parser.add_argument("--solution-json", type=Path, default=_default_solution_path())
    parser.add_argument(
        "--problem-data",
        type=Path,
        help="Problem-data .xlsx file; defaults to data/cleaned/<solution-name>.xlsx.",
    )
    parser.add_argument("--day", type=int, default=0, help="Day index to plot (default: 0 / Monday).")
    parser.add_argument(
        "--depot-block-minutes",
        type=float,
        default=30,
        help="Visual width of pickup and dropoff blocks (default: 30 minutes).",
    )
    parser.add_argument("--output", type=Path, help="Destination PNG path.")
    args = parser.parse_args()

    if args.depot_block_minutes <= 0:
        parser.error("--depot-block-minutes must be positive")

    solution_path = args.solution_json.resolve()
    problem_data_path = (args.problem_data or _infer_problem_data_path(solution_path)).resolve()
    output_path = args.output or solution_path.with_name(
        f"{solution_path.stem}_day{args.day}_schedule.png"
    )
    daily_solution = _load_daily_solution(solution_path, args.day)
    event_durations = _load_event_durations(problem_data_path)
    day_name = DAY_NAMES[args.day] if 0 <= args.day < len(DAY_NAMES) else f"Day {args.day + 1}"
    plot_schedule(
        daily_solution,
        event_durations,
        title=f"{solution_path.stem} — {day_name}",
        output_path=output_path,
        depot_block_minutes=args.depot_block_minutes,
    )
    print(f"Saved schedule plot to {output_path}")


if __name__ == "__main__":
    main()
