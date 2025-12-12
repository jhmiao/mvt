import sys
from pathlib import Path

import numpy as np

# Make sure the v3 package root is on sys.path (so we can import src.*)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.io.data_loader import load_problem_data


def test_split_by_day_basic():
    data_path = Path(__file__).resolve().parents[2] / "data" / "cleaned" / "c201_Even.xlsx"

    problem = load_problem_data(data_path)

    days = problem.split_by_day()
    assert len(days) == problem.total_day

    day_event_counts = []
    for day_problem in days:
        k = day_problem.total_event

        # Shapes adjust to the day's active events
        assert day_problem.event_event_costs.shape == (k, k)
        assert day_problem.home_event_costs.shape[1] == k
        assert day_problem.event_depot_costs.shape == (k,)
        assert day_problem.home_depot_costs.shape[0] == problem.total_nurse
        assert day_problem.event_durations.shape == (k,)
        assert day_problem.time_windows.shape == (k, 1, 2)
        assert day_problem.min_nurses.shape == (k, 2)

        events_today = day_problem.original_event_ids
        assert len(events_today) == k
        print(f"Day {day_problem.day_index}: events_today={events_today.tolist()}")

        day_event_counts.append(k)

    # Expect evenly distributed 10 events/day across 5 days for c101_Even
    assert day_event_counts == [10, 10, 10, 10, 10]
