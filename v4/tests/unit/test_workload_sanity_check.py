import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.workload_sanity_check import compute_baseline_cost
from src.structures.problem_data import ProblemData


def test_compute_baseline_cost_breakdown():
    problem = ProblemData(
        event_event_costs=np.array([[0.0, 10.0], [20.0, 0.0]]),
        home_event_costs=np.array([[100.0, 200.0], [300.0, 400.0]]),
        event_depot_costs=np.array([30.0, 40.0]),
        home_depot_costs=np.array([7.0, 9.0]),
        event_durations=np.array([60.0, 90.0]),
        time_windows=np.ones((2, 1, 2)),
        min_nurses=np.ones((2, 2)),
        total_rn=1,
        total_lvn=1,
        total_nurse=2,
        total_event=2,
        total_day=1,
    )
    m = problem.total_event
    variables = {
        "x[0,1,0,0]": 1.0,
        f"x[{m},0,0,0]": 1.0,
        f"x[1,{m},0,1]": 1.0,
        f"x[{m + 1},1,0,0]": 1.0,
        f"x[0,{m + 2},0,1]": 1.0,
        f"x[{m},{m + 1},0,0]": 1.0,
        f"x[{m + 2},{m},0,1]": 1.0,
        "s[0,0,0]": 123.0,
    }

    result = compute_baseline_cost(problem, variables, objective_value=999.0)

    assert result.event_cost == 10.0
    assert result.home_cost == 500.0
    assert result.depot_event_cost == 70.0
    assert result.depot_home_cost == 16.0
    assert result.baseline_cost == 596.0
    assert result.saved_minus_baseline == 403.0
    assert result.active_x_count == 7
