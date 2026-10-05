from __future__ import annotations

import gurobipy as gp
import pytest
from gurobipy import GRB

from src.solver.objectives import _fairness_dispersion_penalty


def _solve_penalty(values: list[float], weight: float, aggregation: str) -> float:
    model = gp.Model()
    model.Params.OutputFlag = 0
    metric = model.addVars(len(values), lb=0.0, name="metric")
    for index, value in enumerate(values):
        metric[index].LB = value
        metric[index].UB = value
    penalty = _fairness_dispersion_penalty(
        model,
        metric,
        len(values),
        weight,
        aggregation,
        prefix="test",
    )
    model.setObjective(penalty, GRB.MINIMIZE)
    model.optimize()
    objective = float(model.ObjVal)
    model.dispose()
    return objective


def test_mean_absolute_deviation_matches_fast_heuristic_formula() -> None:
    # mean([0, 0, 120]) = 40; absolute deviations sum to 160.
    assert _solve_penalty([0.0, 0.0, 120.0], 2.0, "mean_absolute_deviation") == pytest.approx(320.0)


def test_range_aggregation_remains_available() -> None:
    assert _solve_penalty([0.0, 0.0, 120.0], 2.0, "range") == pytest.approx(240.0)
