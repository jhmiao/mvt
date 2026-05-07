import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from src.structures.problem_data import ProblemData
from src.io.cumulative_state import load_cumulative_state

def add_baseline_objectives(
    model: gp.Model,
    problem_data: ProblemData,
    x,
    s,
    t,
    alpha,
    beta,
    include_weekly_fairness_penalty_hours: bool = False,
    include_weekly_fairness_penalty_leaders: bool = False,
    include_running_fairness_penalty: bool = False,
    workload_penalty_weight: float = 1.0,
    leaders_penalty_weight: float = 10.0,
    leaders_fairness_type: str = "count",
    cumulative_state_path: Path | None = None,
):
    """
    Add baseline travel objective plus optional fairness penalties.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.

    Returns:
    None
    """
    C_event = problem_data.event_event_costs
    C_home = problem_data.home_event_costs
    C_depot_e = problem_data.event_depot_costs
    C_depot_h = problem_data.home_depot_costs
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    event_cost = gp.quicksum(
        C_event[i, j] * gp.quicksum(x[i, j, d, w] for d in range(days) for w in range(n)) for i in range(m) for j in range(m)
    )

    home_cost = gp.quicksum(
        C_home[w,i] * gp.quicksum((x[i, m, d, w] + x[m, i, d, w]) for d in range(days)) for w in range (n) for i in range(m)
    )
    
    depot_event_cost = gp.quicksum(
        C_depot_e[i] * gp.quicksum((x[m+1, i, d, w] + x[i, m+2, d, w] )for d in range(days) for w in range(n)) for i in range(m)
    )

    depot_home_cost = gp.quicksum(
        C_depot_h[w] * gp.quicksum((x[m+2, m, d, w] + x[m, m+1, d, w]) for d in range(days)) for w in range(n)
    )

    objective = event_cost + home_cost + depot_event_cost + depot_home_cost

    if include_weekly_fairness_penalty_hours:
        fairness_penalty = weekly_fairness_objective_hours(
            model=model,
            problem_data=problem_data,
            x=x,
            penalty_weight=workload_penalty_weight,
        )
        objective += fairness_penalty

    if include_weekly_fairness_penalty_leaders:
        fairness_penalty = weekly_fairness_objective_leaders(
            model=model,
            problem_data=problem_data,
            x=x,
            alpha=alpha,
            beta=beta,
            penalty_weight=leaders_penalty_weight,
            leaders_fairness_type=leaders_fairness_type,
        )
        objective += fairness_penalty

    if include_running_fairness_penalty:
        running_fairness_penalty = running_fairness_objective_leaders(
            model=model,
            problem_data=problem_data,
            x=x,
            alpha=alpha,
            beta=beta,
            penalty_weight=leaders_penalty_weight,
            leaders_fairness_type=leaders_fairness_type,
            cumulative_state_path=cumulative_state_path,
        )
        objective += running_fairness_penalty

    model.setObjective(objective, GRB.MINIMIZE)


def weekly_fairness_objective_hours(
    model: gp.Model,
    problem_data: ProblemData,
    x,
    penalty_weight: float = 1.0,
):
    """
    Add a fairness penalty term to the objective:
    penalty_weight * (max_workload - min_workload)
    where workload is total assigned event minutes for each nurse.

    Parameters:
    model (gp.Model): The Gurobi optimization model.
    problem_data (ProblemData): The data required to build the optimization model.

    Returns:
    None
    """
    C_dur = problem_data.event_durations
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    if n <= 1:
        return gp.LinExpr(0.0)

    workload = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="workload_minutes")
    max_workload = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="max_workload_minutes")
    min_workload = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="min_workload_minutes")

    model.addConstrs(
        (
            workload[w]
            == gp.quicksum(
                C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m + 3) for d in range(days))
                for j in range(m)
            )
            for w in range(n)
        ),
        name="fair_workload_def",
    )

    model.addConstrs((max_workload >= workload[w] for w in range(n)), name="fair_max_workload_lb")
    model.addConstrs((min_workload <= workload[w] for w in range(n)), name="fair_min_workload_ub")
    print(f"Added fairness penalty to objective with weight {penalty_weight}")

    return penalty_weight * (max_workload - min_workload)


def weekly_fairness_objective_leaders(
    model: gp.Model,
    problem_data: ProblemData,
    x,
    alpha,
    beta,
    penalty_weight: float = 10.0,
    leaders_fairness_type: str = "count",
):
    """
    Add weekly fairness penalty for leadership roles.
    """
    n = problem_data.total_nurse

    if n <= 1:
        return gp.LinExpr(0.0)

    leader_metric, metric_name = _add_leader_fairness_metric(
        model=model,
        problem_data=problem_data,
        x=x,
        alpha=alpha,
        leaders_fairness_type=leaders_fairness_type,
        prefix="weekly_fair",
    )
    max_leader_metric = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"weekly_max_{metric_name}")
    min_leader_metric = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"weekly_min_{metric_name}")

    model.addConstrs((max_leader_metric >= leader_metric[w] for w in range(n)), name=f"weekly_fair_max_{metric_name}_lb")
    model.addConstrs((min_leader_metric <= leader_metric[w] for w in range(n)), name=f"weekly_fair_min_{metric_name}_ub")
    print(
        f"Added weekly leadership {leaders_fairness_type} fairness penalty "
        f"to objective with weight {penalty_weight}"
    )

    return penalty_weight * (max_leader_metric - min_leader_metric)


# def running_fairness_objective_hours(
#     model: gp.Model,
#     problem_data: ProblemData,
#     x,
#     penalty_weight: float = 10.0,
#     cumulative_state_path: Path | None = None,
# ):
#     """
#     Add running fairness penalty using historical cumulative hours:
#     penalty_weight * sum_w(k[w] * workload[w]).
#     """
#     n = problem_data.total_nurse
#     m = problem_data.total_event
#     days = problem_data.total_day
#     C_dur = problem_data.event_durations

#     if n <= 0:
#         return gp.LinExpr(0.0)

#     cumu_path = _resolve_cumulative_state_path(cumulative_state_path)
#     state = load_cumulative_state(cumu_path, nurse_count=n)
#     k = np.asarray(state["cumu_hours"], dtype=float)
#     print(f"Loaded cumulative hours for {n} nurses from {cumu_path}: {k}")

#     workload = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="running_workload")
#     model.addConstrs(
#         (
#             workload[w]
#             == gp.quicksum(
#                 C_dur[j] * gp.quicksum(x[i, j, d, w] for i in range(m + 3) for d in range(days))
#                 for j in range(m)
#             )
#             for w in range(n)
#         ),
#         name="running_fair_workload_def",
#     )

#     penalty = gp.quicksum(float(k[w]) * workload[w] for w in range(n))

#     print(f"Added running fairness penalty to objective with weight {penalty_weight}")
#     return penalty_weight * penalty

def running_fairness_objective_leaders(
    model: gp.Model,
    problem_data: ProblemData,
    x,
    alpha, beta,
    penalty_weight: float = 10.0,
    leaders_fairness_type: str = "count",
    cumulative_state_path: Path | None = None,
):
    """
    Add running fairness penalty using historical cumulative leader metrics.
    """
    n = problem_data.total_nurse

    if n <= 0:
        return gp.LinExpr(0.0)

    _validate_leaders_fairness_type(leaders_fairness_type)
    cumu_path = _resolve_cumulative_state_path(cumulative_state_path)
    state = load_cumulative_state(cumu_path, nurse_count=n)
    state_key = "cumu_leaders" if leaders_fairness_type == "count" else "cumu_leader_days"
    k = np.asarray(state[state_key], dtype=float)
    print(f"Loaded cumulative leader {leaders_fairness_type} values for {n} nurses from {cumu_path}: {k}")

    leader_metric, metric_name = _add_leader_fairness_metric(
        model=model,
        problem_data=problem_data,
        x=x,
        alpha=alpha,
        leaders_fairness_type=leaders_fairness_type,
        prefix="running_fair",
    )
    max_leader_metric = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"running_max_{metric_name}")
    min_leader_metric = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"running_min_{metric_name}")
    model.addConstrs(
        (max_leader_metric >= float(k[w]) + leader_metric[w] for w in range(n)),
        name=f"running_fair_max_{metric_name}_lb",
    )
    model.addConstrs(
        (min_leader_metric <= float(k[w]) + leader_metric[w] for w in range(n)),
        name=f"running_fair_min_{metric_name}_ub",
    )

    print(
        f"Added running leadership {leaders_fairness_type} fairness penalty "
        f"to objective with weight {penalty_weight}"
    )
    return penalty_weight * (max_leader_metric - min_leader_metric)


def _add_leader_fairness_metric(
    model: gp.Model,
    problem_data: ProblemData,
    x,
    alpha,
    leaders_fairness_type: str,
    prefix: str,
):
    _validate_leaders_fairness_type(leaders_fairness_type)
    n = problem_data.total_nurse
    m = problem_data.total_event
    days = problem_data.total_day

    if leaders_fairness_type == "count":
        leader_count = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name=f"{prefix}_leader_count")
        model.addConstrs(
            (
                leader_count[w]
                == gp.quicksum(alpha[j, d, w] for j in range(m) for d in range(days))
                for w in range(n)
            ),
            name=f"{prefix}_leader_count_def",
        )
        return leader_count, "leader_count"

    leader_days = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name=f"{prefix}_leader_days")
    model.addConstrs(
        (
            leader_days[w]
            == gp.quicksum(x[m, m + 1, d, w] for d in range(days))
            for w in range(n)
        ),
        name=f"{prefix}_leader_days_def",
    )
    return leader_days, "leader_days"


def _validate_leaders_fairness_type(leaders_fairness_type: str) -> None:
    if leaders_fairness_type not in {"count", "day"}:
        raise ValueError(
            f"Unsupported leaders_fairness_type {leaders_fairness_type!r}; expected 'count' or 'day'"
        )


def _resolve_cumulative_state_path(cumulative_state_path: Path | None) -> Path:
    if cumulative_state_path is not None:
        return Path(cumulative_state_path)
    return Path(__file__).resolve().parents[2] / "outputs" / "weeks" / "cumu_hours.xlsx"
