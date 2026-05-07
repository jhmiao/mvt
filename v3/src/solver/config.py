from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class SolverConfig:
    backend: str = "arc"  # "arc" | "route_pool" | "cg"
    solve_by_day: bool = False
    use_warmstart: bool = False
    include_depot: bool = True
    include_weekly_fairness_penalty_hours: bool = False
    include_weekly_fairness_penalty_leaders: bool = False
    include_running_fairness_penalty: bool = False
    workload_penalty_weight: float = 1.0
    leaders_penalty_weight: float = 10.0
    leaders_fairness_type: Literal["count", "day"] = "count"
    half_hour_starts: bool = True
    enforce_max_hours: bool = False
    enforce_hour_balance: bool = False
    work_limit: Optional[float] = None  # Gurobi work units (leave None to disable)
    time_limit: Optional[float] = None  # seconds (leave None to disable)
    seed: int = 42
    gurobi_outputflag: int = 1
    # If provided, fix s[i,d] by event i -> day fixed_event_days[i] (use -1 for unfixed).
    fixed_event_days: Optional[list[int]] = None
    cumulative_state_path: Optional[Path] = None

    # route-pool knobs
    routes_per_nurse: int = 500
