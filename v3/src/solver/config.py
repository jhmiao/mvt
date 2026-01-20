from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SolverConfig:
    backend: str = "arc"  # "arc" | "route_pool" | "cg"
    solve_by_day: bool = False
    use_warmstart: bool = False
    fairness_objective: bool = False
    half_hour_starts: bool = False
    enforce_max_hours: bool = False
    enforce_hour_balance: bool = True
    work_limit: Optional[float] = None  # Gurobi work units (leave None to disable)
    time_limit: Optional[float] = None  # seconds (leave None to disable)
    seed: int = 42
    gurobi_outputflag: int = 1

    # route-pool knobs
    routes_per_nurse: int = 500
