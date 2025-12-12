from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SolverConfig:
    solve_by_day: bool = False
    use_warmstart: bool = False
    fairness_objective: bool = False
    half_hour_starts: bool = False
    enforce_max_hours: bool = False
    work_limit: int = 3600
    seed : int = 42