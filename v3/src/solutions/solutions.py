from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class Solution:
    day: Optional[int]
    assignments: dict
    leaders: dict
    start_times: dict
    objective_value: float
    lower_bound: Optional[float]
    status: str
    extra: Optional[dict] = None

@dataclass
class MergedSolution:
    daily_solutions: list[Solution]   # length 1 for weekly solve
    objective_value: float
    lower_bound: Optional[float]
    status: str
    metrics: Optional[dict] = None
