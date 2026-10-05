from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Solution:
    """
    Represents the solution for a single day (or a unified multi-day view).

    - day:          the day index this solution corresponds to (None for unknown/weekly solve)
    - assignments:  {event: [nurse, ...]} mapping each event to the nurses serving it
    - leaders:      {event: {"pickup": nurse, "dropoff": nurse}} leader assignments if applicable
    - start_times:  {event: start_time}

    The same structure is reused when merging multi-day solutions; `day` is kept to preserve origin.
    """

    day: Optional[int] = None
    assignments: Dict[int, Any] = field(default_factory=dict)
    leaders: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    start_times: Dict[int, float] = field(default_factory=dict)

    # Solver-level information
    objective_value: float = 0.0          # objective for the full horizon contained in this Solution
    lower_bound: Optional[float] = None   # solver-provided LB (global LB if multi-day, per-day LB if day-solve)
    status: str = "UNKNOWN"               # e.g., OPTIMAL, TIME_LIMIT, HEURISTIC, INFEASIBLE

    # Optional metadata (runtime info, fairness metrics, feasibility flags, etc.)
    extra: Optional[Dict[str, Any]] = None


@dataclass
class MergedSolution:
    """
    Represents the final weekly (or multi-day) solution.

    - daily_solutions: list of per-day Solution objects OR
                       a list containing one weekly Solution if not decomposed.
    - objective_value: total objective over the entire horizon
    - lower_bound:     global LB if solved as one model;
                       sum of daily LBs if solved day-by-day
    - status:          aggregated status
    """

    daily_solutions: list[Solution]                 # length 1 for weekly solve

    objective_value: float
    lower_bound: Optional[float]
    status: str                                     # e.g. OPTIMAL, MERGED, TIME_LIMIT

    # Optional: fairness metrics, utilization summaries, etc.
    metrics: Optional[Dict[str, Any]] = None
