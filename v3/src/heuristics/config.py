from dataclasses import dataclass
from typing import Optional

@dataclass
class HeuristicConfig:
    n_initial: int = 100
    top_k: int = 10
    improvement_rounds: int = 5
    seed: int = 0
    num_samples: int = 10000
    work_limit: Optional[float] = None  # Gurobi work units (leave None to disable)
    time_limit: Optional[float] = None  # seconds (leave None to disable)
    seed: int = 42
    gurobi_outputflag: int = 1
