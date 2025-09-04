# src/models/problem_data.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np  # make sure numpy is installed and this import succeeds

@dataclass
class ProblemData:
    C_event: np.ndarray
    C_home: np.ndarray
    C_depot_e: np.ndarray
    C_depot_h: np.ndarray
    C_dur: np.ndarray
    time_window: np.ndarray
    min_nurse: np.ndarray
    nurse_type: np.ndarray
    nr: int
    nl: int
    n: int
    m: int
    day: int