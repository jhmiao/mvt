# src/models/problem_data.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ProblemData:
    event_event_costs: np.ndarray
    home_event_costs: np.ndarray
    event_depot_costs: np.ndarray
    home_depot_costs: np.ndarray
    event_durations: np.ndarray
    time_windows: np.ndarray
    min_nurses: np.ndarray
    # nurse_types: np.ndarray
    total_rn: int
    total_lvn: int
    total_nurse: int
    total_event: int
    total_day: int
    max_hours: np.ndarray = None
    day_index: int = -1  # Optional, indicates which day this instance corresponds to
    original_event_ids: np.ndarray = None  # Optional, maps events in this instance to original problem

    def __post_init__(self):
        # Initialize default max_hours if not provided
        if self.max_hours is None:
            self.max_hours = 25 * np.ones(self.total_nurse)

    def split_by_day(self):
        """Split time windows by day."""
        day_instances = []
        time_window = self.time_windows
        C_event = self.event_event_costs
        C_home = self.home_event_costs
        C_depot_e = self.event_depot_costs
        C_depot_h = self.home_depot_costs
        C_dur = self.event_durations
        min_nurse = self.min_nurses
        m = self.total_event

        for d in range(self.total_day):
            events_today = np.where(time_window[:, d, 1] > 0)[0]
            C_event_today = C_event[np.ix_(events_today, events_today)]
            # C_home_today should keep all rows by only columns of events_today
            C_home_today = C_home[:, events_today]
            C_depot_e_today = C_depot_e[events_today]
            C_depot_h_today = C_depot_h
            C_dur_today = C_dur[events_today]
            time_window_today = time_window[events_today, d:d+1, :]
            min_nurse_today = min_nurse[events_today, :]

            day_instance = ProblemData(
                event_event_costs=C_event_today,
                home_event_costs=C_home_today,
                event_depot_costs=C_depot_e_today,
                home_depot_costs=C_depot_h_today,
                event_durations=C_dur_today,
                time_windows=time_window_today,
                min_nurses=min_nurse_today,
                total_rn=self.total_rn,
                total_lvn=self.total_lvn,
                total_nurse=self.total_nurse,
                total_event=len(events_today),
                total_day=1,
                day_index=d,
                original_event_ids=events_today
            )
            day_instances.append(day_instance)
        return day_instances
