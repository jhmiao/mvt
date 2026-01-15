from typing import List, Tuple, Dict
from src.structures.problem_data import ProblemData
from dataclasses import dataclass

EventCopy = Tuple[int, int, int]  # (i, d, tau)

@dataclass
class CopyIndex:
    copies: List[EventCopy]
    copies_by_event: Dict[int, List[EventCopy]]
    copies_by_day: Dict[int, List[EventCopy]]
    copies_by_event_day: Dict[Tuple[int, int], List[EventCopy]]

def get_feasible_slots(problem_data: ProblemData, i: int, d: int) -> List[int]:
    feasible_slots = []
    # event = problem_data.total_event[i]
    for tau in range(48):
        # need to convert time_windows to time slots... divide by 30 mins
        # time_window[i][d][0] // 30 is the earliest feasible slot for event i on day d
        # time_window[i][d][1] // 30 is the latest feasible slot for event i on day d
        if problem_data.time_windows[i][d][1] == 0:
            continue  # Skip if no time window is defined
        if (tau * 30 >= problem_data.time_windows[i][d][0]) and (tau * 30 <= problem_data.time_windows[i][d][1]):
            feasible_slots.append(tau)
    return feasible_slots

def build_event_copies(problem_data: ProblemData) -> CopyIndex:
    m = problem_data.total_event
    days = problem_data.total_day

    copies: List[EventCopy] = []
    copies_by_event: Dict[int, List[EventCopy]] = {i: [] for i in range(m)}
    copies_by_day: Dict[int, List[EventCopy]] = {d: [] for d in range(days)}
    copies_by_event_day: Dict[Tuple[int, int], List[EventCopy]] = {}

    for i in range(m):
        for d in range(days):
            feasible_slots = get_feasible_slots(problem_data, i, d)
            if not feasible_slots:
                continue

            key = (i, d)
            copies_by_event_day[key] = []

            for tau in feasible_slots:
                v = (i, d, tau)
                copies.append(v)
                copies_by_event[i].append(v)
                copies_by_day[d].append(v)
                copies_by_event_day[key].append(v)

    return CopyIndex(
        copies=copies,
        copies_by_event=copies_by_event,
        copies_by_day=copies_by_day,
        copies_by_event_day=copies_by_event_day,
    )

