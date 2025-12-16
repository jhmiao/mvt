from typing import Optional

import numpy as np

from src.heuristics.config import HeuristicConfig
from src.structures.problem_data import ProblemData


def generate_schedules(data: ProblemData, config: Optional[HeuristicConfig] = None):

    time_window = data.time_windows  # shape (m, day, 2)
    m = data.total_event
    days = data.total_day
    cfg = config or HeuristicConfig()

    # discretized travel & duration (30 min)
    C_event = np.ceil(data.event_event_costs / 30).astype(int)
    C_dur   = np.ceil(data.event_durations / 30).astype(int) 

    all_day_patterns = {}

    for day in range(days):

        # 1. events that happen today
        events_today = [i for i in range(m) if time_window[i,day,0] > 0]
        k = len(events_today)
        # print(f"Day {day}, #events={k}")
        
        # print the duration of each event and the total duration
        total_dur = 0
        for i in events_today:
            dur = C_dur[i]
            total_dur += dur
        # print(f"Average duration of events today: {total_dur / k if k > 0 else 0}")

        # 2. valid start times per event (in 30-min blocks)
        valid_starts = {}
        for i in events_today:
            # tw_start, tw_end = time_window[i,day]  # in minutes
            dur = C_dur[i]
            # print(f"Event {i} duration: {dur}")
            valid_starts[i] = list(range(0, 16 - dur))
            # print(f"Event {i} valid starts: {valid_starts[i]}")

        # 3. storage for unique feasible-pair patterns
        patterns = []              # list of dicts containing schedule, pairs
        pair_sets = []             # list of frozensets representing feasible-pairs

        np.random.seed(cfg.seed)
        # 4. generate schedules (random or partial enumeration)
        for _ in range(cfg.num_samples):

            schedule = {}  # event → start time
            feasible = True


            # random assignment per event
            for e in events_today:
                if not valid_starts[e]:
                    feasible = False
                    break
                schedule[e] = int(np.random.choice(valid_starts[e]))

            if not feasible:
                continue

            # 5. compute feasible-pair set
            pair_set_directed = []
            for i in events_today:
                end_i = schedule[i] + C_dur[i]
                for j in events_today:
                    if i == j:
                        continue
                    if end_i + C_event[i,j] <= schedule[j]:
                        pair_set_directed.append((i,j))

            pair_set_directed = frozenset(pair_set_directed)
            pair_set_reversed = frozenset((j,i) for (i,j) in pair_set_directed)

            canonical = min(
                (tuple(sorted(pair_set_directed)), pair_set_directed),
                (tuple(sorted(pair_set_reversed)), pair_set_reversed),
            )[1]

            # 6. check subset relation using canonicalized sets only
            canonical_set = canonical

            # skip if canonical is already covered by an existing (super)set
            if any(canonical_set.issubset(existing) for existing in pair_sets):
                continue

            # remove any existing canonical that is a strict subset of the new one
            to_remove = [
                idx for idx, existing in enumerate(pair_sets)
                if existing.issubset(canonical_set) and existing != canonical_set
            ]
            for idx in reversed(to_remove):
                del pair_sets[idx]
                del patterns[idx]

            # store new pattern
            patterns.append({"schedule": schedule, "pairs": canonical_set})
            pair_sets.append(canonical_set)
            # print(f"Canonical pattern found: {canonical_set}")

        # # ensure we keep one empty-pairs pattern (schedule all events at time 0)
        # empty_pairs = frozenset()
        # if not any(empty_pairs == existing for existing in pair_sets):
        #     zero_schedule = {e: 0 for e in events_today}
        #     patterns.append({"schedule": zero_schedule, "pairs": empty_pairs})
        #     pair_sets.append(empty_pairs)

        for p in patterns:
            p["pairs"] = list(p["pairs"])   # convert frozenset → list
        all_day_patterns[day] = patterns
        print(f"Day {day}: found {len(patterns)} unique patterns")

    return all_day_patterns
