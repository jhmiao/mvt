# Fast route-pool fairness heuristic

`run_fast_fairness_routes.py` runs an alternating destroy-and-repair search over weekly schedules. The weekly score is travel cost plus weighted absolute deviations of nurse workload and leader-day counts from their respective means. Repairs optimize **daily travel only**; the weekly score controls simulated-annealing acceptance and best-schedule tracking.

## Route pool and temporary restrictions

The instance is split into source days. At construction, the heuristic builds the event copies and one route pool for each source day, once. The pools are read-only and reused for the initial solve and every later repair; each solve builds a new Gurobi RMP model over the reused pool. With the experiment's configuration, each pool contains idle, all one-event, all feasible two-event, and all feasible three-event routes (no sampling or caps).

Temporary bans are constraints on that repair model's route-selection variables (`z = 0`), so they disappear after the solve and never alter the stored pool. Since a split source-day problem has local day index `0`, bans are keyed as `(nurse, 0)`.

## Destroy and fix operators

Iterations alternate workload and leader-day operators. Each selects a nurse from the highest-ranked top-k (rank-biased by default), then a relevant assigned source day.

- **Workload:** choose one positive-work route of the selected nurse. Ban every route for that nurse on the selected day whose work is at least `ceil(selected_route.work * workload_route_restriction_factor)`; the default factor `1.0` therefore requires a strictly lower-work replacement. Re-solve that day using travel cost only.
- **Leader day:** choose a day on which the selected nurse has a depot-using route. Ban every depot-using route for that nurse on that day, then re-solve that day using travel cost only. A non-depot route (including idle, if feasible) is still available.

The repaired day replaces only that day in a candidate week. Feasible candidates are evaluated with the full weekly score and accepted if improving, otherwise probabilistically by simulated annealing; the best candidate is retained separately. Repair solves normally stop after their first feasible solution, subject to the configured time/work limits.
