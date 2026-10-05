# Fast weekly route heuristic: search-quality plan

## Current diagnosis

The performance work is complete enough for search experiments: event copies,
route pools, route-attribute indexes, and one base MIP per day are reused. The
remaining problem is that fast iterations often generate repetitive candidates.

- A repair changes only one donor nurse-day, then solves the same deterministic
  daily travel objective with the same seed and `SolutionLimit=1`. Reusing the
  previous model solution makes returning the same or a closely related first
  feasible solution even more likely.
- The workload operator only forces work off an overloaded donor. It does not
  choose an underloaded receiver. Likewise, the leader operator removes depot
  duty from one nurse but does not direct it toward a low-leader nurse.
- All other nurses in the day remain unrestricted, so a nominally local repair
  may re-optimize the entire day for travel. Workload and leader repairs can
  undo each other's fairness gains.
- Rank-biased top-k selection and fixed one-donor destroy strength can still
  revisit a small set of nurse-days and restriction patterns despite the new
  operator roulette.
- The experiment defaults weight leader-day deviation at `400` versus workload
  deviation at `1`. Because leader days change discretely, this can create large
  objective jumps and plateaus; the intended tradeoff should be verified.
- The static pool and maximum route length impose a final search ceiling: no
  acceptance rule can reach a schedule requiring a route absent from the pool.

## Prioritized changes

### 1. Measure repetition and component tradeoffs

- [x] Add a candidate fingerprint from selected `(source_day, nurse, route_index)`
  values and log whether each proposal/current state has been seen before.
- [x] Log temperature plus travel, workload penalty, and leader penalty before
  and after every proposal, not only the combined objective.
- [x] Track per-operator duplicate, infeasible, accepted, improving, and
  new-best rates. Use these measurements to distinguish a weak neighborhood
  from an over-cooled acceptance rule.

### 2. Diversify cheap repairs

- [ ] Use an iteration-dependent deterministic Gurobi seed instead of the same
  `seed + day` on every repair.
- [ ] Randomize workload destroy strength among several safe levels, including
  forbidding only the incumbent route and forbidding routes above progressively
  lower workload thresholds.
- [ ] Keep a short tabu list of recent `(operator, nurse, day, restriction)`
  signatures and resample rather than immediately repeating them.
- [ ] Generate a small number of fast candidates for the chosen neighborhood,
  score all of them with the full weekly objective, and pass the best distinct
  candidate to SA. Prefer this external scoring over adding the full weekly
  fairness objective to the daily MIP.
- [ ] Optionally add a tiny randomized tie-break term to route travel costs so
  equivalent or near-equivalent daily repairs do not always return the same
  first feasible assignment.

### 3. Make neighborhoods genuinely local and directional

- [ ] Select both an overloaded donor and a compatible underloaded receiver.
  For leader moves, pair a high leader-day nurse with a low leader-day nurse.
- [ ] Fix unrelated nurses to their incumbent route with temporary bounds, and
  unlock only the donor plus a small set of compatible receivers. Expand the
  receiver set only when the restricted repair is infeasible.
- [x] Add an explicit same-type, same-day route swap that moves workload or depot duty
  without requiring a sequence of individually unattractive one-sided moves.
- [x] Add an occasional larger neighborhood that destroys the selected routes
  of all top-k nurses on one day.
- [ ] Consider a two-source-day neighborhood when small neighborhoods have
  produced no new best.

### 4. Add stagnation-aware search control

- [ ] Count iterations since the last new best. After a configured threshold,
  reheat SA, widen top-k/destroy strength, or switch to a larger neighborhood.
- [x] Do not cool on duplicate or infeasible proposals; they did not explore a
  new state. Calibrate initial temperature from a target acceptance probability
  for a representative relative deterioration.
- [x] Replace strict workload/leader alternation with a fixed 35/35/15/10/5
  workload/leader/swap/combined/multi-destroy roulette.
- [ ] Consider adapting those operator weights based primarily on recent
  distinct candidates and new-best discoveries.
- [ ] Add a reproducible restart from the best solution followed by a stronger
  randomized perturbation when reheating fails.

### 5. Address route-pool limitations only if diagnostics require it

- [ ] Measure how often desired donor/receiver moves have no compatible routes.
  If frequent, enlarge or diversify the static pool, allow longer routes, or
  periodically add routes for blocked moves; otherwise keep the current pool.

## Evaluation protocol

- [ ] Compare changes over several seeds using best objective versus wall time,
  time to last new best, distinct-candidate rate, and the three objective
  components. Do not judge improvements from a single trajectory.
- [ ] Run a coefficient-sensitivity check, or normalize the two deviation
  measures before weighting them, so one component does not dominate by scale
  accidentally.
- [ ] Add regression tests for restored bounds, stable route indices, seeded
  reproducibility, tabu expiry, and preservation of the best solution even when
  the current SA state deteriorates.
