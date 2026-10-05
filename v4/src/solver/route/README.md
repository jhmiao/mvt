# Route-Based RMP

This directory contains a route-based restricted master problem (RMP) for joint
event scheduling and nurse routing. The current experiment builds a fixed route
pool for one selected source day, solves the resulting MIP with Gurobi, extracts
the chosen variables, and writes them to JSON.

`run_rmp_routes.py` is the command-line entry point:

```text
cleaned .xlsx
    -> load_problem_data()
    -> build_event_copies()
    -> build_route_pool()
    -> build_master_model()
    -> Gurobi optimize()
    -> extract_route_rmp_result()
    -> v3/outputs/<input>_route_rmp.json
```

Run it from the repository root with:

```bash
python v3/src/experiments/run_rmp_routes.py \
  --file v3/data/cleaned/c101_Random1.xlsx \
  --day 0
```

The script's default project root is `v3`, so the JSON output path is
`v3/outputs/<input-stem>_day<day>_route_rmp.json`.

## Files and responsibilities

| File | Purpose |
| --- | --- |
| `copies.py` | Creates feasible event-time copies `(event, day, time_slot)` from time windows. A slot is 30 minutes and slots range from `0` to `47`. |
| `routes.py` | Defines an immutable `Route`: nurse, day, ordered visits, cost, work, and depot flag. |
| `pool_builder.py` | Creates the static candidate route pool. It generates idle, one-event, and feasible ordered two- and three-event routes, then creates a version for every nurse-day. |
| `master_builder.py` | Creates the Gurobi model, variables, coverage indices, constraints, and objective. |
| `rmp_runner.py` | Orchestrates copy generation, pool construction, model building, Gurobi parameters, solve, and solver metadata collection. |
| `cg_runner.py`, `pricing.py` | Reserved for column generation; both are currently empty. |
| `context.py` | An older, unused context definition. The active `RouteMasterContext` is in `master_builder.py`. |

The experiment script uses the result helpers in `src/solutions`:

| File | Purpose |
| --- | --- |
| `extract_route_rmp_result.py` | Reads the model variables and turns values above `0.5` into scheduled events and selected routes. |
| `route_rmp_result.py` | Defines the JSON-ready result dataclasses. |
| `io_route_rmp.py` | Builds the output filename and saves the result as JSON. |

## Data and event copies

`load_problem_data()` reads the workbook sheets into `ProblemData`:

- travel cost matrices between events, homes, and the depot;
- service duration for each event;
- time windows with shape `(event, day, earliest/latest)`;
- required RN and LVN counts for every event.

`build_event_copies()` creates one copy `(i, d, tau)` for each 30-minute start
slot `tau` inside an event's available window on day `d`. It also indexes copies
by event and by day. The master model must choose exactly one copy for every
event in the input instance.

The data loader currently changes each present latest-start value to at least
`1050 - duration` minutes. Confirm that this is the intended time-window rule
before interpreting schedules.

## Route pool

`build_route_pool()` first creates day-level, nurse-independent visit skeletons,
then instantiates them for each `(nurse, day)` pair. `run_rmp_routes.py`
enumerates every feasible route with one or two visits for its selected day.

- An idle route has no visits and zero cost.
- A single-event route visits one event-time copy.
- A two-event route `(v1, v2)` is included only if service at `v1`, plus travel
  from its event to `v2`'s event, finishes no later than the start of `v2`.
- A three-event route `(v1, v2, v3)` is included only if both consecutive legs
  satisfy that same service-and-travel condition.
- The reusable `RoutePoolConfig` defaults can sample and cap candidate routes.
  The single-day experiment sets `sample_two_event_pairs=False`,
  `max_two_per_day=None`, `sample_three_event_triples=False`,
  `max_three_per_day=None`, and `max_routes_per_nurse_day=None`, so it retains
  all feasible two- and three-event skeletons and all instantiated routes.

Route feasibility is assumed nurse-independent. Home location changes route
cost, but does not affect time feasibility. Depot feasibility is currently a
placeholder: with the default configuration every non-idle route is marked
depot-compatible, and its cost uses home-to-depot and depot-to-event legs.

## Master model

For every event-time copy `v`, the model has binary `y[v]`: whether that copy
is the chosen schedule. For every candidate route `k` for nurse `w` on day
`d`, it has binary `z[w, d, k]`: whether that route is used.

The model adds these constraints:

1. Every event selects exactly one copy: `sum(y[v] for v of event i) = 1`.
2. Every nurse selects exactly one route per day, normally including idle:
   `sum(z[w,d,k]) = 1`.
3. RN and LVN route coverage of each copy must meet the event's respective
   staffing requirements whenever that copy is selected.
4. Depot coverage is added when depot enforcement is enabled.
5. Workload-balance constraints are added when hour balancing or maximum-hour
   enforcement is enabled.

The objective minimizes the sum of selected route costs.

## Important current limitations

This is an integer MIP over a restricted route pool, not a full
column-generation implementation.

- The fixed pool contains routes with at most two visits. It cannot represent a
  nurse route with three or more events.
- Column generation/pricing is not implemented, so the solver cannot add
  missing routes after seeing the LP dual values.
- A selected route is not explicitly constrained to visit only selected
  event-time copies. For a valid discrete model, add a linking constraint such
  as `z[w,d,k] <= y[v]` for each visit `v` in that route.
- `run_rmp_routes.py` disables workload-balance constraints. Maximum-hour
  constraints remain disabled as well.

Consequently, treat the current objective as the best solution within the
one/two/three-event route pool, not as a globally complete routing solution.

## Running only one day

If the workbook itself has `day = 1`, run `run_rmp_routes.py` directly. Its
only model day is local day `0`.

For one day from a multi-day workbook, `run_rmp_routes.py` loads the full
workbook and uses `ProblemData.split_by_day()[args.day]`. Select the source day
with the zero-based `--day` option. The underlying transformation is:

```python
full_problem = load_problem_data(data_path)
day_problem = full_problem.split_by_day()[day]
result = solve_rmp_routes(day_problem, pool_cfg)
```

For day `d`, it:

- keeps only events whose time-window end on `d` is positive;
- slices all event-indexed travel, duration, staffing, and time-window arrays;
- retains every nurse and their home/depot costs;
- changes the time-window array to shape `(events_that_day, 1, 2)`;
- sets `total_day = 1`, with local model day `0`;
- preserves the original event IDs in `day_problem.original_event_ids` and the
  global day index in `day_problem.day_index`.

The result extractor maps local event IDs through `original_event_ids` and maps
the local model day `0` to `day_index`. Output therefore refers to the original
workbook's event IDs and day, and includes the day in the filename.

Before using the day result operationally, complete these route-model steps:

1. Add route-to-schedule linking constraints for every route visit.
2. Decide whether workload balance and depot enforcement belong in a one-day
   model, then expose them as explicit configuration flags rather than relying
   on the current defaults.
3. Generate routes longer than two visits or implement column generation.
4. Validate that every selected event copy receives the required RN/LVN staff
   and that every reported route is time-feasible after the chosen schedule is
   fixed.
