# Route-based restricted master problem

This directory implements a route-based mixed-integer programming (MIP)
approach for assigning nurses to event visits.  It solves a **restricted master
problem (RMP)**: instead of deciding every travel arc directly in the MIP, it
first generates a finite pool of feasible candidate routes and then selects one
route for each nurse and day.

## Solution flow

1. Load the instance, optionally split it into one-day subproblems, and create
   feasible event-time copies.
2. Generate a static pool of candidate routes for each nurse-day.
3. Build and solve a binary MIP that selects routes and event-time copies.
4. Extract the selected event schedule and selected nurse routes as JSON.

The entry point is `solve_rmp_routes()` in `rmp_runner.py`.

## Event-time copies

For every event `i`, day `d`, and feasible half-hour time slot `tau`, the
solver creates an event-time copy:

```text
v = (i, d, tau)
```

The binary variable `y[v]` is one when that copy is selected.  Each original
event must select exactly one feasible copy:

```text
sum(y[v] for v belonging to event i) = 1
```

`copies.py` builds and indexes these copies by event, day, and event-day.

## Candidate route pool

A route is an ordered sequence of event-time copies for one nurse on one day.
The implementation supports:

- an idle route;
- one-event routes;
- feasible ordered two-event routes;
- optional feasible ordered three-event routes.

A consecutive visit pair `(v1, v2)` is feasible when:

```text
end(v1) + travel_cost(event(v1), event(v2)) <= start(v2)
```

A three-event route requires this condition for both consecutive legs.
The same original event cannot appear more than once in a multi-event route.

For each non-idle visit sequence, the pool contains two distinct candidates:

- a direct route (`depot_incl = 0`), travelling home -> visits -> home;
- a depot-using route (`depot_incl = 1`), travelling home -> depot -> visits
  -> depot -> home.

The two candidates have separate costs, allowing the MIP to choose whether the
route uses the depot.

## Master MIP

For every candidate route `k` of nurse `w` on day `d`, the binary variable
`z[w,d,k]` is one when that route is selected.

The master contains the following principal constraints:

1. **One event time:** each event selects exactly one event-time copy.
2. **One nurse-day route:** every nurse selects exactly one route per day;
   idle routes make non-working days possible.
3. **Staffing coverage:** for every selected event-time copy, the selected RN
   and LVN routes that visit it must meet its required staffing levels.
4. **Depot coverage:** every selected event-time copy must be visited by at
   least one selected depot-using route.  This is the mechanism that requires
   a depot visit without forcing every nurse route through the depot.

The objective minimizes total selected-route cost:

```text
minimize sum(route_cost[w,d,k] * z[w,d,k])
```

For a direct route, its cost includes home-to-first-event, all consecutive
event-to-event legs, and last-event-to-home.  A depot route instead includes
home-to-depot legs and depot-to-first / last-event-to-depot legs.

## Implementation optimizations

The code avoids constructing a dense arc-based routing model through several
practical optimizations:

- **Nurse-agnostic day skeletons:** visit sequences are generated once per day
  before being instantiated for each nurse.  Only the home-dependent portion
  of the cost changes by nurse.
- **Cached route attributes:** work, depot status, and internal travel cost
  are calculated once per visit sequence (and depot variant), then reused for
  every nurse.
- **Sparse coverage indices:** `master_builder.py` stores, for each event-time
  copy, only the route variables that cover it, separated into RN, LVN, and
  depot-using sets.  This avoids a large dense route-copy incidence matrix.
- **Configurable candidate limits:** single-, two-, three-event, and final
  nurse-day route pools can be capped.  Candidate pairs and triples can also
  be randomly sampled with a seeded generator instead of fully enumerated.
- **Early stopping:** full enumeration returns as soon as a configured route
  limit is reached.
- **Single-day splitting:** experiment code can split a multi-day workbook and
  build a much smaller RMP for one requested day.

## Scope and limitations

This is a static restricted-route MIP, not a complete branch-and-price or
column-generation method.  It can only choose among generated routes, so a
missing beneficial route cannot be discovered during optimization.

The current pool contains at most three visits per route.  Also, the master
should eventually add route-to-schedule linking constraints such as
`z[w,d,k] <= y[v]` for every visit `v` in a route, ensuring that a selected
route cannot include an event-time copy that the event schedule did not
select.
