# Nurse Routing Optimization

This repository contains code for solving a nurse routing and scheduling optimization problem, where nurses must visit patients at different locations under time and capacity constraints.

## Project Structure
```bash
nurse-routing/
│
├─ config/  
│   ├─ paths.yaml        # paths to input data and output directories  
│   └─ params.yaml       # solver/heuristic parameters (time limits, caps)  
│
├─ data/                 # raw and processed data (not tracked in Git by default) 
│  ├─ raw/               # input excel/csv
│  └─ interim/           # converted npy/pkl for speed 
│
├─ src/
│   ├─ entities/                    # domain objects (pure data; no solver logic)
│   │  ├─ __init__.py
│   │  ├─ event.py                  # EventSpec, EventPlan, Event (pre/post assignment)
│   │  └─ nurse.py                  # Nurse (role, caps, availability, home)
│   │
│   ├─ models/                      # problem & solution structures (no Gurobi code)
│   │  ├─ __init__.py
│   │  ├─ problem_data.py           # ProblemData wrapper around arrays from loader
│   │  ├─ travel_model.py           # TravelModel: cost(role, nurse, i, j[, day])
│   │  └─ solution.py               # Route, Solution (day_routes, day list)
│   │
│   ├─ solver/                      # exact/MI(N)LP solvers, model builders, extractors
│   │  ├─ __init__.py
│   │  ├─ gurobi_builder.py         # build x(i,j,d,w), s(i,d), t(i,d), constraints
│   │  ├─ gurobi_solve.py           # solve routine, params, warm start, returns active_x
│   │  └─ extract.py                # active_x → routes; routes → assignments (update Event.plan)
│   │
│   ├─ heuristics/                  # local search & metaheuristics
│   │  ├─ __init__.py
│   │  ├─ intra_route.py            # 2-opt, Or-opt, best-position reinsertion
│   │  ├─ inter_route.py            # relocate, swap, cross-exchange, merges
│   │  └─ neighborhoods.py          # ruin & recreate, fix-and-opt windows
│   │
│   ├─ timing/                      # schedule propagation & feasibility checks
│   │  ├─ __init__.py
│   │  └─ schedule_passes.py        # earliest-forward, latest-backward, wiggle-room
│   │
│   ├─ io/                          # data in/out; adapters around your existing files
│   │  ├─ __init__.py
│   │  ├─ data_loader.py            # uses your load_problem_data → ProblemData
│   │  ├─ read_pickle.py            # reads feasible pickle (summary['active_x'])
│   │  └─ writers.py                # export CSV/JSON, MIP starts, audit tables
│   │
│   ├─ eval/                        # metrics & sanity checks
│   │  ├─ __init__.py
│   │  └─ metrics.py                # total_travel, route count, feasibility validators
│   │
│   └─ utils/                       # generic helpers, indexing, small tools
│   ├─ __init__.py
│   └─ indexing.py               # node indices, role splits, guards, safe argmin
│
├─ tests/  
│   ├─ test_extract_routes.py   # unit tests for route extraction  
│   ├─ test_timing_passes.py    # unit tests for timing feasibility checks  
│   └─ test_two_opt.py          # unit tests for 2-opt heuristic  
│
├─ notebooks/  
│   ├─ exploration.ipynb        # Jupyter notebooks for data exploration / prototyping  
│
├─ outputs/              # results, logs, figures (generated during runs)  
│
├─ README.md             # project documentation  
└─ requirements.txt      # Python dependencies  

```

## Features

- Models nurse routing and scheduling as a combinatorial optimization problem
- Implements heuristics such as 2-opt and other local search methods
- Integrates with exact solvers for benchmarking and comparison
- Supports customizable constraints (time windows, nurse capacity, etc.)

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/nurse-routing-optimization.git
    cd nurse-routing-optimization
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run an example:
    ```bash
    python main.py --config configs/example.yaml
    ```

## Project Structure

- `src/` — Core algorithms and utilities
- `data/` — Sample datasets and problem instances
- `configs/` — Configuration files for experiments
- `notebooks/` — Jupyter notebooks for analysis and visualization

## Methods

- **2-opt**: Improves routes by iteratively reversing segments to reduce total distance.
- **Local Search**: Applies neighborhood moves to escape local optima.
- **Exact Solvers**: Uses mathematical programming for optimal solutions.


## License

This project is licensed under the MIT License.
