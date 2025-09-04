
import os
import sys
import argparse
import pickle
from datetime import datetime
from pathlib import Path

# Project imports (assume same PYTHONPATH / repo layout)
from data_loader import load_problem_data
from algorithms.continuous import continuous_algorithm

def detect_threads():
    v = os.getenv("SLURM_CPUS_PER_TASK")
    if v and v.isdigit():
        return int(v)
    try:
        import multiprocessing
        return max(1, multiprocessing.cpu_count())
    except Exception:
        return 1

def main():
    parser = argparse.ArgumentParser(description="Run MVT instances on USC CARC Discovery.")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Solver wallclock limit in seconds (CARC job time budget).")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Working directory (defaults to /scratch1/$USER/mvt_runs or current dir if scratch is unavailable).")
    parser.add_argument("--seed", type=int, default=19, help="Random seed for the solver.")
    parser.add_argument("--cap", type=int, default=25, help="Max hours per nurse (if your algorithm uses it).")
    args = parser.parse_args()

    user = os.getenv("USER", "user")
    default_scratch = Path(f"/scratch1/{user}/mvt_runs")
    workdir = Path(args.workdir) if args.workdir else (default_scratch if default_scratch.parent.exists() else Path.cwd()/ "mvt_runs")
    workdir.mkdir(parents=True, exist_ok=True)

    # Instances to run
    instances = ["c101", "c201", "r101", "rc101"]

    # Global run log (minimal prints otherwise)
    runlog = workdir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(runlog, "w") as R:
        R.write(f"Workdir: {workdir}\\n")
        R.write(f"TimeLimit: {args.time_limit}\\n")
        R.write(f"Threads: {detect_threads()}\\n")
        R.write(f"Instances: {instances}\\n")

    for name in instances:
        inst_dir = workdir / name
        inst_dir.mkdir(exist_ok=True, parents=True)

        # Per-instance log file for Gurobi
        grb_log = inst_dir / f"{name}.gurobi.log"
        os.environ["CARC_GRB_LOGFILE"] = str(grb_log)

        # Minimal python-side log
        py_log = inst_dir / f"{name}.run.log"
        with open(py_log, "w") as L:
            L.write(f"[{datetime.now().isoformat()}] Starting instance {name}\\n")
            L.flush()

            # Load data for the instance (adjust to your loader's API as needed)
            data = load_problem_data(name, type='continuous')

            # Prepare any caps your algorithm expects
            # For example: max_hour list of length n
            try:
                import numpy as np
                max_hour = np.ones(data.n) * args.cap
            except Exception:
                max_hour = None

            # Call your solver: use TimeLimit via work_limit parameter to stop if it runs out,
            # otherwise it will reach optimality (MIPGap=0) if possible.
            try:
                summary = continuous_algorithm(
                    data,
                    work_limit=args.time_limit,   # seconds
                    seed_number=args.seed,
                    multiple_tw=None,
                    event_limit=None,
                    pruning=False,
                    min_hour=None,
                    max_hour=max_hour
                )
                # Save pickle of the result/summary per instance
                pkl_path = inst_dir / f"{name}_summary.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(summary, f)

                L.write(f"[{datetime.now().isoformat()}] Finished {name}. Saved: {pkl_path}\\n")
            except Exception as e:
                L.write(f"[{datetime.now().isoformat()}] ERROR on {name}: {e}\\n")

    print(f"All runs complete. See logs and pickles under: {workdir}")

if __name__ == "__main__":
    main()
