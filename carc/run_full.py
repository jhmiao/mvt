import os
import sys
import argparse
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

# Project imports (assume same PYTHONPATH / repo layout)
from data_loader import load_problem_data
from algorithms.continuous import continuous_algorithm
from algorithms.daily_greedy import daily_greedy_heuristic

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

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
    parser.add_argument("--overall-seconds", type=float, default=None,
                        help="Overall wallclock budget for the entire batch (seconds). Each instance consumes remaining time to try for optimality.")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="(Optional) Per-instance cap in seconds. If --overall-seconds is set, this is ignored.")
    parser.add_argument("--buffer-seconds", type=float, default=120.0,
                        help="Time reserved at the end for cleanup/logging (default: 120s).")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Working directory (default: /project2/ssuen_1733/mvt_runs or current dir if scratch unavailable).")
    parser.add_argument("--seed", type=int, default=19, help="Random seed for the solver (algorithm-level).")
    parser.add_argument("--grb-seed", type=int, default=19, help="Gurobi seed for reproducibility (passed via env).")
    parser.add_argument("--threads", type=int, default=None, help="Gurobi threads. Default uses SLURM_CPUS_PER_TASK or all cores.")
    parser.add_argument("--cap", type=int, default=25, help="Max hours per nurse (if your algorithm uses it).")
    parser.add_argument("--instances", type=str, nargs="*", default=["c101","c201","r101","rc101"],
                        help="Instance basenames (without extension). Default: c101 c201 r101 rc101")
    args = parser.parse_args()

    # Workdir
    default_scratch = Path("/project2/ssuen_1733/mvt_runs")
    workdir = Path(args.workdir) if args.workdir else (
        default_scratch if default_scratch.parent.exists() else Path.cwd() / "mvt_runs"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    # Environment knobs for continuous.py parameter injection
    os.environ["CARC_GRB_SEED"] = str(args.grb_seed)
    if args.threads is not None:
        os.environ["CARC_GRB_THREADS"] = str(int(args.threads))

    # Detect/record threads
    effective_threads = args.threads if args.threads is not None else detect_threads()

    # Global run log
    runlog = workdir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(runlog, "w") as R:
        R.write(f"Workdir: {workdir}\n")
        R.write(f"Overall seconds: {args.overall_seconds}\n")
        R.write(f"Per-instance time_limit arg: {args.time_limit}\n")
        R.write(f"Buffer seconds: {args.buffer_seconds}\n")
        R.write(f"Threads: requested={args.threads} effective={effective_threads}\n")
        R.write(f"Seed: algo={args.seed} grb={args.grb_seed}\n")
        R.write(f"Instances: {args.instances}\n")

    # Track overall budget
    t0 = datetime.now()

    def seconds_left():
        if args.overall_seconds is None:
            return None
        used = (datetime.now() - t0).total_seconds()
        return max(0.0, args.overall_seconds - used - args.buffer_seconds)

    for name in args.instances:
        inst_dir = workdir / name
        inst_dir.mkdir(exist_ok=True, parents=True)

        # Per-instance logs
        grb_log = inst_dir / f"{name}.gurobi.log"
        os.environ["CARC_GRB_LOGFILE"] = str(grb_log)
        py_log = inst_dir / f"{name}.run.log"

        with open(py_log, "a") as L:
            L.write(f"[{datetime.now().isoformat()}] Starting instance {name}\n")

            # Compute this instance's time limit
            if args.overall_seconds is not None:
                rem = seconds_left()
                if rem <= 0:
                    L.write(f"[{datetime.now().isoformat()}] No overall time remaining; skipping {name}\n")
                    break
                time_limit = rem
                L.write(f"[{datetime.now().isoformat()}] Allocated time_limit={time_limit:.1f}s from overall budget\n")
            else:
                time_limit = args.time_limit
                L.write(f"[{datetime.now().isoformat()}] Using per-instance time_limit={time_limit}\n")

            # Locate the Excel file relative to submit dir
            xlsx = Path.cwd() / f"{name}.xlsx"
            L.write(f"[{datetime.now().isoformat()}] Loading file: {xlsx.resolve()} exists={xlsx.exists()}\n")
            if xlsx.exists():
                try:
                    L.write(f"[{datetime.now().isoformat()}] sha256={sha256(xlsx)}\n")
                except Exception as e:
                    L.write(f"[{datetime.now().isoformat()}] sha256 error: {e}\n")

            try:
                data_dir = Path(f"/project2/ssuen_1733/data")
                data_path = data_dir / f"{name}.xlsx"
                data = load_problem_data(str(data_path), type='continuous')

                # Prepare max_hour vector if applicable
                try:
                    import numpy as np
                    max_hour = np.ones(data.n) * args.cap
                except Exception:
                    max_hour = None

                # Call solver: strive for optimality; TimeLimit stops if we hit wallclock
                L.write(f"[{datetime.now().isoformat()}] Calling solver with time_limit={time_limit}, seed={args.seed}\n")
                summary = continuous_algorithm(
                    data,
                    work_limit=time_limit,
                    seed_number=args.seed,
                    multiple_tw=None,
                    event_limit=None,
                    pruning=2,
                    min_hour=None,
                    max_hour=max_hour
                )
                # summary = daily_greedy_heuristic(
                #     data,
                #     work_limit=time_limit,
                #     seed_number=args.seed,
                #     event_limit=None,
                #     pruning=2,
                #     update_max_hour=False,
                #     min_hour=None
                # )

                # Save outputs
                pkl_path = inst_dir / f"{name}_summary.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(summary, f)
                L.write(f"[{datetime.now().isoformat()}] Finished {name}. Saved: {pkl_path}\n")

            except Exception as e:
                L.write(f"[{datetime.now().isoformat()}] ERROR on {name}: {e}\n")

    print(f"All runs complete. See logs and pickles under: {workdir}")

if __name__ == "__main__":
    main()
