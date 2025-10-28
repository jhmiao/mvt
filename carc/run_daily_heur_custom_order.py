import os
import sys
import argparse
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from itertools import permutations

# Project imports
from data_loader import load_problem_data
from algorithms.daily_greedy_custom_order import daily_greedy_heuristic


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
    parser = argparse.ArgumentParser(description="Run daily greedy heuristic with custom day order on CARC.")
    parser.add_argument("--worklimit", type=float, default=1000,
                        help="Work limit for each heuristic run (default=1000).")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads (default: detect from SLURM_CPUS_PER_TASK).")
    parser.add_argument("--seed", type=int, default=19, help="Random seed.")
    parser.add_argument("--instances", type=str, nargs="*", default=["c101-a"],
                        help="Instance basenames to run (e.g., c101-a).")
    parser.add_argument("--custom_day_order", type=str, default=None,
                        help="Comma-separated list of day order, e.g. '0,1,2,3,4'.")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for outputs.")
    args = parser.parse_args()

    # ---- Parse day order ----
    if args.custom_day_order is not None:
        try:
            custom_day_order = [int(x) for x in args.custom_day_order.split(",")]
        except ValueError:
            print(f"Invalid --custom_day_order format: {args.custom_day_order}")
            sys.exit(1)
    else:
        custom_day_order = list(range(5))  # default [0,1,2,3,4]

    # ---- Prepare workdir ----
    default_scratch = Path("/project2/ssuen_1733/mvt_runs")
    workdir = Path(args.workdir) if args.workdir else (
        default_scratch if default_scratch.parent.exists() else Path.cwd() / "mvt_runs"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    effective_threads = args.threads if args.threads is not None else detect_threads()

    # ---- Run each instance ----
    for name in args.instances:
        inst_dir = workdir / name
        inst_dir.mkdir(exist_ok=True, parents=True)

        py_log = inst_dir / f"{name}_greedy_{','.join(map(str, custom_day_order))}.log"
        with open(py_log, "a") as L:
            L.write(f"[{datetime.now().isoformat()}] Starting {name} with day_order={custom_day_order}\n")

            # Load problem data
            try:
                data_path = Path(f"/project2/ssuen_1733/data/{name}.xlsx")
                data = load_problem_data(str(data_path), type="continuous")
            except Exception as e:
                L.write(f"[{datetime.now().isoformat()}] ERROR loading data: {e}\n")
                continue

            # Run the heuristic
            try:
                L.write(f"[{datetime.now().isoformat()}] Running daily_greedy_heuristic(work_limit={args.worklimit})\n")
                summary = daily_greedy_heuristic(
                    data,
                    work_limit=args.worklimit,
                    seed_number=args.seed,
                    event_limit=None,
                    pruning=2,
                    update_max_hour=False,
                    custom_day_order=custom_day_order,
                    min_hour=None
                )

                # Save results
                pkl_path = inst_dir / f"{name}_greedy_{','.join(map(str, custom_day_order))}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(summary, f)
                L.write(f"[{datetime.now().isoformat()}] Saved: {pkl_path}\n")

            except Exception as e:
                L.write(f"[{datetime.now().isoformat()}] ERROR running heuristic: {e}\n")

        print(f"Finished instance={name}, day_order={custom_day_order}")


if __name__ == "__main__":
    main()
