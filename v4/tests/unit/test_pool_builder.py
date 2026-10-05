import argparse
from pathlib import Path
from pprint import pprint
import sys
import inspect

V3_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(V3_ROOT))

from src.io.data_loader import load_problem_data
from src.solver.route.copies import build_event_copies
from src.solver.route.pool_builder import build_route_pool, RoutePoolConfig
# import src.solver.route.pool_builder as pool_builder




def main() -> None:
    parser = argparse.ArgumentParser(description="Test build_route_pool on a cleaned xlsx file.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to a cleaned .xlsx file under v3/data/cleaned.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    problem_data = load_problem_data(str(file_path))
    copy_index = build_event_copies(problem_data)

    cfg = RoutePoolConfig(
        seed=0,
        max_routes_per_nurse_day=5,
        max_two_per_day=5,
        two_event_pair_samples=50,
        force_depot_ok_for_working_routes=True,
    )

    pool = build_route_pool(problem_data, copy_index, cfg)
    pprint(pool)


if __name__ == "__main__":
    main()

# run with:
# python v3/tests/unit/test_pool_builder.py --file v3/data/cleaned/c101_Random1.xlsx
