import argparse
from pathlib import Path
from pprint import pprint
import sys

V3_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(V3_ROOT))

from src.io.data_loader import load_problem_data
from src.solver.route.copies import build_event_copies


def main() -> None:
    parser = argparse.ArgumentParser(description="Test build_event_copies on a cleaned xlsx file.")
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

    print(f"Total event copies: {len(copy_index.copies_by_event_day)}")
    pprint(copy_index.copies_by_event_day)

if __name__ == "__main__":
    main()

# run with:
# python v3/tests/unit/test_copies.py --file v3/data/cleaned/c101_Random1.xlsx