import os, sys, pickle
import numpy as np

# Add project root, not src
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(project_root)

from src.io.data_loader import load_problem_data
from src.solver.continuous import continuous_algorithm
from src.solver.extract import routes_from_active_x_t
from src.models.travel_cost import TravelCost


def main():
    pd = load_problem_data('data/raw/c101/c101_data.xlsx', type='continuous')

    summary = continuous_algorithm(
        pd, work_limit=1000, seed_number=1,
        multiple_tw=None, event_limit=None, pruning=2,
        min_hour=None, max_hour=25 * np.ones(pd.n)
    )

    # with open('outputs/c101_summary.pkl', 'wb') as f:
    #     pickle.dump(summary, f)

    active_x = summary['active_x']
    active_t = summary['active_t']
    sol = routes_from_active_x_t(active_x, active_t, pd)
    # travel = TravelCost(pd)

    print("Solution:", sol)

if __name__ == "__main__":
    main()
