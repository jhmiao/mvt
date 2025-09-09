import os, sys, pickle
import numpy as np

# Add project root, not src
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(project_root)

from src.models.context import Context
from src.io.data_loader import load_problem_data
from src.solver.continuous import continuous_algorithm
from src.solver.extract import routes_from_active_x_t
# from src.models.travel_cost import TravelCost
from src.heuristics.lns_gurobi import LNSCfg, lns_with_gurobi, build_full_model


def main():
    pd = load_problem_data('data/raw/c101/c101_data.xlsx', type='continuous')

    # --- get an initial solution by running your full solver once
    summary0 = continuous_algorithm(pd, work_limit=50, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))  # pick a sensible budget
    active_x0 = summary0["active_x"]          # list of (i,j,d,w)
    active_t0 = summary0["active_t"]          # dict {(i,d): t}
    initial_sol = routes_from_active_x_t(active_x0, active_t0, pd)
    # --- feasibility check
    ctx = Context(pd)
    rep0 = ctx.check_solution(initial_sol)
    print("Initial feasible?", getattr(rep0, "feasible", True))

    # --- run LNS with sub-MIP repair

    cfg = LNSCfg(destroy_frac=0.5, iters=100, rng_seed=42, timelimit_s=10, threads=8)
    best_sol = lns_with_gurobi(initial_sol, active_t0, pd, ctx, cfg)

    # # --- final check
    # repF = ctx.check_solution(best_sol)
    # print("Final feasible?", getattr(repF, "feasible", True))
    # summary = continuous_algorithm(
    #     pd, work_limit=1000, seed_number=1,
    #     multiple_tw=None, event_limit=None, pruning=2,
    #     min_hour=None, max_hour=25 * np.ones(pd.n)
    # )

    # # with open('outputs/c101_summary.pkl', 'wb') as f:
    # #     pickle.dump(summary, f)

    # active_x = summary['active_x']
    # active_t = summary['active_t']
    # sol = routes_from_active_x_t(active_x, active_t, pd)
    # travel = TravelCost(pd)

    # print("Solution:", sol)

if __name__ == "__main__":
    main()
