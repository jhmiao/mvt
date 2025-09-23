import os, sys, pickle
import numpy as np

# Add project root, not src
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(project_root)

from src.models.context import Context
from src.io.data_loader import load_problem_data
from src.solver.continuous import continuous_algorithm, continuous_warm_start
from src.solver.extract import routes_from_active_x_t
# from src.models.travel_cost import TravelCost
from src.heuristics.lns_gurobi import LNSCfg, lns_with_gurobi, build_full_model
from src.heuristics.daily_greedy import continuous_algorithm_heuristic


def main():
    pd = load_problem_data('data/raw/rc101/rc101_data.xlsx', type='continuous')

    # run full solver for a long time and save the result
    # summary0 = continuous_warm_start(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))  
    # with open('outputs/rc101_summary.pkl', 'wb') as f:
    #     pickle.dump(summary0, f)
    # file_path2 = 'outputs/c101_lns_summary.pkl'

    # with open(file_path2, "rb") as f:
    #     summary0 = pickle.load(f)

    # --- get an initial solution by running your full solver once
    # summary0 = continuous_algorithm(pd, work_limit=100, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))  
    # active_x0 = summary0["active_x"]          # list of (i,j,d,w)
    # active_t0 = summary0["active_t"]          # dict {(i,d): t}
    # initial_sol = routes_from_active_x_t(active_x0, active_t0, pd)
    # --- feasibility check
    ctx = Context(pd)
    # rep0 = ctx.check_solution(initial_sol)
    # print("Initial feasible?", rep0.ok)

    # --- run daily greedy heuristic
    summary_heuristic = continuous_algorithm_heuristic(pd, work_limit=3000, seed_number=0, event_limit=None, pruning=2, min_hour=None)
    # save summary
    with open('outputs/rc101_greedy_summary-2.pkl', 'wb') as f:
        pickle.dump(summary_heuristic, f)

    active_xh = summary_heuristic["active_x"]          # list of (i,j,d,w)
    active_th = summary_heuristic["active_t"]          # dict {(i,d): t}
    sol_heuristic = routes_from_active_x_t(active_xh, active_th, pd)
    reph = ctx.check_solution(sol_heuristic)
    print("Heuristic feasible?", reph.ok)
    # print("Heuristic details:", reph.violations)
    print("Heuristic obj:", summary_heuristic["objective_value"])

    # --- run LNS with sub-MIP repair

    # cfg = LNSCfg(destroy_frac=0.5, iters=5, rng_seed=42, work_limit=200, threads=8)
    # best_sol, best_obj, summary = lns_with_gurobi(initial_sol, active_t0, pd, ctx, cfg)
    # # for route in best_sol.iter_routes():
    # #     print(route)

    # # save summary
    # with open('outputs/c101_lns_bad_summary.pkl', 'wb') as f:
    #     pickle.dump(summary, f)

    # # # --- final check
    # repF = ctx.check_solution(best_sol)
    # # for route in best_sol.iter_routes():
    # #     print(route) 
    # print("Final feasible?", repF.ok)
    # # print("Final details:", repF.violations)
    # print("Final obj:", best_obj)


    # active_x = summary['active_x']
    # active_t = summary['active_t']
    # sol = routes_from_active_x_t(active_x, active_t, pd)
    # travel = TravelCost(pd)

    # print("Solution:", sol)

if __name__ == "__main__":
    main()
