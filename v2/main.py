import json
import os, sys, pickle
from unittest import case
import numpy as np

# Add project root, not src
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(project_root)

from src.models.context import Context
from src.io.data_loader import load_problem_data, load_problem_data_sample
from src.solver.continuous import continuous_algorithm, continuous_warm_start, continuous_algorithm_discrete, continuous_algorithm_fix_events
from src.solver.continuous_no_depot import continuous_algorithm_no_depot
from src.solver.continuous_no_depot_discount import continuous_algorithm_no_depot_ratio
from src.solver.continuous_lazy import continuous_algorithm_lazy
from src.solver.extract import routes_from_active_x_t
from src.models.travel_cost import TravelCost
# from src.heuristics.lns_gurobi import LNSCfg, lns_with_gurobi, build_full_model
from src.heuristics.new_lns import lns_day, build_full_model
from src.heuristics.daily_greedy import continuous_algorithm_heuristic, compute_day_complexity
from src.solver.continuous_penalty import continuous_algorithm_penalty, continuous_algorithm_adaptive_penalty
from src.heuristics.add_depot import depot_delta_cost, solve_depot, add_depot_to_solution, debug_day_nurses
from src.heuristics.construction import assign_events
from src.heuristics.assignment import assign_nurses, assign_nurses_with_depot, merge_assignments, solution_from_assignments


def main():

    # ======== Exact optimziation experiements with (subsetted) real data ========
    # loop through different combinations of nr, nl, m, and n_tw
    # for nr in [5,10,15,20,25]:
    # # for nr in [20]:
    #     nl = int(nr / 5 * 7)
    #     m = int(nr / 5 * 12)
    #     n = nr + nl
    #     for n_tw in [1,2,3]:
    #         pd = load_problem_data_sample('data/raw/real_data.xlsx', nr=nr, nl=nl, m=m, n_tw=n_tw)
    #         summary = continuous_algorithm(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))
    #         with open(f'outputs/real_nr{nr}_nl{nl}_m{m}_ntw{n_tw}_summary.pkl', 'wb') as f:
    #             pickle.dump(summary, f)

    # ======== Experiments on solomon location distribution data ========
    # loop through different problem types
    case = 'r101'
    for type in ['a']:
    # for type in ['e']:
        pd = load_problem_data(f'v2/data/raw/{case}-tw/{case}-{type}.xlsx', type='continuous')
        # event_assignments = assign_events(pd)
        # save 
        # with open(f'v2/outputs/{case}_{type}_event_assignments.json', 'w') as f:
        #     json.dump(event_assignments, f, indent=2)

        chosen_schedules = {}
        feasible_arcs = {}

        
        # a loop for patterns
        with open(f"v2/outputs/{case}_{type}_event_assignments.json") as f:
            all_day_patterns = json.load(f)
            with open(f'v2/outputs/{case}_{type}_pattern_assignments_value.txt', 'w') as f:
                for pattern_id in range(300):
                    for day_str, patterns in all_day_patterns.items():
                        day = int(day_str)
                        if len(patterns) > 0:
                            chosen_schedules[day] = patterns[pattern_id]["schedule"]
                            feasible_arcs[day] = patterns[pattern_id]["pairs"]
                        else:
                            chosen_schedules[day] = {}  # no patterns for that day
                            feasible_arcs[day] = {}
                        # print(chosen_schedules[day], feasible_arcs[day])
                    assignments_value = assign_nurses_with_depot(pd, 'Both', chosen_schedules, feasible_arcs)
                    # save pattern+_id and corresponding assignments_value
                    # f.write(f"Pattern ID: {pattern_id}, Assignment Value: {assignments_value}\n")
        break
        # read json from 'v2/outputs/{case}_{type}_event_assignments.json'
        with open(f"v2/outputs/{case}_{type}_event_assignments.json") as f:
            all_day_patterns = json.load(f)
            for day_str, patterns in all_day_patterns.items():
                day = int(day_str)
                if len(patterns) > 0:
                    chosen_schedules[day] = patterns[104]["schedule"]
                    feasible_arcs[day] = patterns[104]["pairs"]
                else:
                    chosen_schedules[day] = {}  # no patterns for that day
                    feasible_arcs[day] = {}
        # print("Chosen schedules:", chosen_schedules)
        # print("Arcs:", feasible_arcs)
        # assignments = assign_nurses_with_depot(pd, 'RN', chosen_schedules, feasible_arcs)
        # assignments_LVN = assign_nurses(pd, 'LVN', chosen_schedules, feasible_arcs)
        assignments_both = assign_nurses_with_depot(pd, 'Both', chosen_schedules, feasible_arcs)


        assignment_rn = assign_nurses_with_depot(pd, 'RN', chosen_schedules, feasible_arcs)
        print("RN assignment:", assignment_rn)
        assignments_RN = assign_nurses(pd, 'RN', chosen_schedules, feasible_arcs)

        print("RN assignments:", assignments_RN)
        print("LVN assignments:", assignments_LVN)
        merged_assignments = merge_assignments(assignments_RN, assignments_LVN)
        # print("Merged assignments:", merged_assignments)
        
        sol = solution_from_assignments(pd, merged_assignments, chosen_schedules)
        # print("Constructed solution:", sol)

        # summary = continuous_algorithm_fix_events(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n), chosen_schedules=chosen_schedules, feasible_arcs=None)

        # summary = continuous_algorithm_discrete(pd, work_limit=10000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))
        # summary = continuous_algorithm(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))

        # print(compute_day_complexity(pd))

        # # set day_order to be descending based on the second term of complexity
        # day_complexity = compute_day_complexity(pd)
        # day_order = sorted(range(pd.day), key=lambda d: day_complexity[d][2] + day_complexity[d][3], reverse=True)
        # print("Day order:", day_order)
        
        # # set day_weights be the fourth term of complexity
        # day_weights = [day_complexity[d][4] for d in range(pd.day)]
        # # normalize to sum to 1
        # total_weight = sum(day_weights)
        # if total_weight > 0:
        #     day_weights = [w / total_weight for w in day_weights]
        # print("Day weights:", day_weights)
        
        # summary = continuous_algorithm(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))
        # summary = continuous_algorithm(pd, work_limit=1000, seed_number=0, pruning=1)
        # summary = continuous_algorithm_heuristic(pd, work_limit=1000, seed_number=0, pruning=1, custom_day_order=day_order, day_weights=day_weights)
        
        # with open(f'outputs/informs-two-stage/c101_{type}_fullM_summary.pkl', 'wb') as f:
        #     pickle.dump(summary, f)

        # read pkl from 'outputs/c201_penalty_summary.pkl'
    #     case_name = f'{case}-{type}'
    #     # with open(f'outputs/full_4037848/{case_name}/{case_name}_summary.pkl', 'rb') as f:
    #     with open (f"/Users/jinghongmiao/Code/mvt-code/v2/outputs/full-20/{case_name}_summary.pkl", 'rb') as f:
    #     # with open(f'outputs/informs-two-stage/c101_{type}_fullM_summary.pkl', 'rb') as f:
    #         summary = pickle.load(f)

    #     active_x = summary["active_x"]          # list of (i,j,d,w)
    #     active_t = summary["active_t"]          # dict {(i,d): t}
    #     objective_value = summary["objective_value"]
    #     gap = summary["gap"]
    #     sol = routes_from_active_x_t(active_x, active_t, pd)
    #     # for route in sol.iter_routes():
    #     #     print(route)

    #     travel = TravelCost(pd)
    #     # get phase-1 travel cost
    #     total_travel_cost = travel.total_cost(sol)
    #     print(f"Objective value from solver: {objective_value}")
    #     print(f"Total travel cost: {total_travel_cost}")

        delta_cost = depot_delta_cost(pd, sol)
        # print(delta_cost)
    #     # debug_day_nurses(delta_cost, 2, [25, 26])
        AM_solution = solve_depot(pd, delta_cost, mode="AM")
        PM_solution = solve_depot(pd, delta_cost, mode="PM")
    # #     # new_sol = solve_depot(pd, delta_cost)
        print(f"Delta cost AM: {AM_solution}")
        print(f"Delta cost PM: {PM_solution}")
        ctx = Context(pd)
        rep = ctx.check_solution(sol)
        print("Initial feasible?", rep.ok)
        print("Initial details:", rep.violations)

    # #     # print(f"Original gap (no depot): {gap}")
    #     ctx = Context(pd)
    #     rep = ctx.check_solution(sol)
    #     print("Initial feasible?", rep.ok)
    #     # print("Initial details:", rep.violations)

    #     new_sol1 = add_depot_to_solution(pd, sol, AM_solution, mode="AM")
    #     new_sol2 = add_depot_to_solution(pd, new_sol1, PM_solution, mode="PM")
    #     new_travel_cost = travel.total_cost(new_sol2)
    #     print(f"New total travel cost (with depot): {new_travel_cost}")
    #     # for route in new_sol.iter_routes():
    #     #     print(route)

    #     ctx = Context(pd)
    #     rep = ctx.check_solution(new_sol2)
    #     print("Final feasible?", rep.ok)
    #     # print("Final details:", rep.violations)
    #     # print("Final obj:", new_sol.objective_value)

    #     # start LNS from new_sol
    #     # LNS iterations
    #     no_improv = 0
    #     for it in range(2):
    #         print("=============================================")
    #         print(f"Instance {case_name} starting LNS iteration {it}")
    #         improved = False
    #         for d in range(5):
    #             # Choose which days to free
    #             if no_improv >= 5:
    #                 # Free two consecutive days (wrap around)
    #                 d_next = (d + 1) % 5
    #                 print(f"No improvement in 5 rounds → freeing days {d} and {d_next}")
    #                 free_days = [d, d_next]
    #                 active_x, active_t, new_sol = lns_day(
    #                     sol, active_x, active_t, pd, ctx,
    #                     day_idx=free_days,    # <-- pass list of days to lns_day()
    #                     worklimit=800
    #                 )
    #             else:
    #                 free_days = [d]
    #                 active_x, active_t, new_sol = lns_day(
    #                     sol, active_x, active_t, pd, ctx,
    #                     day_idx=d,
    #                     worklimit=200
    #                 )


    #             print(f"Completed LNS for day(s): {free_days}")
    #             obj_new = travel.total_cost(new_sol)
    #             print("Objective after LNS:", obj_new)
    #             rep = ctx.check_solution(new_sol)
    #             print("Feasible?", rep.ok)

    #             # depot modification
    #             new_delta_cost = depot_delta_cost(pd, new_sol)
    #             # print(delta_cost)
    #             # debug_day_nurses(delta_cost, 2, [25, 26])
    #             AM_solution = solve_depot(pd, new_delta_cost, mode="AM")
    #             PM_solution = solve_depot(pd, new_delta_cost, mode="PM")
    #             new_sol1 = add_depot_to_solution(pd, new_sol, AM_solution, mode="AM")
    #             new_sol2 = add_depot_to_solution(pd, new_sol1, PM_solution, mode="PM")
    #             new_travel_cost = travel.total_cost(new_sol2)
    #             print(f"New total travel cost (with depot): {new_travel_cost}")


    #             # Check improvement
    #             if new_travel_cost < travel.total_cost(sol) - 1e-4:
    #                 sol = new_sol2
    #                 # update active_x
    #                 active_x = {}
    #                 for route in sol.iter_routes():
    #                     # get consecutive pairs of visits in the route
    #                     d = route.day_idx
    #                     w = route.nurse
    #                     nodes = route.nodes
    #                     for idx in range(len(nodes) - 1):
    #                         i = nodes[idx]
    #                         j = nodes[idx + 1]
    #                         # cast i and j to m if they are 100 + w
    #                         if i >= 100:
    #                             i = pd.m
    #                         if j >= 100:
    #                             j = pd.m
    #                         # set x(i,j,d,w) = 1
    #                         active_x[(i, j, d, w)] = 1.0

    #                 improved = True
    #                 no_improv = 0
    #                 print(f"Iteration {it}, improved on day {d}: obj={obj_new:.2f}")
    #             else:
    #                 no_improv += 1

    #         if not improved:
    #             print(f"No improvement in iteration {it}. Consecutive no-improvement count = {no_improv}")


        # compare solutions
        # if ctx.compare_solutions(sol, new_sol2):
        #     print("Solutions are identical.")
        # else:
        #     print("Solutions differ.")
        

            

    # ======== solver full solution ========
    # # run full solver for a long time and save the result
    # # summary0 = continuous_algorithm(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))  
    # # ctx = Context(pd)
    # # active_x0 = summary0["active_x"]          # list of (i,j,d,w)
    # # active_t0 = summary0["active_t"] 
    # # initial_sol = routes_from_active_x_t(active_x0, active_t0, pd)

    # # rep0 = ctx.check_solution(initial_sol)
    # # print("Initial feasible?", rep0.ok)
    # # with open('outputs/rc101_summary.pkl', 'wb') as f:
    # #     pickle.dump(summary0, f)
    # # file_path2 = 'outputs/c101_lns_summary.pkl'

    # # with open(file_path2, "rb") as f:
    # #     summary0 = pickle.load(f)

    

    # --- get an initial solution by running your full solver once
    # summary0 = continuous_algorithm(pd, work_limit=1000, seed_number=0, pruning=1, max_hour=25 * np.ones(pd.n))
    # # with open(f'outputs/c201_full-{type}.pkl', 'wb') as f:
    # #     pickle.dump(summary0, f)
    # active_x0 = summary0["active_x"]          # list of (i,j,d,w)
    # active_t0 = summary0["active_t"]          # dict {(i,d): t}
    # initial_sol = routes_from_active_x_t(active_x0, active_t0, pd)
    # # --- feasibility check
    # ctx = Context(pd)
    # rep0 = ctx.check_solution(initial_sol)
    # print("Initial feasible?", rep0.ok)

    # # --- run daily greedy heuristic
    # summary_heuristic = continuous_algorithm_heuristic(pd, work_limit=1000, seed_number=0, event_limit=None, pruning=2, min_hour=None)
    # # save summary
    # with open(f'outputs/r101_greedy-{type}.pkl', 'wb') as f:
    #     pickle.dump(summary_heuristic, f)

    # active_xh = summary_heuristic["active_x"]          # list of (i,j,d,w)
    # active_th = summary_heuristic["active_t"]          # dict {(i,d): t}
    # sol_heuristic = routes_from_active_x_t(active_xh, active_th, pd)
    # reph = ctx.check_solution(sol_heuristic)
    # print("Heuristic feasible?", reph.ok)
    # # print("Heuristic details:", reph.violations)
    # print("Heuristic obj:", summary_heuristic["objective_value"])

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
