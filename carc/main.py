from data_loader import load_problem_data
from algorithms.continuous import continuous_algorithm
from algorithms.daily_greedy import daily_greedy_heuristic
# from algorithms.helpers import get_sorted_travel_time_cumsum
import numpy as np
import pickle

# time_limit = 5000 # in seconds
work_limit = 1000

seed = 19

def main():

    # continuous algorithm
    data_continuous = load_problem_data('c101-e.xlsx', type='continuous')
    cap = 25
    max_hour_list = np.ones(data_continuous.n) * cap

    # pickel_file_path = '/Users/jinghongmiao/Code/mvt-code/result-250819/h_r20_l30_m50_wl1000_max25_19.pkl'
    # sol = load_solution_from_pickle(pickel_file_path, data_continuous)
    # # print C_depot_e and C_depot_h
    # print("C_depot_e:", data_continuous.C_depot_e)
    # print("C_depot_h:", data_continuous.C_depot_h)
    # # View a specific nurse/day route:
    # d, w = 0, 9
    # print(f"Route for day {d}, nurse {w}:", sol.day_routes[(d,w)].nodes)
    # 3) Inspect or compute travel
    # print("Total travel (min):", total_travel_minutes(data_continuous, sol))
    # save to csv
    # get_sorted_travel_time_cumsum(data_continuous).to_csv('sorted_travel_time_cumsum.csv', index=True)

    # result = greedy_event_clustering(data_continuous)
    # print("Clusters of events for scheduling:", result)
    # heuristic_combine(data_continuous, 480)


    # summary_continuous = continuous_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=1, min_hour=None, max_hour=max_hour_list)
    # print(summary_continuous)
    # out_path1 = f'/Users/jinghongmiao/Code/mvt-code/result-250819/ca_r20_l30_m50_wl{work_limit}_max{cap}_{seed}.pkl'
    # Save as pickle
    # with open(out_path1, "wb") as f:
        # pickle.dump(summary_continuous, f)
    # print(f"Continuous algorithm result saved to {out_path1}")

    # heuristic algorithm
    # summary_heuristic = daily_greedy_heuristic(data_continuous, work_limit, seed, event_limit=None, pruning=2, update_max_hour=True, min_hour=None)
    summary_heuristic = daily_greedy_heuristic(data_continuous, work_limit, seed, event_limit=None, pruning=2, update_max_hour=True, min_hour=None)
    print(summary_heuristic['objective_value'])
    # out_path2 = f'/Users/jinghongmiao/Code/mvt-code/result-250812/h_r20_l30_m50_wl{work_limit}_max{cap}_{seed}.pkl'
    # # Save as pickle
    # with open(out_path2, "wb") as f:
    #     pickle.dump(summary_heuristic, f)
    # print(f"Heuristic result saved to {out_path2}")

    # summary_warm_start = continuous_warm_start(data_continuous, work_limit, seed, event_limit=None, pruning=1, min_hour=None, max_hour=max_hour_list)
    # out_path3 = f'/Users/jinghongmiao/Code/mvt-code/result-250722/warm_r10_l10_m10_wl{work_limit}_max{cap}_{seed}.pkl'
    # # Save as pickle
    # with open(out_path3, "wb") as f:
    #     pickle.dump(summary_warm_start, f)
    # print(f"Continuous algorithm result saved to {out_path3}")

if __name__ == "__main__":
    main()
# data_continuous = load_problem_data('problem_data_1.xlsx', type = 'continuous')




# print("max hour list:", max_hour_list.shape)
# print(continuous_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=2, min_hour=0, max_hour=max_hour_list))
# print(continuous_algorithm_heuristic(data_continuous, work_limit, seed_number=seed, event_limit=None, pruning=1, min_hour=0))

# continuous_fairness_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=2, min_hour=0, max_hour=None, fairness_param=10000)

# continuous_fairness_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=2, min_hour=0, max_hour=None, fairness_param=50)

# continuous_fairness_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=2, min_hour=0, max_hour=None, fairness_param=10)

# continuous_fairness_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=2, min_hour=0, max_hour=None, fairness_param=1)

# print(continuous_algorithm(data_continuous, work_limit, seed, event_limit=None, pruning=2, min_hour=0, max_hour=None))



# continuous_algorithm(data_continuous, work_limit, seed, multiple_tw=1, event_limit=2, pruning=2, min_hour=0)
# print(continuous_algorithm_daily(data_continuous, work_limit, 4, event_limit=None, pruning=2, min_hour=0))
# print(continuous_algorithm_heuristic(data_continuous, work_limit, 5, event_limit=0, pruning=0, min_hour=0))
# continuous_algorithm_daily(data_continuous, work_limit, seed_number=seed, event_limit=0, pruning=0, min_hour=0)



# data_discrete = load_problem_data('problem_data_1.xlsx', type = 'discrete')
# print(discrete_algorithm(data_discrete, work_limit, seed, multiple_tw=2, full_day=300, event_limit=None, pruning=2, min_hour=0))
# print(discrete_algorithm(data_discrete, work_limit, seed, multiple_tw=3, full_day=300, event_limit=None, pruning=2, min_hour=0))
# print(discrete_algorithm(data_discrete, work_limit, seed, multiple_tw=4, full_day=300, event_limit=None, pruning=2, min_hour=0))

# discrete_algorithm_daily(data_discrete, work_limit, 4, full_day=300, event_limit=None, pruning=2, min_hour=0)