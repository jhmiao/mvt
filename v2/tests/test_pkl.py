import pickle
import pandas as pd
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.solution import Route, Solution
from src.io.data_loader import load_problem_data, load_problem_data_sample
from src.models.context import Context
from src.solver.extract import routes_from_active_x_t

# file_path = '/Users/jinghongmiao/Code/mvt-code/v2/outputs/full_4037848/c101-a/c101-a_summary.pkl'
# for nr in [5,10,15,20,25]:
#     nl = int(nr / 5 * 7)
#     m = int(nr / 5 * 12)
#     n = nr + nl
#     for n_tw in [1,2,3]:
#         # pd = load_problem_data_sample('data/raw/real_data.xlsx', nr=nr, nl=nl, m=m, n_tw=n_tw)
#         summary_path = f"/Users/jinghongmiao/Code/mvt-code/v2/outputs/real-experiments/real_nr{nr}_nl{nl}_m{m}_ntw{n_tw}_summary.pkl"
#         try:
#             print(f"real_nr{nr}_nl{nl}_m{m}_ntw{n_tw}")
#             with open(summary_path, "rb") as f:
#                 summary = pickle.load(f)
#             print(f"Objective value = {summary.get('objective_value')}")
#             print(f"Bound = {summary.get('bound')}")
#             print(f"Gap = {summary.get('gap')}")
#             print(f"Runtime (sec) = {summary.get('runtime_sec')}")
#             print(f"Work limit = {summary.get('work_limit')}")
#             print("-----")
#         except Exception as e:
#             print(f"Error - {e}")
# for base in base_cases:
#     for suf in suffixes:
#         case_name = f"{base}{suf}"
#         # Build file path for summary and problem data
#         summary_path = f"/Users/jinghongmiao/Code/mvt-code/v2/outputs/full-20/{case_name}_summary.pkl"
#         data_path = f"data/raw/{base}/{base}_data.xlsx"
#         try:
#             pd = load_problem_data(data_path, type='continuous')
#             print(case_name)
#             with open(summary_path, "rb") as f:
#                 summary = pickle.load(f)
#             print(f"{case_name}: Objective value = {summary.get('objective_value')}")
#             print(f"{case_name}: Gap = {summary.get('gap')}")
#         except Exception as e:
#             print(f"{case_name}: Error - {e}")

# summary_path = f"/Users/jinghongmiao/Code/mvt-code/v2/outputs/c101-a_greedy_0,1,2,3,4.pkl"
# pd = load_problem_data('data/raw/c101-tw/c101-a.xlsx', type='continuous')
# with open(summary_path, "rb") as f:
#     summary = pickle.load(f)
# print(f"Objective value = {summary.get('objective_value')}")
# print(f"Gap = {summary.get('gap')}")
# # file_path = '/Users/jinghongmiao/Code/mvt-code/result-250805/ca_r12_l12_m15_wl1000_max40_18.pkl'
# # file_path = '/Users/jinghongmiao/Code/mvt-code/v2/outputs/full_4037848/c101-a/c101-a_summary.pkl'
# # pd = load_problem_data('data/raw/c101/c101_data.xlsx', type='continuous')

# # with open(file_path, "rb") as f:import pickle
# # import pandas as pd
# # import sys, pathlib

# base_cases = ["c101", "c201", "r101", "rc101"]
# suffixes = ["-a", "-b", "-c", "-d", "-e"]

# for base in base_cases:
#     for suf in suffixes:
#         case_name = f"{base}{suf}"
#         # Build file path for summary and problem data
#         summary_path = f"/Users/jinghongmiao/Code/mvt-code/v2/outputs/full_4037848/{case_name}/{case_name}_summary.pkl"
#         data_path = f"data/raw/{base}/{base}_data.xlsx"
#         try:
#             pd = load_problem_data(data_path, type='continuous')
#             print(case_name)
#             with open(summary_path, "rb") as f:
#                 summary = pickle.load(f)
#             print(f"{case_name}: Objective value = {summary.get('objective_value')}")
#             print(f"{case_name}: Gap = {summary.get('gap')}")
#         except Exception as e:
#             print(f"{case_name}: Error - {e}")

    # summary = pickle.load(f)


# print("Objective value:", summary["objective_value"])

# sol = routes_from_active_x_t(summary["active_x"], summary["active_t"], pd)
# print("Initial solution routes:")
# for route in sol.iter_routes():
#     print(route)
# # --- feasibility check
# ctx = Context(pd)
# rep0 = ctx.check_solution(sol)

# print("Feasible?", rep0.ok)
# print("Run time (s):", summary.get("runtime_sec"), sum(summary.get("runtime_sec", [])))
# # if rep0.ok is False:
    # print("Details:", rep0.violations)

# now you can inspect what’s inside:
# print("Active x:", summary["active_x"])
# print("Active s:", summary["active_s"])
# print("Active t:", summary["active_t"])
# print("Active alpha:", summary["active_alpha"])
# print("Active beta:", summary["active_beta"])
# print("Work time by nurse:", summary["work_time_by_nurse"])
# print("Objective value:", summary["objective_value"])

summary_path0 = f"/Users/jinghongmiao/Code/mvt-code/v2/outputs/rc101-perm/"
pd = load_problem_data('data/raw/rc101-tw/rc101-a.xlsx', type='continuous')

# go through files in the directory until reaching .pkl files
# go deeper into subdirectories if needed
# ...existing code...
import os

pkl_found = False
for root, dirs, files in os.walk(summary_path0):
    for file in files:
        if file.endswith(".pkl"):
            pkl_found = True
            full_path = os.path.join(root, file)
            print(os.path.relpath(full_path, summary_path0))
            try:
                with open(full_path, "rb") as f:
                    summary = pickle.load(f)
                print(f"Objective value = {summary.get('objective_value')}")
                # print(f"Gap = {summary.get('gap')}")
            except Exception as e:
                print(f"Error reading {full_path}: {e}")

if not pkl_found:
    print(f"No .pkl files found under {summary_path0}")
# ...existing code...