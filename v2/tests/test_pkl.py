import pickle
import pandas as pd
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.solution import Route, Solution
from src.io.data_loader import load_problem_data
from src.models.context import Context
from src.solver.extract import routes_from_active_x_t

# file_path = '/Users/jinghongmiao/Code/mvt-code/result-250805/ca_r12_l12_m15_wl1000_max40_18.pkl'
file_path = '/Users/jinghongmiao/Code/mvt-code/v2/outputs/rc101_greedy_summary-2.pkl'
pd = load_problem_data('data/raw/rc101/rc101_data.xlsx', type='continuous')

with open(file_path, "rb") as f:
    summary = pickle.load(f)

# print("Objective value:", summary["objective_value"])

sol = routes_from_active_x_t(summary["active_x"], summary["active_t"], pd)
# print("Initial solution routes:")
# for route in sol.iter_routes():
#     print(route)
# --- feasibility check
ctx = Context(pd)
rep0 = ctx.check_solution(sol)

print("Feasible?", rep0.ok)
print("Run time (s):", summary.get("runtime_sec"), sum(summary.get("runtime_sec", [])))
# if rep0.ok is False:
    # print("Details:", rep0.violations)

# now you can inspect what’s inside:
# print("Active x:", summary["active_x"])
# print("Active s:", summary["active_s"])
# print("Active t:", summary["active_t"])
# print("Active alpha:", summary["active_alpha"])
# print("Active beta:", summary["active_beta"])
# print("Work time by nurse:", summary["work_time_by_nurse"])
# print("Objective value:", summary["objective_value"])

