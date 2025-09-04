import pickle
import os

# Define the parameters used in file naming
nr = 4          # Number of RNs
nl = 3          # Number of LVNs
m = 25          # Number of events
time_limit = 300  # in seconds
event_limit = None  # or replace with an integer if used
pruning = 0
min_hour = 0
seed_number = 42

# Construct the file path
# file_path = f'/Users/jinghongmiao/Code/mvt-code/result-250521/ca_{nr}_{nl}_{m}_tl{time_limit}_el{event_limit}_p{pruning}_mh{min_hour}_max25_seed{seed_number}.pkl'
file_path = '/Users/jinghongmiao/Code/mvt-code/result-250812/ca_r20_l30_m50_wl1000_max25_19.pkl'

# Load and display the summary
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        summary = pickle.load(f)

    print("\n===== Summary from Pickle File =====")
    print(f"Objective Value: {summary['objective_value']}")
    print(f"Runtime (sec): {summary['runtime_sec']}")
    print(f"MIP Gap: {summary['gap']}\n")

    print("Top 5 Active Assignments (x[i,j,d,w]):")
    for x_entry in summary.get('active_x', [])[:5]:
        print(f"  {x_entry}")

    print("\nScheduled Events (s[i,d]):")
    for s_entry in summary.get('active_s', []):
        print(f"  {s_entry}")

    print("\nStart Times (t[i,d]):")
    for (i, d), start_time in summary.get('active_t', {}).items():
        print(f"  Event {i} on Day {d}: starts at {start_time}")


    print("\nPick-up Leaders (alpha[i,d,w]):")
    for alpha_entry in summary.get('active_alpha', []):
        print(f"  {alpha_entry}")

    print("\nDrop-off Leaders (beta[i,d,w]):")
    for beta_entry in summary.get('active_beta', []):
        print(f"  {beta_entry}")

else:
    print(f"File not found: {file_path}")
