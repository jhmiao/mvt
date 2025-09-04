import pickle


file_path1 = f'/Users/jinghongmiao/Code/mvt-code/result-250429/da_23_35_61_fd300_elNone_p2_mh10_seed2.pkl'
file_path2 = f'/Users/jinghongmiao/Code/mvt-code/result-250429/da_23_35_61_fd300_elNone_p2_mh0_seed2.pkl'


with open(file_path1, "rb") as f:
    model1 = pickle.load(f)

with open(file_path2, "rb") as f:
    model2 = pickle.load(f)

# Example: compare scheduled arcs
# diff_x = set(model1["active_x"]) ^ set(model2["active_x"])  # symmetric difference
# print(diff_x)

# Compare start time differences
for key in model1["event_start_times"]:
    if key in model2["event_start_times"]:
        delta = abs(model1["event_start_times"][key] - model2["event_start_times"][key])
        if delta > 0:
            print(f"Event {key} time differs by {delta} minutes.")
