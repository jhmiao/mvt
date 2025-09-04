import pandas as pd
import matplotlib.pyplot as plt

file_path1 = "compare-output-1.rtf"
file_path2 = "compare-output-2.rtf"
file_path3 = "compare-output-3.rtf"

def extract_data_from_rtf(file_path):
    rows = []
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Start from line 4 (index 3)
    for line in lines[3:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            row = parts[-5:]  # get last 5 entries
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=["objective", "bound", "gap", "It", "time"])
    # print(df.head(20))

    # Replace '-' with NaN
    df.replace("-", pd.NA, inplace=True)

    # Convert to appropriate numeric types
    df["objective"] = pd.to_numeric(df["objective"], errors="coerce")
    df["bound"] = pd.to_numeric(df["bound"], errors="coerce")
    df["gap"] = df["gap"].str.replace("%", "", regex=False)
    df["gap"] = pd.to_numeric(df["gap"], errors="coerce")
    df["It"] = pd.to_numeric(df["It"], errors="coerce")
    df["time"] = df["time"].str.replace("s\\", "", regex=False)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    # drop rows with all NaN values
    df.dropna(how='all', inplace=True)
    # drop column ["It"]
    df.drop(columns=["It"], inplace=True)

    return df

# Extract data from each RTF file
df1 = extract_data_from_rtf(file_path1) 
df2 = extract_data_from_rtf(file_path2)
df3 = extract_data_from_rtf(file_path3)

print(df1.tail())
print(df2.tail())
print(df3.tail())

# plot the data
# plot objective and bound against time for each dataframe, on the same plot
# plt.figure(figsize=(12, 6))
# plt.plot(df1["time"], df1["objective"], label="Baseline1", c='blue')
# plt.plot(df1["time"], df1["bound"], label="Bound 1", c='blue', linestyle='--')
# plt.plot(df2["time"], df2["objective"], label="Continuous2", c='orange')
# plt.plot(df2["time"], df2["bound"], label="Bound 2",c='orange', linestyle='--')
# plt.plot(df3["time"], df3["objective"], label="Discrete3", c='green')
# plt.plot(df3["time"], df3["bound"], label="Bound 3",c='green', linestyle='--')
# plt.xlabel("Time (s)")
# plt.ylabel("Value")
# plt.title("Objective and Bound vs Time")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("objective_bound_vs_time.png")
# plt.show()

# plot for "time" <= threshold (e.g., 1200)
threshold = 1200
plt.figure(figsize=(12, 6))
plt.plot(df1[df1["time"] <= threshold]["time"], df1[df1["time"] <= threshold]["objective"], label="Baseline1", c='blue')
plt.plot(df1[df1["time"] <= threshold]["time"], df1[df1["time"] <= threshold]["bound"], label="Bound 1", c='blue', linestyle='--')
plt.plot(df2[df2["time"] <= threshold]["time"], df2[df2["time"] <= threshold]["objective"], label="Continuous2", c='orange')
plt.plot(df2[df2["time"] <= threshold]["time"], df2[df2["time"] <= threshold]["bound"], label="Bound 2",c='orange', linestyle='--')
plt.plot(df3[df3["time"] <= threshold]["time"], df3[df3["time"] <= threshold]["objective"], label="Discrete3", c='green')
plt.plot(df3[df3["time"] <= threshold]["time"], df3[df3["time"] <= threshold]["bound"], label="Bound 3",c='green', linestyle='--')
plt.xlabel("Time (s)")
plt.title(f"Objective and Bound vs Time (Less than {threshold}s)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"objective_bound_vs_time_less_than_{threshold}.png")
plt.show()

# plot for "time" greater than a threshold (e.g., 1000)
# threshold = 1200
# plt.figure(figsize=(12, 6))
# plt.plot(df1[df1["time"] > threshold]["time"], df1[df1["time"] > threshold]["objective"], label="Baseline1", c='blue')
# plt.plot(df1[df1["time"] > threshold]["time"], df1[df1["time"] > threshold]["bound"], label="Bound 1", c='blue', linestyle='--')
# plt.plot(df2[df2["time"] > threshold]["time"], df2[df2["time"] > threshold]["objective"], label="Continuous2", c='orange')
# plt.plot(df2[df2["time"] > threshold]["time"], df2[df2["time"] > threshold]["bound"], label="Bound 2", c='orange', linestyle='--')
# plt.plot(df3[df3["time"] > threshold]["time"], df3[df3["time"] > threshold]["objective"], label="Discrete3", c='green')
# plt.plot(df3[df3["time"] > threshold]["time"], df3[df3["time"] > threshold]["bound"], label="Bound 3", c='green', linestyle='--')
# plt.xlabel("Time (s)")
# plt.ylabel("Value")
# plt.title(f"Objective and Bound vs Time (Greater than {threshold}s)")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"objective_bound_vs_time_greater_than_{threshold}.png")
# plt.show()
