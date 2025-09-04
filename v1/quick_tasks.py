import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Read the Excel file
file_path = 'input_parameters_real.xlsx'

# Read the settings (time_limit, seed_number, nr, nl, m, block)
settings_df = pd.read_excel(file_path, sheet_name='Settings')
settings = settings_df.set_index('Parameter')['Value'].to_dict()

time_limit = int(settings['time_limit'])
seed_number = int(settings['seed_number'])
nr = int(settings['nr'])
nl = int(settings['nl'])
n = nr + nl
m = int(settings['m'])
block = int(settings['block'])


# Read the time_window matrix
time_window_df = pd.read_excel(file_path, sheet_name='Time_Window')
time_window_flat = time_window_df.values
time_window = time_window_flat.reshape((m, block, 2))

print(time_window[:10,:,:])


with open("input_15_35_75_seed120.txt", "r") as file:
    # print the number of lines in the file
    print(len(file.readlines()))

random.seed(10)

nr = 4
block = 5
m=10
T=12

min_values = np.array([1, 2, 3, 4, 5, 6, 7])
min_probabilities_RN = np.array([0.4, 0.4, 0.2, 0, 0, 0, 0])
min_probabilities_LVN = np.array([0.1, 0.3, 0.3, 0.1, 0.1, 0.05, 0.05])
min_nurse = np.zeros((m, 2), dtype=int)
for i in range(m):
    min_nurse[i][0] = np.random.choice(min_values, p = min_probabilities_RN)
    min_nurse[i][1] = np.random.choice(min_values, p = min_probabilities_LVN)

# print(min_nurse)

distance_matrix = np.random.randint(5, 26, size=(m, m))

# Step 2: Set the diagonal elements to 0
np.fill_diagonal(distance_matrix, 0)

time_window = np.zeros((m, T*block), dtype=int)

# print(distance_matrix)
# Populate the matrix with feasible time windows
for i in range(m):
    for j in range(block*T):
        # generate a full random time window table
        time_window[i, j] = np.random.choice([1,0],p=[0.2,0.8])

# print(time_window)
# print(15 % 10)

# print(sum(C_home[i][j] for i in range(2) for j in range(2)))

# Create a 20x15 matrix with positive integers around 15
rows, cols = 20, 15
matrix_20x15 = np.random.randint(5, 21, size=(rows, cols))

# print(matrix_20x15)

# time_window = np.array([[[240, 330],[300,330],[0,0],[0,0],[0,0]],
#                         [[0,0],[0,0],[120,240],[240,360],[0,0]],
#                         [[0,0],[0,0],[0,0],[0,0],[60,180]],
#                         [[0,0],[0,0],[270,270],[0,0],[0,0]],
#                         [[60,360],[0,0],[0,0],[0,0],[0,0]],
#                         [[120,210],[60,150],[0,0],[0,0],[0,0]],
#                         [[0,0],[0,0],[180,300],[180,300],[0,0]]])

# print(np.shape(time_window))

# x_*jdw = 1 if nurse w goes from depot to event j on day d, 0 otherwise
xdepot_event = [[[1 for w in range(nr)] for d in range(block)] for j in range(m)]
sum = np.sum(xdepot_event, axis=1)
# print(np.shape(sum))



C_depot = np.random.randint(5, 21, size=(m+nr))
tiled_array = np.tile(C_depot, (4, 1))
# print(tiled_array)


# Define the array of possible values
dur_values = np.array([1,2,3,4,6,8])
# Define the probability distribution
dur_probabilities = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # Adjust probabilities as needed
# Generate the array
C_dur = np.random.choice(dur_values, size=m, p = dur_probabilities)
# print(C_dur)

# Function to generate a random time window
def generate_time_window():
    start = np.random.randint(1, 11) * 30  # Start time: 0 to 300 by 30s
    end = np.random.randint(start // 30 + 1, 12) * 30  # End time: start + 30 to 330 by 30s
    return [start, end]

# Initialize the matrix
time_window = np.zeros((m, block, 2), dtype=int)

# Populate the matrix with feasible time windows
for i in range(m):
    for j in range(block):
        time_window[i, j] = random.choice([[0,0], generate_time_window()])

# print(time_window)

def get_time(minute):
    hour = minute // 60 + 9
    minute = minute % 60
    if minute < 10:
        minute = "0" + str(minute)
    return f"{hour}:{minute}"

# print(get_time(240))  # 13:00
# print(int(15.0))


matrix = np.array([[0, 0, 0, 1, 1],
                   [0, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0]])
row1 = np.where(np.sum(matrix[:, 0:4], axis=1) == 1)[0][0]
row2 = np.where(np.sum(matrix[:, 0:4], axis=1) == 1)[0][1]
y_day = np.sum(np.sum(matrix[:, 0:4], axis=0))
# print(y_day)  

# read quick_tasks.txt
gap = []
runtime = []
with open("quick_task.txt", "r") as file:
    data = file.read().splitlines()
    # for each line in the file
    for line in data:
        # split the line into a list of values
        values = line.split()
        # if the first value is 'H'
        if values[0] == 'H':
            # save the last value as 'runtime' and the third to last value as 'gap'
            # take 's' off the end of the runtime value and convert it to a float
            runtime.append(float(values[-1][:-1]))
            gap.append(float(values[-3][:-1]))

# plot gas vs runtime
plt.plot(runtime,gap)
plt.ylabel("Gap (%)")
plt.xlabel("Runtime (s)")
plt.title("Gap vs Runtime")
plt.show()