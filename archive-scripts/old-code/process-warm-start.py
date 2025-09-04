import numpy as np

m = 61  # number of events
nr = 23  # number of RNs
nl = 35  # number of LVNs
n = nr + nl  # total number of nurses
block = 5 # number of blocks/days


# initiate a 4d numpy array of zeros, of size (75+2) x (75+2) x 5 x 50
# the first 2 dimensions represent the event number, plus 2 for the home and depot
# the third dimension represents the day number
# the fourth dimension represents the nurse number
x_matrix = np.zeros((m+2, m+2, block, n))
s_matrix = np.zeros((m, block))
t_matrix = np.zeros((m, block))
alpha_matrix = np.zeros((m, block, n))
beta_matrix = np.zeros((m, block, n))

with open("input_real_2event.txt", "r") as file:
    # for each line starting from the 7th line
    for i, line in enumerate(file):
        if i >= 4:
            # get the event number and the nurse number
            # each line is of form x_0_*_2_0, 0 where x_0_*_2_0 is the variable name and 0 is the value
            var_name, var_value = line.split(", ")
            var_name = var_name.split("_")

            if var_name[0] == 'x':
                # -1 denotes the depot, -2 denotes the home
                event_number_1 = int(var_name[1])
                event_number_2 = int(var_name[2])
                day_number = int(var_name[3])
                nurse_number = int(var_name[4])
                # store the value in the x_matrix
                x_matrix[event_number_1][event_number_2][day_number][nurse_number] = int(var_value)

            elif var_name[0] == 's':
                event_number = int(var_name[1])
                day_number = int(var_name[2])
                s_matrix[event_number][day_number] = int(float(var_value))
                
            elif var_name[0] == 't':
                event_number = int(var_name[1])
                day_number = int(var_name[2])
                t_matrix[event_number][day_number] = int(float(var_value))
            elif var_name[0] == 'alpha':
                event_number = int(var_name[1])
                day_number = int(var_name[2])
                nurse_number = int(var_name[3])
                alpha_matrix[event_number][day_number][nurse_number] = int(float(var_value))
            elif var_name[0] == 'beta':
                event_number = int(var_name[1])
                day_number = int(var_name[2])
                nurse_number = int(var_name[3])
                beta_matrix[event_number][day_number][nurse_number] = int(float(var_value))

np.save("x_matrix.npy", x_matrix)  # Save as a binary .npy file
np.save("s_matrix.npy", s_matrix)  # Save as a binary .npy file
np.save("t_matrix.npy", t_matrix)  # Save as a binary .npy file
np.save("alpha_matrix.npy", alpha_matrix)  # Save as a binary .npy file
np.save("beta_matrix.npy", beta_matrix)  # Save as a binary .npy file
