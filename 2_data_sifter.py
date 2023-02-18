import numpy as np

x2, y2 = np.loadtxt("thermal_4min_b.txt", unpack=True,skiprows=3)
new_x2, new_y2 = [], []

floored_value = int(np.floor(len(x2)/1000))
for i in range(int(len(x2)/floored_value)):
    new_x2.append(x2[i*floored_value])
    new_y2.append(y2[i*floored_value])
    
np.savetxt("thermal_4min_b_1000.txt", np.column_stack((new_x2, new_y2)), fmt="%.3f", delimiter="\t")