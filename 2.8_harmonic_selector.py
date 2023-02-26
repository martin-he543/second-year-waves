import numpy as np

amp, phase = np.loadtxt('2.8_amp_phase_data_pure.txt', delimiter=",", unpack=True)

n = 50  # Choose number 1 bigger
amp_n, phase_n = [], []
for i in range(9):
    amp_n.append(amp[i*n - 2])
    phase_n.append(phase[i*n - 2])
    
print(amp_n)
print(phase_n)