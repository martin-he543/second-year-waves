import numpy as np

amp, phase = np.loadtxt('2.8_amp_phase_data_pure.txt', delimiter=",", unpack=True)

n = 50  # Choose number 1 bigger
harm = 4 # Choose harmonic n = 51 - harm
amp_n, phase_n = [], []
for i in range(9):
    amp_n.append(amp[i*n - harm])
    phase_n.append(phase[i*n - harm])
    
print(amp_n)
print(phase_n)