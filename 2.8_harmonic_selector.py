import numpy as np

amp, phase = np.loadtxt('2.7_fourier_output_amp_phase.txt', delimiter=",", unpack=True)
tau = [60, 60, 120, 120, 240, 240, 360, 480, 960]
n = 50  # Choose number 1 bigger
harm = 2 # Choose harmonic n = 51 - harm
amp_n, phase_n = [], []
for i in range(9):
    amp_n.append(amp[i*n - harm]/(200*np.pi))
    phase_n.append(phase[i*n - harm]/(tau[i]))
    
print(amp_n)
print(phase_n)