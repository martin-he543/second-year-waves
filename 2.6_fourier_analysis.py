import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.fftpack import fft as fft

titleFont =     {'fontname': 'Bodoni 72', 'size': 12}
subtitleFont =  {'fontname': 'Bodoni 72', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'Bodoni 72', 'size': 9, 'style':'italic'}
annotationFont ={'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1= {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2= {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1, 'color': 'red'}
lineStyleBold1= {'linewidth': 1, 'color': 'green'}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =      {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font = fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

#%% Task 2.6 - Fourier Analysis of a Signal

time, x = np.loadtxt('thermal_1min_a.txt', skiprows=3, unpack=True)

t = np.linspace(time[1], time[-1], 2400)
X = fft(x)
X[111:-111] = 0
x_trunc = np.real(np.fft.ifft(X))

amp = np.abs(X[:4]) / len(x) * 2
phase = np.angle(X[:4])

for n in range(4):
    print(f"a_{n} = {amp[n]:.2f}, phi_{n} = {phase[n]:.2f}")

plt.plot(t, x, label='Original Data')
plt.plot(t, x_trunc, label='Truncated Fourier Series')
plt.legend()
plt.show()