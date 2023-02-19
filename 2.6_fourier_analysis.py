import numpy as np
import scipy as sp
import uncertainties as unc
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.fftpack import fft as fft
from scipy import interpolate
from uncertainties import ufloat
from uncertainties import unumpy

titleFont =      {'fontname': 'C059', 'size': 14, 'weight': 'bold'}
subtitleFont =   {'fontname': 'C059', 'size': 8, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =      {'fontname': 'SF Mono', 'size': 7}
errorStyle =     {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =     {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =      {'linewidth': 0.8}
lineStyleR =     {'linewidth': 0.8, 'color': 'red'}
lineStyleP =     {'linewidth': 0.8, 'color': 'purple'}
lineStyleG =     {'linewidth': 0.8, 'color': 'green'}
lineStyleBoldR = {'linewidth': 2, 'color': 'red'}
lineStyleBoldP = {'linewidth': 2, 'color': 'purple'}
lineStyleBoldG = {'linewidth': 2, 'color': 'green'}
histStyle =      {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =       {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font =           fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

#%% Task 2.6 - Fourier Analysis of a Signal

dataset_list = ["thermal_1min_a.txt","thermal_1min_b.txt","thermal_2min_a.txt", "thermal_2min_b.txt", "thermal_4min_a.txt", "thermal_4min_b.txt", "thermal_6min.txt", "thermal_8min.txt", "thermal_16min.txt"]

# subtitle_list = []
for i in range(len(dataset_list)):
    for j in range(1,21):
        path = dataset_list[i]
        time, x = np.loadtxt(path, skiprows=3, unpack=True); t_n = j
        X = fft(x); X[t_n:-t_n] = 0
        x_trunc = np.real(np.fft.ifft(X))
        amp = np.abs(X[:4]) / len(x) * 2
        phase = np.angle(X[:4])

        # for n in range(t_n):
        #     # subtitle_list.append([amp[n], phase[n]])
        #     print(f"A_{n} = {amp[n]:.4f}, Δφ_{n} = {phase[n]:.4f}")

        t = np.linspace(time[1], time[-1], len(x))
        plt.plot(t, x, label='Original Data', **pointStyle)
        plt.plot(t, x_trunc, label='Truncated Fourier Series', **lineStyleR)

        # subtitle_text = ""
        # for n in range(t_n):
        #     subtitle_text += f"A_{n} = {amp[n]:.4f}, Δφ_{n} = {phase[n]:.4f} \n"
        plt.suptitle("Task 2.6A: Truncating a Fourier Series, n = {:.0f}".format(t_n), **titleFont)
        plt.title(path, **subtitleFont)
        plt.xlabel("Time / ds", **axesFont)
        plt.ylabel("Temperature / °C", **axesFont)
        plt.xticks(**ticksFont)
        plt.yticks(**ticksFont)

        plt.legend(loc="best", prop=font)
        plt.savefig("Plots/Task2.6A_truncation_" + path.replace(".txt","") + "_n=" + str(t_n) + ".png", dpi=500)
        plt.clf()
        #plt.show()
        #print(subtitle_list)