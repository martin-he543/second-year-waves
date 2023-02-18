import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate

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

#%% Task 2.3 - First ‘back of the envelope’ estimate of 𝑫.

x1, y1 = np.loadtxt("thermal_4min_a.txt", unpack=True,skiprows=3)
x2, y2 = np.loadtxt("thermal_4min_b.txt", unpack=True,skiprows=3)

def sinusodial(x, a, b, c, d):  return a * np.sin(b * x + c) + d

tau = 1200
sin1_fit, sin1_cov = curve_fit(sinusodial, x1, y1, p0=[10.3, -0.00264, 0, 52])
square_wave = -sin1_fit[0] * sp.signal.square(2 * np.pi * (0.5/tau) * x1) + sin1_fit[3]
square_wave_2 = -sin1_fit[0] * sp.signal.square(2 * np.pi * (0.5/tau) * x1)
sin2_fit, sin2_cov = curve_fit(sinusodial, x1, y1, p0=[10.3, -0.00264, 0, 52])

sin2_ufit = sinusodial(x2, 10, 0.00032, 11.9, 52)

plt.figure()
plt.plot(x1, sinusodial(x1, *sin1_fit), **lineStyleBold, label="Sinusodial Fit")
plt.plot(x1, square_wave, **lineStyleBold1, label="Square Wave")
plt.plot(x1,y1, **lineStyle, label="Data")
plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
plt.title("T = " + str(tau) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.legend(loc="lower left", bbox_to_anchor=(0.87, 0.995), prop=font)
plt.savefig("Plots/Task2.3_4A.png", dpi=1000)
plt.show()

plt.figure()
plt.plot(x2,y2, **lineStyle, label="Data")
plt.plot(x2, square_wave_2 + sin2_ufit, **lineStyleBold1, label="Square Wave")
plt.plot(x2, sin2_ufit, label="Sine Envelope")
plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
plt.title("T = " + str(tau) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.legend(loc="lower left", bbox_to_anchor=(0.87, 0.995), prop=font)
plt.savefig("Plots/Task2.3_4B.png", dpi=1000)
plt.show()

#%% Task 2.4 - Physical Meaning and Key Features of All Datasets.
dataset_list = ["thermal_1min_a.txt","thermal_1min_b.txt","thermal_2min_a.txt", "thermal_2min_b.txt", "thermal_4min_a.txt", "thermal_4min_b.txt", "thermal_6min.txt", "thermal_8min.txt", "thermal_16min.txt"]


for i in range(len(dataset_list)):
    x, y = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
    plt.plot(x,y, **lineStyle, label="Data")
    plt.suptitle("Thermalization of a System", **titleFont)
    plt.title(dataset_list[i], **subtitleFont)
    plt.xlabel("Time / s", **axesFont)
    plt.ylabel("Temperature / K", **axesFont)
    plt.xticks(**ticksFont)
    plt.yticks(**ticksFont)
    plt.legend(loc="upper right", prop=font)
    plt.show()
