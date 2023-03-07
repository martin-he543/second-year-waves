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
import sys, difflib
np.set_printoptions(threshold=sys.maxsize)

titleFont =      {'fontname': 'C059', 'size': 14, 'weight': 'bold'}
subtitleFont =   {'fontname': 'C059', 'size': 8, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'SF Mono', 'size': 7, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 7.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =      {'fontname': 'SF Mono', 'size': 7}
errorStyle =     {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =     {'mew': 1, 'ms': 3, 'color': 'blue'}
pointStyleR =    {'mew': 1, 'ms': 3, 'color': 'red'}
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


node, mean1C1, std1C1 , mean1C2, std1C2, mean2C1, std2C1 , mean2C2, std2C2 = np.loadtxt("3.6_RiseTime_Oscilloscope.csv", unpack=True, delimiter=",", skiprows=1)

plt.suptitle("Rise Times by Node, Original", **titleFont)
plt.title("[2023.03.03]", **subtitleFont)
plt.plot(node, mean1C1, 'x-', label="Voltage 1, $V_{1}$ / V", **pointStyle)
plt.plot(node, mean1C2, 'x-', label="Voltage 2, $V_{2}$ / V", **pointStyleR)
# plt.errorbar(node, mean1C1, yerr=std1C1, **errorStyle)
# plt.errorbar(node, mean1C2, yerr=std1C2, **errorStyle)
plt.xlabel("Node Number", **axesFont)
plt.ylabel("Time / μs", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="upper left", prop=font)
plt.show()

plt.suptitle("Rise Times by Node, Switched", **titleFont)
plt.title("[2023.03.03]", **subtitleFont)
plt.plot(node, mean2C1, 'x-', label="Voltage 1, $V_{1}$ / V", **pointStyle)
plt.plot(node, mean2C2, 'x-', label="Voltage 2, $V_{2}$ / V", **pointStyleR)
# plt.errorbar(node, mean2C1, yerr=std2C1, **errorStyle)
# plt.errorbar(node, mean2C2, yerr=std2C2, **errorStyle)
plt.xlabel("Node Number", **axesFont)
plt.ylabel("Time / μs", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="upper left", prop=font)
plt.show()
