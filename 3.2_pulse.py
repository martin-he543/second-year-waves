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


t1, V1 = np.loadtxt("Data/2023.02.28/SAVE03/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
plt.suptitle("Task 3.2: Square Pulse Function Generator", **titleFont)
plt.title("Node 1", **subtitleFont)
plt.plot(t1, V1, label="Voltage 1, $V_{1}$ / V")

plt.xlabel("Time, t / s", **axesFont)
plt.ylabel("Voltage, V / V", **axesFont)
plt.xticks( **ticksFont)
plt.yticks( **ticksFont)
plt.legend(loc="upper right", prop=font)
plt.show()