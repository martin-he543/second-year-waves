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

f, fSD, C1_Amp, C1_Amp_SD,  C2_Amp, C2_Amp_SD = np.loadtxt("3.7B_AmplitudeRatio.csv", delimiter=",", unpack=True, skiprows=1)
def linear(x, a, b):    return a*x + b
def exponential(x, a, b): return a*np.exp(b*x)

Amp_Ratio = C1_Amp / C2_Amp
Amp_Error = C1_Amp_SD + C2_Amp_SD

cf_exp, cov_exp = curve_fit(exponential, f, Amp_Ratio, maxfev = 100000)

# cf_linear, cov_linear = curve_fit(linear, f, Mean, maxfev = 100000)
plt.suptitle("Task 3.7A: Phase - Frequency Relationship", **titleFont)
plt.title("Phase Difference by Frequencies", **subtitleFont)

# plt.plot(f, C1_Amp, "o", **pointStyle, label="C1")
# plt.plot(f, C2_Amp, "o", **pointStyle, label="C2")
plt.plot(f, Amp_Ratio, "o", **pointStyle, label="C1/C2")
# plt.plot(f, exponential(f, *cf_exp), **lineStyleR, label="Exponential Fit")
plt.errorbar(f, Amp_Ratio, yerr=Amp_Error, **errorStyle)
plt.errorbar(f, Amp_Ratio, xerr=fSD, **errorStyle)

plt.xlabel("Frequency / Hz", **axesFont)
plt.ylabel("Amplitude Ratio", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.legend(loc="best", prop=font)
plt.show()