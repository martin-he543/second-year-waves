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


text_files = ["SAVE33", "SAVE34", "SAVE35", "SAVE36", "SAVE37", "SAVE38", "SAVE39", "SAVE40", "SAVE41", "SAVE42", "SAVE43"]
nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
baseline = [-2.38, -2.39, -2.36, -2.32, -2.22, -2.08, -1.48, -0.14, -1.10, -1.58, -0.64]

t1, V1 = np.loadtxt("Data/2023.02.28/SAVE31/WFM.CSV", delimiter=",", skiprows=1, unpack=True)

for i in range(len(text_files)):
    t2, V2 = np.loadtxt("Data/2023.02.28/" + text_files[i] + "/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
    plt.suptitle("Task 3.3B: Phase Difference Relationship", **titleFont)
    # plt.title("Node {:.0f} and {:.0f}".format(nodes[i][0], nodes[i][1]), **subtitleFont)
    plt.plot(t1, V1 + 2.50, label=f"Original Voltage / V")
    plt.plot(t2, V2 - baseline[i], label=f"Voltage, Node 1 and {nodes[i]} / V")

    plt.xlabel("Time, t / s", **axesFont)
    plt.ylabel("Voltage, V / V", **axesFont)
    plt.xticks( **ticksFont)
    plt.yticks( **ticksFont)
    plt.legend(loc="upper right", prop=font)
    plt.show()