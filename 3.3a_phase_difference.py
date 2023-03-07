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


text_files = [["SAVE12","SAVE14"], ["SAVE15","SAVE16"], ["SAVE17","SAVE18"], ["SAVE19","SAVE20"], ["SAVE21","SAVE22"], ["SAVE23","SAVE24"], ["SAVE25","SAVE26"], ["SAVE27","SAVE28"], ["SAVE29","SAVE30"], ["SAVE31","SAVE32"]]
nodes = [[1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], [1,9], [1,10], [1,11]]

for i in range(len(text_files)):
    t1, V1 = np.loadtxt("Data/2023.02.28/" + text_files[i][0] + "/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
    t2, V2 = np.loadtxt("Data/2023.02.28/" + text_files[i][1] + "/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
    
    plt.suptitle("Task 3.3: Phase Difference Relationship", **titleFont)
    plt.title("Node {:.0f} and {:.0f}".format(nodes[i][0], nodes[i][1]), **subtitleFont)
    plt.plot(t1, V1, label="Voltage 1, $V_{1}$ / V")
    plt.plot(t2, V2, label="Voltage 2, $V_{2}$ / V")
    
    plt.xlabel("Time, t / s", **axesFont)
    plt.ylabel("Voltage, V / V", **axesFont)
    plt.xticks( **ticksFont)
    plt.yticks( **ticksFont)
    plt.legend(loc="upper right", prop=font)
    plt.show()