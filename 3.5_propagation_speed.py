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


def substring_after(s, delim):  return s.partition(delim)[2]
def superscripter(string):
    superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '¯'}; new_string = ''
    for char in string:
        if char.lower() in superscript_map:
            new_string += superscript_map[char.lower()]
        else:   new_string += char
    return new_string
def formatter(D_gamma):
    D_gamma = "{:.4E}".format(D_gamma)
    D_gamma = D_gamma.replace("E"," ×10")
    D_gamma = D_gamma.replace("+/-"," ± ")
    exp_gamma = substring_after(D_gamma, " ×10")
    diff_gamma = ''.join([x[-1] for x in difflib.ndiff(exp_gamma, D_gamma) if x[0] != ' '])
    exp_gamma = int(exp_gamma)
    exp_gamma = str(exp_gamma)
    exp_gamma = superscripter(exp_gamma)
    D_gamma_final = diff_gamma + exp_gamma
    return(D_gamma_final)

text_files = [["SAVE49", "SAVE50"], ["SAVE51", "SAVE52"], ["SAVE54", "SAVE55"]]

for i in range(len(text_files)):
    t1, V1 = np.loadtxt("Data/2023.02.28/" + text_files[i][0] + "/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
    t2, V2 = np.loadtxt("Data/2023.02.28/" + text_files[i][1] + "/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
    t1_max = t1[np.argmax(V1)]
    t2_max = t2[np.argmax(V2)]
    phase_diff = np.abs(t2_max - t1_max)
    speed = np.abs(4/(t2_max - t1_max))
    formatted_phase_diff = formatter(phase_diff)
    formatted_speed = formatter(speed)

    plt.suptitle("Oscilloscope Trace", **titleFont)
    plt.title("[2023.02.28] Δφ = " + formatted_phase_diff + " s; v = " + formatted_speed + " m.s¯¹", **subtitleFont)
    plt.plot(t1, V1, label="Voltage 1, $V_{1}$ / V", **lineStyle)
    plt.plot(t2, V2, label="Voltage 2, $V_{2}$ / V", **lineStyleR)
    plt.xlabel("Time / s", **axesFont)
    plt.ylabel("Voltage / V", **axesFont)
    plt.xticks(**ticksFont)
    plt.yticks(**ticksFont)
    plt.ylim(-3, 0.5)

    plt.axvline(t1_max, **lineStyle)
    plt.annotate("t = {:.6f}s".format(t1_max), xy=(t1_max, -1.5), xytext=(t1_max, -1.5), rotation=270, **annotFontWeak)
    plt.axvline(t2_max, **lineStyleR)
    plt.annotate("t = {:.6f}s".format(t2_max), xy=(t2_max, -1.5), xytext=(t2_max, -1.5), rotation=270, **annotFontWeak)
    plt.fill_between(t1, -4, 1, where=(t1 >= np.min([t2_max, t1_max])) & (t1 <= np.max([t2_max, t1_max])), alpha=0.3, color='green')

    plt.annotate("", xy=(t2_max, -2.9), xytext=(t1_max, -2.9), arrowprops=dict(arrowstyle="<->", linewidth=0.3, color="black"))
    plt.annotate("Δφ = " + formatted_phase_diff + " s", xy=((t1_max + t2_max)/2, -2.8), xytext=((t1_max + t2_max)/2, -2.8), ha="center", va="center", color="black", **annotFontMini1)

    plt.legend(loc="upper right", prop=font)
    plt.show()