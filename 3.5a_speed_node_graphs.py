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

phase, speed = np.loadtxt("3.5a_speeds.txt", unpack=True)
nodes = [[1,2],[2,4],[2,6],[3,7],[7,8],[9,11],[1,11],[6,11]]
node_diff = [4, 8, 16, 16, 4, 8, 40, 20]

def linear(x, a, b):    return a*x + b

cf_phase, cov_phase = curve_fit(linear, node_diff, phase)
cf_speed, cov_speed = curve_fit(linear, node_diff, speed)
cf_list = np.linspace(0, 40, 1000)

plt.suptitle("Task 3.5A: Phase Difference by Node Difference", **titleFont)
plt.title("Linear Fitting: m = " + formatter(cf_phase[0]) + "; c = " + formatter(cf_phase[1]), **subtitleFont)
plt.plot(node_diff, phase, 'x', label="Data", **pointStyle)
plt.plot(cf_list, linear(cf_list, *cf_phase), label="Linear Fit", **lineStyleR)
plt.xlabel("Node Difference", **axesFont)
plt.ylabel("Phase Difference / μs", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.ticklabel_format(useMathText=True)
plt.savefig("Plots/Task3.5A_Phase.png", dpi=1000)
plt.show()

plt.suptitle("Task 3.5A: Speed of Pulse Propagation by Node Difference", **titleFont)
plt.title("Linear Fitting: m = " + formatter(cf_speed[0]) + "; c = " + formatter(cf_speed[1]), **subtitleFont)
plt.plot(node_diff, speed, 'x', label="Data", **pointStyle)
plt.plot(cf_list, linear(cf_list, *cf_speed), label="Linear Fit", **lineStyleR)
plt.xlabel("Node Difference", **axesFont)
plt.ylabel("Speed of Pulse Propagation / m.s¯¹", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.ticklabel_format(useMathText=True)
plt.savefig("Plots/Task3.5A_Speed.png", dpi=1000)
plt.show()