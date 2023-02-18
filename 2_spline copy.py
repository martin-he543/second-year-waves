import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit

titleFont =      {'fontname': 'C059', 'size': 12}
subtitleFont =   {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =      {'fontname': 'SF Mono', 'size': 7}
errorStyle =     {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =     {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =      {'linewidth': 0.5}
lineStyleR =     {'linewidth': 0.5, 'color': 'red'}
lineStyleG =     {'linewidth': 0.5, 'color': 'green'}
lineStyleBoldR = {'linewidth': 2, 'color': 'red'}
lineStyleBoldP = {'linewidth': 2, 'color': 'purple'}
lineStyleBoldG = {'linewidth': 2, 'color': 'green'}
histStyle =      {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =       {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font =           fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

x1, y1 = np.loadtxt("thermal_4min_a.txt", unpack=True,skiprows=3)
x2, y2 = np.loadtxt("thermal_4min_b.txt", unpack=True,skiprows=3)
tau, T_range = 1200, 50
order = [1, 2, 0, 4, 3, 5, 6, 7]

def Sinusoidal(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def DoubleSinusoidal(x, a, b, c, d, e, f, g):  return a * np.sin(b * x + c) + d * np.sin(e * x + f) + g
def Square(x, a, b, c, d):  return a * np.sign(np.sin(b * x + c)) + d

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], maxfev=100000)
print(fitting)

plt.plot(x2, y2, label='Fitted Sine Wave', **pointStyle)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Data', **lineStyleBoldR)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                 color='red', alpha=0.2, label='Difference (γ)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                 Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                 color='green', alpha=0.2, label='Difference (Δφ)')

plt.plot(x2, DoubleSinusoidal(x2, fitting[0], fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)

gamma, delta_phi = "{:.2f}".format(T_range/fitting[0]), "{:.2f}".format(tau - 1/(fitting[2]))
plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
plt.title("T = " + str(tau) + " ds; γ = " + str(gamma) + "; Δφ = " + str(delta_phi), **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
           loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
plt.show()

