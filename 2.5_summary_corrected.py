import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate
import uncertainties as unc
from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties import ufloat_fromstr
import difflib

titleFont =      {'fontname': 'C059', 'size': 14, 'weight': 'bold'}
subtitleFont =   {'fontname': 'C059', 'size': 8, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =      {'fontname': 'SF Mono', 'size': 7}
errorStyle =     {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': '', 'zorder': 1}
pointStyle =     {'mew': 1, 'ms': 3, 'color': 'blue', 'zorder': 10}
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

def formatter(D_gamma, D_delta_phi):
    D_gamma = "{:.4E}".format(D_gamma)
    D_delta_phi = "{:.4E}".format(D_delta_phi)
    D_gamma = D_gamma.replace("E"," ×10")
    D_delta_phi = D_delta_phi.replace("E"," ×10")
    D_gamma = D_gamma.replace("+/-"," ± ")
    D_delta_phi = D_delta_phi.replace("+/-"," ± ")

    exp_gamma = substring_after(D_gamma, " ×10")
    exp_delta_phi = substring_after(D_delta_phi, " ×10")

    diff_gamma = ''.join([x[-1] for x in difflib.ndiff(exp_gamma, D_gamma) if x[0] != ' '])
    diff_delta_phi = ''.join([x[-1] for x in difflib.ndiff(exp_delta_phi, D_delta_phi) if x[0] != ' '])

    exp_gamma = int(exp_gamma)
    exp_delta_phi = int(exp_delta_phi)
    exp_gamma = str(exp_gamma)
    exp_delta_phi = str(exp_delta_phi)
    exp_gamma = superscripter(exp_gamma)
    exp_delta_phi = superscripter(exp_delta_phi)

    D_gamma_final = diff_gamma + exp_gamma
    D_delta_phi_final = diff_delta_phi + exp_delta_phi

    return(D_gamma_final, D_delta_phi_final)


x = np.genfromtxt('2.5_data_corrected.txt', dtype="str")
x_uf = []
for i in range(len(x)):     x_uf.append(ufloat_fromstr(x[i]))

T = [60, 60, 120, 120, 240, 240, 360, 480, 960]
D1 = x_uf[:9]
D2 = x_uf[9:18]

D1_n, D1_s = [], []
D2_n, D2_s = [], []
for i in range(len(D1)): D1_n.append(D1[i].n); D1_s.append(D1[i].s)
for i in range(len(D2)): D2_n.append(D2[i].n); D2_s.append(D2[i].s)

D1_mean = np.mean(D1_n)
D2_mean = np.mean(D2_n)
D_mean = np.mean(x_uf)

def exponential(x, a, b, c):    return a * np.exp(b * x) + c
def logarithmic(x, a, b, c):    return a * np.log(b * x) + c
# fit1, cov1 = curve_fit(exponential, T, D1_n, sigma=D1_s, absolute_sigma=True, maxfev=1000000)
# fit2, cov2 = curve_fit(exponential, T, D2_n, sigma=D2_s, absolute_sigma=True, maxfev=1000000)

fig, ax = plt.subplots()
ax.plot(T, D1_n, 'x', label="D from Amplitude Transmission Factor", **lineStyleBoldR)
ax.errorbar(T, D1_n, label="Errorbars: D from Amplitude Transmission Factor", yerr=D1_s, **errorStyle)
ax.plot(T, D2_n, 'x', label="D from Phase Difference", **lineStyleBoldG)
ax.errorbar(T, D2_n, label="Errorbars: D from Phase Difference", yerr=D2_s, **errorStyle)

# ax.plot(x, exponential(x, *fit1), label="Fit: D from Amplitude Transmission Factor", **lineStyle)
# ax.plot(x, exponential(x, *fit2), label="Fit: D from Phase Difference", **lineStyle)
# ax.annotate("Literature Value", xy=(10, 1.24e-7), xytext=(10, 1.24e-7))
# ax.annotate("Compromise Value for $D_{γ}$", xy=(10, D1_mean), xytext=(10, D1_mean))
# ax.annotate("Compromise Value for $D_{Δφ}$", xy=(10, D2_mean), xytext=(10, D2_mean))
ax.axhline(1.24e-7, linestyle="dashed", label="Literature Value", **lineStyle)
ax.axhline(D1_mean, linestyle="dashed", label="Compromise Value for $D_{γ}$", **lineStyleR)
ax.axhline(D2_mean, linestyle="dashed", label="Compromise Value for $D_{Δφ}$", **lineStyleG)

# D1_mean_f = f"{D1_mean:.4E}"
# D2_mean_f = f"{D2_mean:.4E}"
# D_mean_f = f"{D_mean.n:.4E}"

D1_f, D2_f = formatter(np.mean(D1), np.mean(D2))
D_f, D_mean_f = formatter(D_mean, D_mean)

fig.suptitle("Task 2.5C: Summary Table of Values", **titleFont)
ax.set_title(f"$D_γ$: {D1_f} m².s¯¹; $D_Δᵩ$: {D2_f} m².s¯¹; $D_T$: {D_f} m².s¯¹", **subtitleFont)
ax.set_xlabel("Time / s", **axesFont)
ax.set_ylabel("Temperature / °C", **axesFont)
ax.ticklabel_format(useMathText=True)
ax.tick_params(axis='both', labelsize=7)
fig.legend(loc="lower right", prop=font, bbox_to_anchor=(0.60, 0.57))
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_xlim(0.9, 20)
plt.savefig("Plots/Task2.5_Summary.png", bbox_inches="tight", dpi=1000)
plt.show()

