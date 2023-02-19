import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate

titleFont =     {'fontname': 'C059', 'size': 12}
subtitleFont =  {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont ={'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1= {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2= {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'black', 'ls': '', 'linewidth': 0.5}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStylePurple =     {'linewidth': 0.5, 'color': 'purple'}
lineStyleBold = {'linewidth': 1, 'color': 'red'}
lineStyleBold1= {'linewidth': 1, 'color': 'green'}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =      {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font = fnt.FontProperties(family='C059', weight='normal', style='italic', size=7)

T, D1, D2, unc_D1, unc_D2 = np.loadtxt("2.5_data.tsv", delimiter="\t", skiprows=1, unpack=True)

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c
def logarithmic(x, a, b, c):
    return a * np.log(b * x) + c

x = np.linspace(0, 20, 1000)
total_D = D1 + D2
total_unc_D = unc_D1 + unc_D2

fit1, cov1 = curve_fit(exponential, T, D1, sigma=unc_D1, absolute_sigma=True, maxfev=1000000)
fit2, cov2 = curve_fit(exponential, T, D2, sigma=unc_D2, absolute_sigma=True, maxfev=1000000)
fitD, covD = curve_fit(exponential, T, total_D, sigma=total_unc_D, absolute_sigma=True, maxfev=1000000)

fig, ax = plt.subplots()
ax.plot(T, D1, 'x', label="D from Amplitude Transmission Factor", **lineStyleBold)
ax.errorbar(T, D1, label="Errorbars: D from Amplitude Transmission Factor", yerr=unc_D1, **errorStyle)
ax.plot(T, D2, 'x', label="D from Phase Difference", **lineStyleBold1)
ax.errorbar(T, D2, label="Errorbars: D from Phase Difference", yerr=unc_D2, **errorStyle)

ax.plot(x, exponential(x, *fit1), label="Fit: D from Amplitude Transmission Factor", **lineStyle)
ax.plot(x, exponential(x, *fit2), label="Fit: D from Phase Difference", **lineStyle)
ax.plot(x, exponential(x, *fitD), label="Fit: D from Both Methods", **lineStyle)

ax.axhline(1.24e-7, linestyle="dashed", label = "Literature Value", **lineStyle)
ax.axhline(1.24e-5, linestyle="dashed", label = "Compromise Value", **lineStylePurple)

ax.set_title("Time and Temperature", **titleFont)
ax.set_xlabel("Time / s", **axesFont)
ax.set_ylabel("Temperature / °C", **axesFont)
ax.ticklabel_format(useMathText=True)
ax.tick_params(axis='both', labelsize=7)
fig.legend(loc="lower right", prop=font, bbox_to_anchor=(0.60, 0.57))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.9, 20)
plt.savefig("Plots/2.5_summary_plot.png", bbox_inches="tight", dpi=1000)
plt.show()

variables = ["a", "b", "c"]
for i in range(3):
    print(variables[i],": ",fit1[i], "±", np.sqrt(cov1[i,i]))
for i in range(3):
    print(variables[i],": ",fit2[i], "±", np.sqrt(cov2[i,i]))
for i in range(3):
    print(variables[i],": ",fitD[i], "±", np.sqrt(covD[i,i]))