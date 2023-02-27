import numpy as np
import scipy as sp
import uncertainties as unc
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy import interpolate
from uncertainties import ufloat
from uncertainties import unumpy
import difflib
from scipy.special import jn

titleFont =      {'fontname': 'C059', 'size': 14}
subtitleFont =   {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 7, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =      {'fontname': 'SF Mono', 'size': 7}
errorStyle =     {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =     {'mew': 1, 'ms': 3, 'color': 'blue'}
pointStyleR =    {'mew': 1, 'ms': 3, 'color': 'red'}
lineStyle =      {'linewidth': 0.5}
lineStyleR =     {'linewidth': 0.5, 'color': 'red'}
lineStyleP =     {'linewidth': 2, 'color': 'purple'}
lineStyleG =     {'linewidth': 0.5, 'color': 'green'}
lineStyleBoldR = {'linewidth': 2, 'color': 'red'}
lineStyleBoldP = {'linewidth': 2, 'color': 'purple'}
lineStyleBoldG = {'linewidth': 2, 'color': 'green'}
histStyle =      {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =       {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font =           fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

def exponential(x, a, b, c):    return a * np.exp(b * x) + c
def linear(x, a, b):            return a * x + b
def bessel(x, a, b, c, d):      return jn(0, a*x+b)/d + c
def quadratic(x, a, b, c):      return a * x**2 + b * x + c
#%% n = 49

T = [60, 60, 120, 120, 240, 240, 360, 480, 960]

D_gamma_cf = [1.5506609070308076e-07, 9.943464464268348e-08, 1.0976512933994218e-07, 1.255971360052302e-07, 1.206746049626923e-07, 1.286380841594988e-07, 1.3596768121084294e-07, 1.7268644535538645e-07, 1.5322550352182592e-07]
D_delta_phi_cf = [1.3681216024102855e-05, 0.007540244198096596, 3.6444892448750743e-06, 2.8354801941960967e-06, 9.958734671759949e-07, 8.941154721024194e-07, 5.286281606320546e-07, 3.312284623952762e-07, 1.399963568278951e-07]
D_gamma_cf = [1.4056238475637676e-07, 9.233515001917558e-08, 9.777087184771964e-08, 1.1108724752513297e-07, 1.0209307918714879e-07, 1.082428432243557e-07, 1.093242188131685e-07, 1.2860905904814425e-07, 9.718652665657725e-08]
D_delta_phi_cf =[2.438562459335277e-06, 0.001508048263215552, 5.028927388028666e-07, 3.2943292388414826e-07, 8.38844339330183e-08, 1.0317907609453367e-07, 9.050698853130719e-08, 7.552956285268148e-08, 1.1055721314804244e-07]
D_gamma_cf = [1.4012201705981109e-07, 8.907555415884607e-08, 9.776761697071957e-08, 1.1109105795312777e-07, 1.0209322821101747e-07, 1.0824294283325401e-07, 1.0932422149376444e-07, 1.286090590760632e-07, 9.718652665624755e-08]
D_delta_phi_cf = [0.001508048263215552, 2.4385624593358423e-06, 5.028927391725038e-07, 3.294329300712711e-07, 8.388417765333403e-08, 1.0317906836329535e-07, 9.05069937836661e-08, 7.552956531743935e-08, 1.1055721314857877e-07]
# 0.001508048263215552

cf_gamma_cf, cov_gamma_cf = curve_fit(exponential, T, D_gamma_cf, maxfev = 1000000)
cf_delta_phi_cf, cov_delta_phi_cf = curve_fit(exponential, T, D_delta_phi_cf, maxfev = 1000000)

cf_linear_gamma_cf, cov_linear_gamma_cf = curve_fit(linear, T, D_gamma_cf)
cf_linear_delta_phi_cf, cov_linear_delta_phi_cf = curve_fit(linear, T, D_delta_phi_cf)

cf_bessel_gamma_cf, cov_bessel_gamma_cf = curve_fit(bessel, T, D_gamma_cf, maxfev = 1000000)
cf_bessel_delta_phi_cf, cov_bessel_delta_phi_cf = curve_fit(bessel, T, D_delta_phi_cf, maxfev = 1000000)

cf_quadratic_gamma_cf, cov_quadratic_gamma_cf = curve_fit(quadratic, T, D_gamma_cf, maxfev = 1000000)
cf_quadratic_delta_phi_cf, cov_quadratic_delta_phi_cf = curve_fit(quadratic, T, D_delta_phi_cf, maxfev = 1000000)

x = np.linspace(0, 1000, 1000)


fig, ax = plt.subplots(figsize=(8,6))
fig.suptitle('Task 2.8A: Bessel Analysis', **titleFont)
# ax.set_title(f"n = 49",**subtitleFont)

x_values = np.linspace(0, 960, 10000)
ax.plot(T, D_gamma_cf, 'o', label="$D_{\gamma}$, Task 2.5 Values", **pointStyle)
ax.plot(T, D_delta_phi_cf, 'o', label="$D_{Δ\phi}$, Task 2.5 Values", **pointStyleR)
#ax.plot(x, bessel(x, *cf_bessel_gamma_cf), **lineStyle)
#ax.plot(x, bessel(x, *cf_bessel_delta_phi_cf), **lineStyleR)
# ax.plot(x_values, exponential(x_values, *cf_gamma_cf), **lineStyle)
# ax.plot(x_values, exponential(x_values, *cf_delta_phi_cf), **lineStyleR)
# ax.plot(x_values, linear(x_values, *cf_linear_gamma_cf), **lineStyle)
# ax.plot(x_values, linear(x_values, *cf_linear_delta_phi_cf), **lineStyleR)
# ax.plot(x_values, quadratic(x_values, *cf_quadratic_gamma_cf), **lineStyle)
# ax.plot(x_values, quadratic(x_values, *cf_quadratic_delta_phi_cf), **lineStyleR)

ax.ticklabel_format(useMathText=True)
ax.tick_params(axis='both', labelsize=7)
ax.set_yscale("log")
ax.set_xlabel("T / s", **axesFont)
ax.set_ylabel("D / m².s¯¹", **axesFont)
ax.set_xlim(0, 1000)
# ax.set_ylim(5e-8, 1e-7)
fig.legend(loc="upper right", prop=font)
plt.savefig("Plots/Task2.8_Bessel_CF.png", dpi=1000)
plt.show()


# for i in range(len(D_gamma_cf)):    print(T[i],", {:.12f}".format(D_gamma_cf[i]))
# for i in range(len(D_delta_phi_cf)):    print(T[i],", {:.12f}".format(D_delta_phi_cf[i]))