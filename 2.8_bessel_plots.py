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

#%% n = 49

T = [60, 60, 120, 120, 240, 240, 360, 480, 960]
D_gamma_cf = [1.5506609070308076e-07, 9.943464464268348e-08, 1.0976512933994218e-07, 1.255971360052302e-07, 1.206746049626923e-07, 1.286380841594988e-07, 1.3596768121084294e-07, 1.7268644535538645e-07, 1.5322550352182592e-07]
D_delta_phi_cf = [1.3681216024102855e-05, 0.007540244198096596, 3.6444892448750743e-06, 2.8354801941960967e-06, 9.958734671759949e-07, 8.941154721024194e-07, 5.286281606320546e-07, 3.312284623952762e-07, 1.399963568278951e-07]

fig, ax = plt.subplots(figsize=(8,6))
fig.legend(loc="lower right", prop=font)
fig.suptitle('Task 2.8A: Bessel Analysis', **titleFont)
ax.set_title(f"n = 49",**subtitleFont)

ax.plot(T, D_gamma_cf, 'o', label="D_gamma, Task 2.4", **pointStyle)
ax.plot(T, D_delta_phi_cf, 'o', label="D_delta_phi, Task 2.4", **pointStyleR)

ax.ticklabel_format(useMathText=True)
ax.tick_params(axis='both', labelsize=7)
ax.set_yscale("log")
ax.set_xlabel("T / s", **axesFont)
ax.set_ylabel("D / m².s¯¹", **axesFont)
plt.show()


