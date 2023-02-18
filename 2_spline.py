import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit

titleFont =     {'fontname': 'Bodoni 72', 'size': 12}
subtitleFont =  {'fontname': 'Bodoni 72', 'size': 9, 'style':'italic'}
axesFont =      {'fontname': 'Bodoni 72', 'size': 9, 'style':'italic'}
annotationFont ={'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak = {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1= {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2= {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =     {'fontname': 'SF Mono', 'size': 7}
errorStyle =    {'mew': 1, 'ms': 3, 'capsize': 3, 'color': 'blue', 'ls': ''}
pointStyle =    {'mew': 1, 'ms': 3, 'color': 'blue'}
lineStyle =     {'linewidth': 0.5}
lineStyleBold = {'linewidth': 1, 'color': 'red'}
lineStyleBold1= {'linewidth': 1, 'color': 'green'}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =      {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font = fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

x1, y1 = np.loadtxt("thermal_4min_a.txt", unpack=True,skiprows=3)
x2, y2 = np.loadtxt("thermal_4min_b.txt", unpack=True,skiprows=3)

plt