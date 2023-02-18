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

def sinusodial(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def linear(x, a, b):            return a * x + b
def cubic(x, a, b, c, d):       return a * x**3 + b * x**2 + c * x + d

tau = 1200      # Half-period in deciseconds
num_tau = int(np.floor(x1[-1]/tau))
x1_sliced = []

for i in range(num_tau + 1): # Data in slices of tau for each period
    if i == num_tau:
        x1_sliced.append([i * tau, int(x2[-1]) + 1])
    else:
        x1_sliced.append([i * tau, (i + 1) * tau])

fourA_sliced = []
for j in range(num_tau + 1):
    fourA_sliced.append([x2[x1_sliced[j][0]: x1_sliced[j][1]], y2[x1_sliced[j][0]: x1_sliced[j][1]]])

linear_fitted = []
for k in range(num_tau + 1):
    linear_fitted.append(curve_fit(linear, fourA_sliced[k][0], fourA_sliced[k][1]))
    linear_params = [linear_fitted[k][0][0], linear_fitted[k][0][1]]
    print("Linear fit parameters: ", linear_params)
    plt.plot(fourA_sliced[k][0], linear(fourA_sliced[k][0], *linear_params), label="Linear fit", **lineStyleBold)

# plt.plot(fourA_sliced[0][0], linear(fourA_sliced[0][0], *lin1), label="Linear fit", **lineStyleBold)
plt.plot(x2,y2, label="Data", **pointStyle)
plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
plt.title("T = " + str(tau) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.show()