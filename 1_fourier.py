import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate

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
lineStyleBold = {'linewidth': 1}
histStyle =     {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =      {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font = fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

#%% Task 1.2 - Fourier Series of a Square Wave
# Create your dataset that represents the signal of a square wave.
# Plot a square wave of amplitude 50 and period 240.
# The square wave should have 1000 samples.

t = np.linspace(0, 480, 2000)

n = np.linspace(1, 100, 100)
a_n = np.sin(n * np.pi)/(n * np.pi)
b_n = (1 - np.cos(n * np.pi))/(n * np.pi)

fourierTotals = []
fourierSum = 0
# Create a function that takes in a time and returns the value of the square wave at that time.
for i in range(0, len(n)):
    fourierTerm = 2*(a_n[i]*np.cos((1/120)*np.pi*n[i]*t) + b_n[i]*np.sin((1/120)*np.pi*n[i]*t))
    fourierSum += fourierTerm
    fourierTotals.append(fourierSum)

fig, axs = plt.subplots(nrows = 1, ncols = 5, figsize=(24, 4))
fig.suptitle("Fourier Series", **titleFont)
for i in range(5):
    axs[i].tick_params(axis='both', which='major', labelsize=6)
    axs[i].set_title('Square Wave', **titleFont)
    axs[i].set_xlabel('Time / s', **axesFont)
    axs[i].set_ylabel('Amplitude / °C', **axesFont)
    axs[i].plot(t, sp.signal.square(2 * np.pi * (1/240) * t), **lineStyle, label="Simulated Square Wave")
    axs[i].plot(t, fourierTotals[i], **lineStyle, label="Fourier Series Term " + str(i+1))
    axs[i].legend(loc='upper right', prop=font)
    
plt.show()

#%% Task 1.3 - Numerical Integration Test

x1, y1 = np.loadtxt("Task1.3_Semicircle_low.txt", unpack=True,skiprows=1)
x2, y2 = np.loadtxt("Task1.3_Semicircle_high.txt", unpack=True,skiprows=1)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
fig.suptitle("Numerical Integration of Semicircles", **titleFont)
axs[0].tick_params(axis='both', which='major', labelsize=6)
axs[1].tick_params(axis='both', which='major', labelsize=6)

axs[0].set_title("Semicircle, Low Resolution", **titleFont)
axs[0].set_xlabel("x / a.u.", **axesFont)
axs[0].set_ylabel("y / a.u.", **axesFont)
axs[0].plot(x1, y1, 'x-', **pointStyle, **lineStyle, label="Simulated Semicircle")
axs[0].bar(x1, y1, width = 0.1, **barStyle, label="Numerical Integration of Semicircle")
axs[0].legend(loc='upper right', prop=font)

axs[1].set_title("Semicircle, High Resolution", **titleFont)
axs[1].set_xlabel("x / a.u.", **axesFont)
axs[1].set_ylabel("y / a.u.", **axesFont)
axs[1].plot(x2, y2, 'x-', **pointStyle, **lineStyle, label="Simulated Semicircle")
axs[1].bar(x2, y2, width = 0.01, **barStyle, label="Numerical Integration of Semicircle")
axs[1].legend(loc='upper right', prop=font)
plt.show()

integral_1 = 0
integral_2 = 0
for i in range(0, len(x1)):
    integral_1 += y1[i] * 0.1
for i in range(0, len(x2)):
    integral_2 += y2[i] * 0.01
theoretical = np.pi/2

percentage_1 = abs(integral_1 - theoretical)/theoretical * 100
percentage_2 = abs(integral_2 - theoretical)/theoretical * 100
print(integral_1, " with percentage difference of", percentage_1, "%")
print(integral_2, " with percentage difference of", percentage_2, "%")