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
from uncertainties import ufloat_fromstr
from uncertainties import unumpy

titleFont =      {'fontname': 'C059', 'size': 14, 'weight': 'bold'}
subtitleFont =   {'fontname': 'C059', 'size': 8, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 7, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
ticksFont =      {'fontname': 'SF Mono', 'size': 7}
errorStyle =     {'mew': 1, 'ms': 1, 'capsize': 3, 'color': 'white', 'zorder': 13}
pointStyle =     {'mew': 1, 'ms': 3, 'zorder': 100}
lineStyle =      {'linewidth': 0.8, 'zorder': 100}
lineStyleR =     {'linewidth': 0.8, 'color': 'red'}
lineStyleP =     {'linewidth': 0.8, 'color': 'purple'}
lineStyleG =     {'linewidth': 0.8, 'color': 'green'}
lineStyleBoldR = {'linewidth': 2, 'color': 'red'}
lineStyleBoldP = {'linewidth': 2, 'color': 'purple'}
lineStyleBoldG = {'linewidth': 2, 'color': 'green'}
histStyle =      {'facecolor': 'green', 'alpha': 0.5, 'edgecolor': 'black'}
barStyle =       {'color': 'green', 'edgecolor': 'black', 'linewidth': 0.25} 
font =           fnt.FontProperties(family='C059', weight='normal', style='italic', size=8)

D_vals = np.genfromtxt("2.7_fourier_output_D.txt", dtype="str")

def linear(x, a, b):            return a*x + b
def exponential(x, a, b, c):    return a*np.exp(b*x) + c

harmonics_1a_gamma, harmonics_1a_phi = [], []
harmonics_1a_gamma_unc, harmonics_1a_phi_unc = [], []
for i in range(0,25):
    harmonics_1a_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_1a_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_1a_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_1a_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)

harmonics_1b_gamma, harmonics_1b_phi = [], []
harmonics_1b_gamma_unc, harmonics_1b_phi_unc = [], []
for i in range(25,50):
    harmonics_1b_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_1b_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_1b_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_1b_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)

harmonics_2a_gamma, harmonics_2a_phi = [], []
harmonics_2a_gamma_unc, harmonics_2a_phi_unc = [], []
for i in range(50,75):
    harmonics_2a_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_2a_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_2a_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_2a_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)
    
harmonics_2b_gamma, harmonics_2b_phi = [], []
harmonics_2b_gamma_unc, harmonics_2b_phi_unc = [], []
for i in range(75,100):
    harmonics_2b_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_2b_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_2b_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_2b_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)
    
harmonics_4a_gamma, harmonics_4a_phi = [], []
harmonics_4a_gamma_unc, harmonics_4a_phi_unc = [], []
for i in range(100,125):
    harmonics_4a_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_4a_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_4a_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_4a_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)
    
harmonics_4b_gamma, harmonics_4b_phi = [], []
harmonics_4b_gamma_unc, harmonics_4b_phi_unc = [], []
for i in range(125,150):
    harmonics_4b_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_4b_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_4b_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_4b_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)
    
harmonics_6_gamma, harmonics_6_phi = [], []
harmonics_6_gamma_unc, harmonics_6_phi_unc = [], []
for i in range(150,175):
    harmonics_6_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_6_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_6_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_6_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)
    
harmonics_8_gamma, harmonics_8_phi = [], []
harmonics_8_gamma_unc, harmonics_8_phi_unc = [], []
for i in range(175,200):
    harmonics_8_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_8_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_8_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_8_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)

harmonics_16_gamma, harmonics_16_phi = [], []
harmonics_16_gamma_unc, harmonics_16_phi_unc = [], []
for i in range(200,225):
    harmonics_16_gamma.append(ufloat_fromstr(D_vals[i][0]).n)
    harmonics_16_phi.append(ufloat_fromstr(D_vals[i][1]).n)
    harmonics_16_gamma_unc.append(ufloat_fromstr(D_vals[i][0]).s)
    harmonics_16_phi_unc.append(ufloat_fromstr(D_vals[i][1]).s)

x_values = np.linspace(1,49,25)
line_values = np.linspace(1,49,1000)

# for j in range(25):
#     print([np.linspace(1,49,25)[j],harmonics_1a_gamma[j]])

linear_1a_gamma = curve_fit(linear, x_values, harmonics_1a_gamma, sigma=harmonics_1a_gamma_unc)
linear_1a_phi = curve_fit(linear, x_values, harmonics_1a_phi, sigma=harmonics_1a_phi_unc)
linear_1b_gamma = curve_fit(linear, x_values, harmonics_1b_gamma, sigma=harmonics_1b_gamma_unc)
linear_1b_phi = curve_fit(linear, x_values, harmonics_1b_phi, sigma=harmonics_1b_phi_unc)
linear_2a_gamma = curve_fit(linear, x_values, harmonics_2a_gamma, sigma=harmonics_2a_gamma_unc)
linear_2a_phi = curve_fit(linear, x_values, harmonics_2a_phi, sigma=harmonics_2a_phi_unc)
linear_2b_gamma = curve_fit(linear, x_values, harmonics_2b_gamma, sigma=harmonics_2b_gamma_unc)
linear_2b_phi = curve_fit(linear, x_values, harmonics_2b_phi, sigma=harmonics_2b_phi_unc)
linear_4a_gamma = curve_fit(linear, x_values, harmonics_4a_gamma, sigma=harmonics_4a_gamma_unc)
linear_4a_phi = curve_fit(linear, x_values, harmonics_4a_phi, sigma=harmonics_4a_phi_unc)
linear_4b_gamma = curve_fit(linear, x_values, harmonics_4b_gamma, sigma=harmonics_4b_gamma_unc)
linear_4b_phi = curve_fit(linear, x_values, harmonics_4b_phi, sigma=harmonics_4b_phi_unc)
linear_6_gamma = curve_fit(linear, x_values, harmonics_6_gamma, sigma=harmonics_6_gamma_unc)
linear_6_phi = curve_fit(linear, x_values, harmonics_6_phi, sigma=harmonics_6_phi_unc)
linear_8_gamma = curve_fit(linear, x_values, harmonics_8_gamma, sigma=harmonics_8_gamma_unc)
linear_8_phi = curve_fit(linear, x_values, harmonics_8_phi, sigma=harmonics_8_phi_unc)
linear_16_gamma = curve_fit(linear, x_values, harmonics_16_gamma, sigma=harmonics_16_gamma_unc)
linear_16_phi = curve_fit(linear, x_values, harmonics_16_phi, sigma=harmonics_16_phi_unc)

exponential_1a_gamma = curve_fit(exponential, x_values, harmonics_1a_gamma, sigma=harmonics_1a_gamma_unc)
exponential_1a_phi = curve_fit(exponential, x_values, harmonics_1a_phi, sigma=harmonics_1a_phi_unc)
exponential_1b_gamma = curve_fit(exponential, x_values, harmonics_1b_gamma, sigma=harmonics_1b_gamma_unc)
exponential_1b_phi = curve_fit(exponential, x_values, harmonics_1b_phi, sigma=harmonics_1b_phi_unc)
exponential_2a_gamma = curve_fit(exponential, x_values, harmonics_2a_gamma, sigma=harmonics_2a_gamma_unc)
exponential_2a_phi = curve_fit(exponential, x_values, harmonics_2a_phi, sigma=harmonics_2a_phi_unc)
exponential_2b_gamma = curve_fit(exponential, x_values, harmonics_2b_gamma, sigma=harmonics_2b_gamma_unc)
exponential_2b_phi = curve_fit(exponential, x_values, harmonics_2b_phi, sigma=harmonics_2b_phi_unc)
exponential_4a_gamma = curve_fit(exponential, x_values, harmonics_4a_gamma, sigma=harmonics_4a_gamma_unc)
exponential_4a_phi = curve_fit(exponential, x_values, harmonics_4a_phi, sigma=harmonics_4a_phi_unc)
exponential_4b_gamma = curve_fit(exponential, x_values, harmonics_4b_gamma, sigma=harmonics_4b_gamma_unc)
exponential_4b_phi = curve_fit(exponential, x_values, harmonics_4b_phi, sigma=harmonics_4b_phi_unc)
exponential_6_gamma = curve_fit(exponential, x_values, harmonics_6_gamma, sigma=harmonics_6_gamma_unc)
exponential_6_phi = curve_fit(exponential, x_values, harmonics_6_phi, sigma=harmonics_6_phi_unc)
exponential_8_gamma = curve_fit(exponential, x_values, harmonics_8_gamma, sigma=harmonics_8_gamma_unc)
exponential_8_phi = curve_fit(exponential, x_values, harmonics_8_phi, sigma=harmonics_8_phi_unc)
exponential_16_gamma = curve_fit(exponential, x_values, harmonics_16_gamma, sigma=harmonics_16_gamma_unc)
exponential_16_phi = curve_fit(exponential, x_values, harmonics_16_phi, sigma=harmonics_16_phi_unc)



order = 10
def tenth(x, coeff):
    return coeff[0]*x**10 + coeff[1]*x**9 + coeff[2]*x**8 + coeff[3]*x**7 + coeff[4]*x**6 + coeff[5]*x**5 + coeff[6]*x**4 + coeff[7]*x**3 + coeff[8]*x**2 + coeff[9]*x**1 + coeff[10]

# coeff[0]*x**9 + coeff[1]*x**8 + coeff[2]*x**7 + coeff[3]*x**6 + coeff[4]*x**5 + coeff[5]*x**4 + coeff[6]*x**3 + coeff[7]*x**2 + coeff[8]*x**1 + coeff[9]
# coeff[0]*x**8 + coeff[1]*x**7 + coeff[2]*x**6 + coeff[3]*x**5 + coeff[4]*x**4 + coeff[5]*x**3 + coeff[6]*x**2 + coeff[7]*x**1 + coeff[8]
# coeff[0]*x**7 + coeff[1]*x**6 + coeff[2]*x**5 + coeff[3]*x**4 + coeff[4]*x**3 + coeff[5]*x**2 + coeff[6]*x**1 + coeff[7]
# coeff[0]*x**6 + coeff[1]*x**5 + coeff[2]*x**4 + coeff[3]*x**3 + coeff[4]*x**2 + coeff[5]*x**1 + coeff[6]
# coeff[0]*x**5 + coeff[1]*x**4 + coeff[2]*x**3 + coeff[3]*x**2 + coeff[4]*x**1 + coeff[5]
# coeff[0]*x**4 + coeff[1]*x**3 + coeff[2]*x**2 + coeff[3]*x**1 + coeff[4]
# coeff[0]*x**3 + coeff[1]*x**2 + coeff[2]*x**1 + coeff[3]
# coeff[0]*x**2 + coeff[1]*x**1 + coeff[2] 

polyPlotter_1a_gamma = np.polyfit(x_values, harmonics_1a_gamma, order)
polyPlotter_1a_phi = np.polyfit(x_values, harmonics_1a_phi, order)
polyPlotter_1b_gamma = np.polyfit(x_values, harmonics_1b_gamma, order)
polyPlotter_1b_phi = np.polyfit(x_values, harmonics_1b_phi, order)
polyPlotter_2a_gamma = np.polyfit(x_values, harmonics_2a_gamma, order)
polyPlotter_2a_phi = np.polyfit(x_values, harmonics_2a_phi, order)
polyPlotter_2b_gamma = np.polyfit(x_values, harmonics_2b_gamma, order)
polyPlotter_2b_phi = np.polyfit(x_values, harmonics_2b_phi, order)
polyPlotter_4a_gamma = np.polyfit(x_values, harmonics_4a_gamma, order)
polyPlotter_4a_phi = np.polyfit(x_values, harmonics_4a_phi, order)
polyPlotter_4b_gamma = np.polyfit(x_values, harmonics_4b_gamma, order)
polyPlotter_4b_phi = np.polyfit(x_values, harmonics_4b_phi, order)
polyPlotter_6_gamma = np.polyfit(x_values, harmonics_6_gamma, order)
polyPlotter_6_phi = np.polyfit(x_values, harmonics_6_phi, order)
polyPlotter_8_gamma = np.polyfit(x_values, harmonics_8_gamma, order)
polyPlotter_8_phi = np.polyfit(x_values, harmonics_8_phi, order)
polyPlotter_16_gamma = np.polyfit(x_values, harmonics_16_gamma, order)
polyPlotter_16_phi = np.polyfit(x_values, harmonics_16_phi, order)

def polyPlotter(x, polycoeffs):
    y = np.zeros(1000)
    for i in range(len(polycoeffs)):
        y += polycoeffs[i]*x**i
    return y


fig, ax = plt.subplots(figsize=(10,6))

ax.plot(x_values, harmonics_1a_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_1a_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_1b_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_1b_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_2a_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_2a_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_2b_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_2b_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_4a_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_4a_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_4b_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_4b_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_6_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_6_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_8_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_8_phi, 'x', **pointStyle)
ax.plot(x_values, harmonics_16_gamma, 'x', **pointStyle)
ax.plot(x_values, harmonics_16_phi, 'x', **pointStyle)

ax.plot(line_values, linear(line_values, *linear_1a_gamma[0]), label='Dᵧ: 1 minute, A', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_1a_phi[0]), label='Dᵩ: 1 minute, A', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_1b_gamma[0]), label='Dᵧ: 1 minute, B', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_1b_phi[0]), label='Dᵩ: 1 minute, B', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_2a_gamma[0]), label='Dᵧ: 2 minutes, A', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_2a_phi[0]), label='Dᵩ: 2 minutes, A', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_2b_gamma[0]), label='Dᵧ: 2 minutes, B', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_2b_phi[0]), label='Dᵩ: 2 minutes, B', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_4a_gamma[0]), label='Dᵧ: 4 minutes, A', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_4a_phi[0]), label='Dᵩ: 4 minutes, A', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_4b_gamma[0]), label='Dᵧ: 4 minutes, B', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_4b_phi[0]), label='Dᵩ: 4 minutes, B', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_6_gamma[0]), label='Dᵧ: 6 minutes', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_6_phi[0]), label='Dᵩ: 6 minutes', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_8_gamma[0]), label='Dᵧ: 8 minutes', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_8_phi[0]), label='Dᵩ: 8 minutes', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_16_gamma[0]), label='Dᵧ: 16 minutes', **lineStyle)
ax.plot(line_values, linear(line_values, *linear_16_phi[0]), label='Dᵩ: 16 minutes', **lineStyle)

# ax.plot(line_values, exponential(line_values, *exponential_1a_gamma[0]), label='Dᵧ: 1 minute, A', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_1a_phi[0]), label='Dᵩ: 1 minute, A', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_1b_gamma[0]), label='Dᵧ: 1 minute, B', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_1b_phi[0]), label='Dᵩ: 1 minute, B', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_2a_gamma[0]), label='Dᵧ: 2 minutes, A', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_2a_phi[0]), label='Dᵩ: 2 minutes, A', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_2b_gamma[0]), label='Dᵧ: 2 minutes, B', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_2b_phi[0]), label='Dᵩ: 2 minutes, B', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_4a_gamma[0]), label='Dᵧ: 4 minutes, A', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_4a_phi[0]), label='Dᵩ: 4 minutes, A', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_4b_gamma[0]), label='Dᵧ: 4 minutes, B', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_4b_phi[0]), label='Dᵩ: 4 minutes, B', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_6_gamma[0]), label='Dᵧ: 6 minutes', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_6_phi[0]), label='Dᵩ: 6 minutes', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_8_gamma[0]), label='Dᵧ: 8 minutes', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_8_phi[0]), label='Dᵩ: 8 minutes', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_16_gamma[0]), label='Dᵧ: 16 minutes', **lineStyle)
# ax.plot(line_values, exponential(line_values, *exponential_16_phi[0]), label='Dᵩ: 16 minutes', **lineStyle)

# ax.plot(line_values, polyPlotter(line_values, polyPlotter_1a_gamma), label='Dᵧ: 1 minute, A', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_1a_phi), label='Dᵩ: 1 minute, A', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_1b_gamma), label='Dᵧ: 1 minute, B', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_1b_phi), label='Dᵩ: 1 minute, B', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_2a_gamma), label='Dᵧ: 2 minutes, A', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_2a_phi), label='Dᵩ: 2 minutes, A', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_2b_gamma), label='Dᵧ: 2 minutes, B', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_2b_phi), label='Dᵩ: 2 minutes, B', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_4a_gamma), label='Dᵧ: 4 minutes, A', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_4a_phi), label='Dᵩ: 4 minutes, A', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_4b_gamma), label='Dᵧ: 4 minutes, B', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_4b_phi), label='Dᵩ: 4 minutes, B', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_6_gamma), label='Dᵧ: 6 minutes', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_6_phi), label='Dᵩ: 6 minutes', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_8_gamma), label='Dᵧ: 8 minutes', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_8_phi), label='Dᵩ: 8 minutes', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_16_gamma), label='Dᵧ: 16 minutes', **lineStyle)
# ax.plot(line_values, polyPlotter(line_values, polyPlotter_16_phi), label='Dᵩ: 16 minutes', **lineStyle)

# ax.plot(line_values, tenth(line_values, polyPlotter_1a_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_1a_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_1b_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_1b_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_2a_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_2a_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_2b_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_2b_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_4a_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_4a_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_4b_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_4b_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_6_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_6_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_8_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_8_phi), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_16_gamma), **lineStyle)
# ax.plot(line_values, tenth(line_values, polyPlotter_16_phi), **lineStyle)

ax.errorbar(x_values, harmonics_1a_gamma, yerr=harmonics_1a_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_1a_phi, yerr=harmonics_1a_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_1b_gamma, yerr=harmonics_1b_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_1b_phi, yerr=harmonics_1b_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_2a_gamma, yerr=harmonics_2a_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_2a_phi, yerr=harmonics_2a_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_2b_gamma, yerr=harmonics_2b_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_2b_phi, yerr=harmonics_2b_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_4a_gamma, yerr=harmonics_4a_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_4a_phi, yerr=harmonics_4a_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_4b_gamma, yerr=harmonics_4b_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_4b_phi, yerr=harmonics_4b_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_6_gamma, yerr=harmonics_6_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_6_phi, yerr=harmonics_6_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_8_gamma, yerr=harmonics_8_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_8_phi, yerr=harmonics_8_phi_unc, **errorStyle)
ax.errorbar(x_values, harmonics_16_gamma, yerr=harmonics_16_gamma_unc, **errorStyle)
ax.errorbar(x_values, harmonics_16_phi, yerr=harmonics_16_phi_unc, **errorStyle)

ax.set_xlabel("$n^{th}$ Harmonic", **axesFont)
ax.set_ylabel("D / m².s¯¹", **axesFont)
ax.ticklabel_format(useMathText=True)
ax.tick_params(axis='both', labelsize=7)
# ax.set_yscale("log")
ax.set_title(f"Complete Values up to n = 49, with Errorbars",**subtitleFont)
fig.suptitle('Task 2.7B: D Values for Various Periods', **titleFont)
fig.legend(loc="lower right", prop=font, bbox_to_anchor=(1, 0.13))
ax.set_ylim(1e-8, 0.04e-4)
plt.savefig(f"Plots/Task2.7B_DValues_Linear_LoglessScale_NoConnection_0.04.png", dpi=300)
plt.show()