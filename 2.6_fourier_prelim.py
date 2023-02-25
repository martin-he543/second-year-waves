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
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontSmall = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontBold =  {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
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

dataset_list = ["1 MINUTE (A).txt","1 MINUTE (B).txt","2 MINUTES (A).txt", "2 MINUTES (B).txt", "4 MINUTES (A).txt", "4 MINUTES (B).txt", "6 MINUTES.txt", "8 MINUTES.txt", "16 MINUTES.txt"]
path = dataset_list[0]
x, y = np.loadtxt(path, skiprows=3, unpack=True); t_n = 3

#%% METHOD 3: TRUNCATING A FOURIER SERIES BY NUMERICAL INTEGRATION


#%% METHOD 1: TRUNCATING A FOURIER SERIES BY FFT
for i in range(len(dataset_list)):
    path = dataset_list[i]
    n = 3
    time, x = np.loadtxt(path, skiprows=3, unpack=True); t_n = n
    X = fft(x); X[t_n:-t_n] = 0
    x_trunc = np.real(np.fft.ifft(X))
    amp = np.abs(X[:t_n + 1]) / len(x) * 2
    phase = np.angle(X[:t_n + 1])

    list = []
    for n in range(1,t_n + 1):
        print(f"A_{n} = {amp[n]:.4E}, Δφ_{n} = {phase[n]:.4E}")
        list.append([amp[n], phase[n]])

    gamma = ufloat(list[1][0], 0)
    delta_phi = ufloat(list[1][1], 0)
    r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
    D_gamma = (np.pi/4)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
    D_delta_phi = (np.pi/4)*(r_inner - r_outer)**2/(2*delta_phi**2)
    
    print(f"D_gamma = {D_gamma:.12E}\nD_delta_phi = {D_delta_phi:.12E}")

    t = np.linspace(time[1], time[-1], len(x))
    plt.plot(t, x, label='Original Data', **pointStyle)
    plt.plot(t, x_trunc, label='Truncated Fourier Series', **lineStyleR)

    plt.suptitle("Task 2.6A: [Method 1] Truncating a Fourier Series, n = {:.0f}".format(t_n), **titleFont)
    plt.title(path, **subtitleFont)
    plt.xlabel("Time / ds", **axesFont)
    plt.ylabel("Temperature / °C", **axesFont)
    plt.xticks(**ticksFont)
    plt.yticks(**ticksFont)

    plt.legend(loc="best", prop=font)
    plt.show()
    
""" #%% METHOD 2: TRUNCATING A FOURIER SERIES BY CURVE FIT
import sympy as sym
from sympy import pi, plot, fourier_series
from sympy.abc import x

dataset_list = ["1 MINUTE (A)_coefficients.txt","1 MINUTE (B)_coefficients.txt","2 MINUTES (A)_coefficients.txt", "2 MINUTES (B)_coefficients.txt", "4 MINUTES (A)_coefficients.txt", "4 MINUTES (B)_coefficients.txt", "6 MINUTES_coefficients.txt", "8 MINUTES_coefficients.txt", "16 MINUTES_coefficients.txt"]

for m in range(len(dataset_list)):
    coeff_list = np.genfromtxt("Output/" + dataset_list[m], dtype="str")
    cfds, cods  = [], []
    for k in range(len(coeff_list)):
        cfds.append(ufloat_fromstr(coeff_list[k]).n)
        cods.append(ufloat_fromstr(coeff_list[k]).s)
        
    a, b, c, d, e, f, g = cfds[0], cfds[1], cfds[2], cfds[3], cfds[4], cfds[5], cfds[6]
    f = (a * sym.sin(b * x + c)) + d * sym.sin(e * x + f) + g
    s = fourier_series(f, (x, -5000, 5000))
    s1 = s.truncate(n = 3)
    s2 = s.truncate(n = 5)
    s3 = s.truncate(n = 7)

    p = plot(f, s1, s2, s3, show=False, legend=True)
    p[0].line_color = (0, 0, 0)
    p[0].label = 'original'
    p[1].line_color = (0.7, 0.7, 0.7)
    p[1].label = 'n=3'
    p[2].line_color = (0.5, 0.5, 0.5)
    p[2].label = 'n=5'
    p[3].line_color = (0.3, 0.3, 0.3)
    p[3].label = 'n=7'
    p.show()

 """

