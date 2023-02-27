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
from uncertainties import unumpy
import sys, difflib
np.set_printoptions(threshold=sys.maxsize)

titleFont =      {'fontname': 'C059', 'size': 14, 'weight': 'bold'}
subtitleFont =   {'fontname': 'C059', 'size': 8, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
annotationFont = {'fontname': 'C059', 'size': 6.1, 'weight': 'bold'}
annotFontWeak =  {'fontname': 'C059', 'size': 6, 'weight': 'normal'}
annotFontMini1 = {'fontname': 'C059', 'size': 5.5, 'weight': 'normal'}
annotFontMini2 = {'fontname': 'C059', 'size': 8, 'weight': 'bold'}
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

def Sinusoidal(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def DoubleSinusoidal(x, a, b, c):  return a * np.sin(b * x + c) + fitting_list[i][3] * np.sin(fitting_list[i][4] * x + fitting_list[i][5]) + fitting_list[i][6]

def substring_after(s, delim):  return s.partition(delim)[2]

def superscripter(string):
    superscript_map = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '¯'}; new_string = ''
    for char in string:
        if char.lower() in superscript_map:
            new_string += superscript_map[char.lower()]
        else:   new_string += char
    return new_string

def formatter(gamma, delta_phi):
    D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
    D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

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

    plot_title = "T = " + dataset_list[i].replace(".txt","") + "; $D_γ$ = " + str(D_gamma_final) + " m².s¯¹, $D_{Δφ}$ = " + str(D_delta_phi_final) + " m².s¯¹"

    # print(D_gamma_final)
    # print(D_delta_phi_final)   
    return(plot_title)

#%% Task 2.6 - Fourier Analysis of a Signal

dataset_list = ["1 MINUTE (A).txt","1 MINUTE (B).txt","2 MINUTES (A).txt", "2 MINUTES (B).txt", "4 MINUTES (A).txt", "4 MINUTES (B).txt", "6 MINUTES.txt", "8 MINUTES.txt", "16 MINUTES.txt"]
dataset_values = [2, 2, 1, 1, 0.5, 0.5, 1/3, 0.25, 0.125]
iter = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
iter = [49]
fitting_list = [[ 1.14315331e+00, -1.04270843e-02,  2.18685847e+00,  2.79251361e-01, 1.88453743e-03, -2.43625158e+00,  5.49363961e+01], [ 3.33244678e-01, -1.04651884e-02,  3.13783192e+00,  6.60374876e+00, 7.22865456e-05,  7.00102393e-01,  4.43863986e+01],  [ 2.45952847e+00, -5.23969260e-03, -4.89878393e+00, -1.44727781e+03, 4.40365418e-06, -1.55799938e+00, -1.39432611e+03], [ 3.20855380e+00, -5.14761643e-03, -5.39367086e+00,  9.26017797e+04, 4.62202842e-06,  4.69683282e+00,  9.26568709e+04], [ 9.36128576e+00, -2.62544700e-03,  4.45078418e-02,  2.63654678e-01, 7.92086040e-04,  1.17249465e+01,  5.12891291e+01], [ 1.02111182e+01, -2.60313065e-03, -2.96335546e-01,  3.97283403e+01, 1.41269834e-04,  1.31578673e+01,  2.12405828e+01], [ 1.76305647e+01, -1.74569029e-03, -6.94304285e+00,  2.60984118e+03, -2.54320115e-06, -1.55054158e+00,  2.66136231e+03], [ 2.81370442e+01, -1.31018225e-03, -7.56743718e+00,  2.98618640e+03, 3.79780011e-06,  1.52969650e+00, -2.93337587e+03], [ 3.89088596e+01, -6.48049847e-04, -1.94113513e+00,  2.32054371e+05, -1.20475654e-06, -1.55568360e+00,  2.32105536e+05]]
p0_list = [[0.1, -0.00864, 0, 10, 0,  0, 54.8], [0.1, -0.00864, 0, 10, 0,  0, 54.8], [32.5, -0.005, -5, 10, 0, 0, 52.5], [2.5, -0.005, -5, 10, 0.001,  2, 62.2], [10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], [10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], [18, -0.0018, -6.5, 10, 0, 0, 50], [32, -0.0013, -7.4, 10, 0, 0, 50], [40, -0.00065, -2, 10, 0, 0, 55], ufloat(1417.621,18.452)]


# subtitle_list = []
for i in range(len(dataset_list)):
    for n in iter:
        list, t_n = [], n
        # for j in range(1,21):
        path = dataset_list[i]
        time, x = np.loadtxt(path, skiprows=3, unpack=True)
        X = fft(x); X[t_n:-t_n] = 0
        x_trunc = np.real(np.fft.ifft(X))
        # print(x_trunc)
        amp = np.abs(X[:t_n + 1]) / len(x) * 2
        phase = np.angle(X[:t_n + 1])

        for m in range(t_n + 1):
            # if n == iter[-1]:  
                # print(f"A_{m} = {amp[m]:.4E}, Δφ_{m} = {phase[m]:.4E}")
            # print(f"A_{n} = {amp[n]:.4E}, Δφ_{n} = {phase[n]:.4E}")
            # print(f"{phase[n]:.12f}")
            list.append([amp[m], phase[m]])
            
        r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057/2, 0.00001)
        gamma = np.abs(ufloat(list[-2][0], 0))/(200/np.pi)
        delta_phi = ufloat(list[-2][1], 0)*10
        plot_title = formatter(gamma, delta_phi)
                
        if t_n == 2:    t_n = 1
        try:
            D_gamma = (np.pi*dataset_values[i])*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
            D_gamma = str("{:.4E}".format(D_gamma))
            D_gamma = D_gamma.replace("+/-", " ± ")
            D_gamma = D_gamma.replace("E", " E")
            D_delta_phi = (np.pi*dataset_values[i])*(r_inner - r_outer)**2/(2*delta_phi**2)
            D_delta_phi = str("{:.4E}".format(D_delta_phi))
            D_delta_phi = D_delta_phi.replace("+/-", " ± ")
            D_delta_phi = D_delta_phi.replace("E", " E")
            subtitle = "$Dᵧ$ = " + D_gamma + ", $D_Δᵩ$ = " + D_delta_phi
        except ZeroDivisionError:   print("FAILED")
            
        # print(f"Dᵧ({t_n}) = {D_gamma}, D_Δᵩ({t_n}) = {D_delta_phi}")
            
        t = np.linspace(time[1], time[-1], len(x))
        shift_down = fitting_list[i][3] * np.sin(fitting_list[i][4] * t + fitting_list[i][5]) + fitting_list[i][6]
        plt.plot(t, x - shift_down, label='Original Data', **pointStyle)
        plt.plot(t, x_trunc - shift_down, label='Truncated Fourier Series', **lineStyleR)
        cf_trunc, cov_trunc = curve_fit(Sinusoidal, t, x_trunc - shift_down, p0=[fitting_list[i][0], fitting_list[i][1], fitting_list[i][2], 0])      
        plt.plot(t, Sinusoidal(t, *cf_trunc), label='Fitted Truncated Fourier Series', **lineStyleBoldG)
        print(cf_trunc)
        
        plt.suptitle("Task 2.6A: Truncating a Fourier Series, n = {:.0f}".format(t_n), **titleFont)
        plt.title(plot_title, **subtitleFont)
        plt.xlabel("Time / ds", **axesFont)
        plt.ylabel("Temperature / °C", **axesFont)
        plt.xticks(**ticksFont)
        plt.yticks(**ticksFont)

        plt.legend(loc="upper right", prop=font)
        #plt.savefig("Plots/Task2.6A_truncation_" + path.replace(".txt","") + "_n=" + str(t_n) + ".png", dpi=500)
        #plt.clf()
        plt.show()
        plt.clf()