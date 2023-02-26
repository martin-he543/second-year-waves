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

#%% Task 2.6 - Fourier Analysis of a Signal

dataset_list = ["1 MINUTE (A).txt","1 MINUTE (B).txt","2 MINUTES (A).txt", "2 MINUTES (B).txt", "4 MINUTES (A).txt", "4 MINUTES (B).txt", "6 MINUTES.txt", "8 MINUTES.txt", "16 MINUTES.txt"]
dataset_values = [2, 2, 1, 1, 0.5, 0.5, 1/3, 0.25, 0.125]
iter = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
ranges = [[1, 50],[51, 100], [101, 150], [151, 200], [201, 250], [251, 300],[301, 350], [351, 400], [401, 450], [451, 500], [501, 550], [551, 600]]
phase_data = np.loadtxt("2.6_harmonic_phase_data.txt")

# subtitle_list = []
for i in range(len(dataset_list)):
    phase_data_range = phase_data[ranges[i][0]:ranges[i][1]]
    for n in iter:
        list, t_n = [], n
        # for j in range(1,21):
        path = dataset_list[i]
        time, x = np.loadtxt(path, skiprows=3, unpack=True)
        X = fft(x); X[t_n:-t_n] = 0
        x_trunc = np.real(np.fft.ifft(X))
        amp = np.abs(X[:t_n + 1]) / len(x) * 2
        phase = np.angle(X[:t_n + 1])

        for m in range(t_n + 1):
            if n == iter[-1]:  
                print(f"A_{m} = {amp[m]:.4E}, Δφ_{m} = {phase[m]:.4E}")
            # print(f"A_{n} = {amp[n]:.4E}, Δφ_{n} = {phase[n]:.4E}")
            # print(f"{phase[n]:.12f}")
            list.append([amp[m], phase[m]])

        gamma = np.abs(ufloat(list[-2][0], 0))/(200/np.pi)
        delta_phi = ufloat(list[-2][1], 0)*10
        
        r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
        
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
            
        print(f"Dᵧ({t_n}) = {D_gamma}, D_Δᵩ({t_n}) = {D_delta_phi}")
            
        t = np.linspace(time[1], time[-1], len(x))
        plt.plot(t, x, label='Original Data', **pointStyle)
        plt.plot(t, x_trunc, label='Truncated Fourier Series', **lineStyleR)

        # subtitle_text = ""
        # for n in range(t_n):
        #     subtitle_text += f"A_{n} = {amp[n]:.4f}, Δφ_{n} = {phase[n]:.4f} \n"
        plt.suptitle("Task 2.6A: Truncating a Fou rier Series, n = {:.0f}".format(t_n), **titleFont)
        plt.title("Source: " + path + "; " + subtitle, **subtitleFont)
        plt.xlabel("Time / ds", **axesFont)
        plt.ylabel("Temperature / °C", **axesFont)
        plt.xticks(**ticksFont)
        plt.yticks(**ticksFont)

        plt.legend(loc="best", prop=font)
        # plt.savefig("Plots/Task2.6A_truncation_" + path.replace(".txt","") + "_n=" + str(t_n) + ".png", dpi=500)
        #plt.clf()
        #plt.show()
        plt.clf()