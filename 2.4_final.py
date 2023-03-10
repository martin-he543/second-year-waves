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

def Sinusoidal(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def DoubleSinusoidal(x, a, b, c, d, e, f, g):  return a * np.sin(b * x + c) + d * np.sin(e * x + f) + g
def DoubleSinusoidalFlipped(x, a, b, c, d, e, f, g):  return -(a * np.sin(b * x + c)) + d * np.sin(e * x + f) + g
def Square(x, a, b, c, d):      return a * np.sign(np.sin(b * x + c)) + d
def substring_after(s, delim):  return s.partition(delim)[2]

def superscripter(string):
    superscript_map = {'0': '???', '1': '??', '2': '??', '3': '??', '4': '???', '5': '???', '6': '???', '7': '???', '8': '???', '9': '???', '-': '??'}; new_string = ''
    for char in string:
        if char.lower() in superscript_map:
            new_string += superscript_map[char.lower()]
        else:   new_string += char
    return new_string

D_gamma1, D_delta_phi1 = [], []

def formatter(gamma, delta_phi):
    D_gamma = angular_freq_list[j]*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
    D_delta_phi = angular_freq_list[j]*(r_inner - r_outer)**2/(2*delta_phi**2)
    D_gamma1.append(D_gamma)
    D_delta_phi1.append(D_delta_phi)
    
    D_gamma = "{:.4E}".format(D_gamma)
    D_delta_phi = "{:.4E}".format(D_delta_phi)
    D_gamma = D_gamma.replace("E"," ??10")
    D_delta_phi = D_delta_phi.replace("E"," ??10")
    D_gamma = D_gamma.replace("+/-"," ?? ")
    D_delta_phi = D_delta_phi.replace("+/-"," ?? ")

    exp_gamma = substring_after(D_gamma, " ??10")
    exp_delta_phi = substring_after(D_delta_phi, " ??10")

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

    plot_title = "T = " + file.replace(".txt","") + "; ?? = {:.4f} ?? {:.4f}, ???? = {:.4f} ?? {:.4f} rad, \n$D_??$ = ".format(gamma.n, gamma.s, delta_phi.n, delta_phi.s) + str(D_gamma_final) + " m??.s????, $D_??$ = " + str(D_delta_phi_final) + " m??.s????"

    print(D_gamma_final)
    print(D_delta_phi_final)   
    return(plot_title)

def formatter2(gamma, delta_phi):
    gamma_disp = "{:.12E}".format(gamma)
    delta_phi_disp = "{:.12E}".format(delta_phi)
    gamma_disp = gamma_disp.replace("E"," ??10")
    delta_phi_disp = delta_phi_disp.replace("E"," ??10")
    gamma_disp = gamma_disp.replace("+/-"," ?? ")
    delta_phi_disp = delta_phi_disp.replace("+/-"," ?? ")
    
    exp_gamma = substring_after(gamma, " ??10")
    exp_delta_phi = substring_after(delta_phi, " ??10")
    diff_gamma = ''.join([x[-1] for x in difflib.ndiff(exp_gamma, gamma_disp) if x[0] != ' '])
    diff_delta_phi = ''.join([x[-1] for x in difflib.ndiff(exp_delta_phi, delta_phi_disp) if x[0] != ' '])
    
    exp_gamma = int(exp_gamma)
    exp_delta_phi = int(exp_delta_phi)
    exp_gamma = str(exp_gamma)
    exp_delta_phi = str(exp_delta_phi)
    exp_gamma = superscripter(exp_gamma)
    exp_delta_phi = superscripter(exp_delta_phi)
    
    gamma_final = diff_gamma + exp_gamma
    delta_phi_final = diff_delta_phi + exp_delta_phi
    print(gamma_final, delta_phi_final)
      
dataset_list = ["1 MINUTE (A).txt","1 MINUTE (B).txt","2 MINUTES (A).txt", "2 MINUTES (B).txt", "4 MINUTES (A).txt", "4 MINUTES (B).txt", "6 MINUTES.txt", "8 MINUTES.txt", "16 MINUTES.txt"]
tau_list = [600, 600, 1200, 1200, 2400, 2400, 3600, 3800, 9600]
phi_1_list = [ufloat(209.497,1.171), ufloat(299.835,0.36), ufloat(264.214,0.079), ufloat(172.801,16.777), ufloat(1213.546,1.946), ufloat(1093.014,0.3), ufloat(1421.635,4.923), ufloat(1417.621,18.452), ufloat(1852.415,152.329)]
p0_list = [[0.1, -0.00864, 0, 10, 0,  0, 54.8], [0.1, -0.00864, 0, 10, 0,  0, 54.8], [32.5, -0.005, -5, 10, 0, 0, 52.5], [2.5, -0.005, -5, 10, 0.001,  2, 62.2], [10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], [10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], [18, -0.0018, -6.5, 10, 0, 0, 50], [32, -0.0013, -7.4, 10, 0, 0, 50], [40, -0.00065, -2, 10, 0, 0, 55], ufloat(1417.621,18.452)]
filename_list = ["Plots/Task2.4_1min_a.png", "Plots/Task2.4_1min_b.png", "Plots/Task2.4_2min_a.png", "Plots/Task2.4_2min_b.png", "Plots/Task2.4_4min_a.png", "Plots/Task2.4_4min_b.png", "Plots/Task2.4_6min.png", "Plots/Task2.4_8min.png", "Plots/Task2.4_16min.png"]
filename_list_zoomed = ["Plots/Task2.4_1min_a_zoomed.png", "Plots/Task2.4_1min_b_zoomed.png", "Plots/Task2.4_2min_a_zoomed.png", "Plots/Task2.4_2min_b_zoomed.png", "Plots/Task2.4_4min_a_zoomed.png", "Plots/Task2.4_4min_b_zoomed.png", "Plots/Task2.4_6min_zoomed.png", "Plots/Task2.4_8min_zoomed.png", "Plots/Task2.4_16min_zoomed.png"]
amplitude, T_range = ufloat(200/np.pi,0), ufloat(50,0)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057/2, 0.00001)
angular_freq_list = [np.pi/30, np.pi/30, np.pi/60, np.pi/60, np.pi/120, np.pi/120, np.pi/180, np.pi/240, np.pi/480]

delta_phi_list = [ufloat(600-(300-209.497),1.171), ufloat(600-(300-299.835),0.36), ufloat(1200-(600-264.214),0.079), ufloat(1200-(600-172.801),16.777), ufloat(1213.546,1.946), ufloat(1093.014,0.3), ufloat(1421.635,4.923), ufloat(1417.621,18.452), ufloat(1852.415,152.329)]
fixer_upper = [0, 0, 0, 0, np.pi, np.pi, np.pi, np.pi, np.pi]

gamma1, delta_phi1, fitting1 = [], [], []

for j in range(9):
    # print(j+1,"/9")
    file = dataset_list[j]
    x_data, y_data = np.loadtxt(file, unpack=True, skiprows=3)
    
    fitting, covariance = curve_fit(DoubleSinusoidal, x_data, y_data, p0 = p0_list[j], maxfev = 1000000)
    print(fitting)
    fitting1.append(fitting)
    ufloat_list = []
    for i in range(7):
        variables = ["a","b","c","d","e","f","g"]
        print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
        ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))
    # print(gamma, delta_phi)
    coefficients = []
    for l in range(len(fitting)):
        coefficients.append(ufloat(fitting[l], np.sqrt(covariance[l][l])))

    # The fitting for the original sine wave
    plt.figure(figsize = (8, 6))
    plt.plot(x_data, y_data, label='Data', **pointStyle)    
    plt.plot(x_data, Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
    plt.plot(x_data, DoubleSinusoidal(x_data, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
    # The actual fittings
    plt.plot(x_data, DoubleSinusoidalFlipped(x_data, amplitude.n, fitting[1], 0, fitting[3], fitting[4], fitting[5], fitting[6]), label='Expected Sine Wave', **lineStyleR)
    plt.plot(x_data, -Square(x_data, T_range.n, fitting[1], 0, 0) + Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), label='Expected Square Wave', **lineStyleG)
    # The difference between the actual and fitted sine waves
    plt.fill_between(x_data, DoubleSinusoidal(x_data, *fitting), DoubleSinusoidalFlipped(x_data, amplitude.n, fitting[1], 0, fitting[3], fitting[4], fitting[5], fitting[6]), color='red', alpha=0.2, label='Difference (??)')
    
    plt.suptitle("Task 2.4: First 'Back of the Envelope' Estimate of D", **titleFont)
    plt.xlabel("Time / ds", **axesFont)
    plt.ylabel("Temperature / ??C", **axesFont)
    plt.xticks(**ticksFont)
    plt.yticks(**ticksFont)
    
    gamma = np.abs(coefficients[0]/amplitude)
    delta_phi = 2*np.pi*(((delta_phi_list[j])/tau_list[j]))
    gamma_disp = "{:.12E}".format(gamma)
    delta_phi_disp = "{:.12E}".format(delta_phi)
    print("?? = ", gamma_disp)
    print("???? = ", delta_phi_disp, "s", phi_1_list[j])
    
    plot_title = formatter(gamma, delta_phi)
    
    gamma1.append(gamma.n)
    delta_phi1.append(delta_phi.n)
    
    plt.title(plot_title, **subtitleFont)
    plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.15), prop=font)
    # plt.savefig(filename_list[j], dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.clf()
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, label='Data', **pointStyle)
    plt.plot(x_data, Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
    plt.plot(x_data, DoubleSinusoidal(x_data, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)

    plt.suptitle("Task 2.4: 'Back of the Envelope' Estimate of D", **titleFont)
    plt.xlabel("Time / ds", **axesFont)
    plt.ylabel("Temperature / ??C", **axesFont)
    plt.xticks(**ticksFont)
    plt.yticks(**ticksFont)

    plt.title(plot_title, **subtitleFont)
    plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
    # plt.savefig(filename_list_zoomed[j], dpi=1000, bbox_inches='tight')
    # plt.show()
    plt.clf()
    
    coefficients = []
    for l in range(len(fitting)):
        coefficients.append("{:.12f}".format(ufloat(fitting[l], np.sqrt(covariance[l][l]))))
    coefficients.append(gamma)
    coefficients.append(delta_phi)
    
    # np.savetxt(dataset_list[j] + "_coefficients.txt", coefficients, fmt="%s")
    
    
print("END")
# print(gamma1)
# print(delta_phi1)
# print(fitting1)

print(D_gamma1)
print(D_delta_phi1)