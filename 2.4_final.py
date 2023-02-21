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
import sys

titleFont =      {'fontname': 'C059', 'size': 12}
subtitleFont =   {'fontname': 'C059', 'size': 7, 'style':'italic'}
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

#%% 1 minute data
# file='%s'%(sys.argv[1]) #this is the data
file = "1 minute (A).txt"
x_data, y_data = np.loadtxt(file, unpack=True,skiprows=3)

def Sinusoidal(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def DoubleSinusoidal(x, a, b, c, d, e, f, g):  return a * np.sin(b * x + c) + d * np.sin(e * x + f) + g
def Square(x, a, b, c, d):      return a * np.sign(np.sin(b * x + c)) + d
def substring_after(s, delim):  return s.partition(delim)[2]

i = 0
tau, T_range = 300, 50
fitting, covariance = curve_fit(DoubleSinusoidal, x_data, y_data, p0=[0.1, -0.00864, 0, 10, 0,  0, 54.8], maxfev=100000)
# plt.plot(x_data, Square(x_data, fitting[0], fitting[1], tau/2, 0) + Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)
# plt.fill_between(x_data, Square(x_data, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), 
                # Square(x_data, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]),
                # color='green', alpha=0.2, label='Difference (Δφ)')
plt.plot(x_data, y_data, label='Data', **pointStyle)
plt.plot(x_data, Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x_data, DoubleSinusoidal(x_data, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x_data, DoubleSinusoidal(x_data, 200/np.pi, fitting[1], tau/2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x_data, Square(x_data, T_range, fitting[1], tau/2, 0) + Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)
plt.fill_between(x_data, DoubleSinusoidal(x_data, *fitting), DoubleSinusoidal(x_data, 200/np.pi, fitting[1], tau/2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (γ)')
plt.suptitle("Task 2.5: First 'Back of the Envelope' Estimate of D", **titleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)


tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(209.497,1.171)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"±",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))
# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_gamma_val = "{:.4E}".format(D_gamma.n)
D_gamma_val = D_gamma_val.replace("E","× 10^")
D_gamma_unc = "{:.4E}".format(D_gamma.s)
D_gamma_unc = D_gamma_unc.replace("E","× 10^")
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)
D_delta_phi_val = "{:.4E}".format(D_delta_phi.n)
D_delta_phi_val = D_delta_phi_val.replace("E","× 10^")
D_delta_phi_unc = "{:.4E}".format(D_delta_phi.s)
D_delta_phi_unc = D_delta_phi_unc.replace("E","× 10^")


exp_delta_phi_val = int(substring_after(D_gamma_val, "× 10^"))

plot_title = "T = " + file.replace(".txt","") + ", γ = {:.4f} ± {:.4f}, Δφ = {:.4f} ± {:.4f}, \n$D_γ$ = ".format(gamma.n, gamma.s, delta_phi.n, delta_phi.s) + D_gamma_val + "±" + D_gamma_unc + "mm².s¯¹, $D_φ$ = " + D_delta_phi_val +  "±" + D_delta_phi_unc + "mm².s¯¹"

plt.title(plot_title, **subtitleFont)
plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.15), prop=font)
plt.show()
# plt.savefig("Plots/Task2.4_1min_a.png", dpi=1000, bbox_inches='tight')
# plt.plot(x_data, Square(x_data, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)
tau, T_range = 300, 50
plt.plot(x_data, y_data, label='Data', **pointStyle)
plt.plot(x_data, Sinusoidal(x_data, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x_data, DoubleSinusoidal(x_data, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.title(plot_title, **subtitleFont)
plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
# plt.savefig("Plots/Task2.4_1min_a_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()

print(fitting)
print("Given the form: y ≡ a*sin(bx+c)+d*sin(ex+f)+g, we have")
print("γ: {:.6f} unitless".format(gamma))
print("φ₁: {:.6f} ds".format(phi_1))
print("Δφ: {:.6f} s".format(delta_phi))
print("Dᵧ: {:.12f} m²·s¯¹".format(D_gamma))
print("Dᵩ: {:.12f} m²·s¯¹".format(D_delta_phi))