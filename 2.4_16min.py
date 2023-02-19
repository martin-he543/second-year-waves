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

titleFont =      {'fontname': 'C059', 'size': 12}
subtitleFont =   {'fontname': 'C059', 'size': 9, 'style':'italic'}
axesFont =       {'fontname': 'C059', 'size': 9, 'style':'italic'}
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

dataset_list = ["thermal_16min.txt"]

def Sinusoidal(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def DoubleSinusoidal(x, a, b, c, d, e, f, g):  return a * np.sin(b * x + c) + d * np.sin(e * x + f) + g
def Square(x, a, b, c, d):  return a * np.sign(np.sin(b * x + c)) + d

i = 0
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
tau, T_range = 4800, 50

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[40, -0.00065, -2, 10, 0, 0, 55], maxfev=100000)
print(fitting)

gamma = "{:.3f}".format(T_range/fitting[0])
delta_phi = "{:.3f}".format(np.abs(tau - 1852.415))

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)


plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(tau) + " ds; γ = " + str(gamma) + "; Δφ = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
# plt.savefig("Plots/Task2.4_16min.png", dpi=1000, bbox_inches='tight')
plt.show()


plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, fitting[0], fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; γ = " + str(gamma) + "; Δφ = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (γ)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (Δφ)')

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_16min_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()



tau, T_range, phi_1 = ufloat(4800.000,0), ufloat(50,0), ufloat(1852.415,152.329)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ≡ a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"±",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("γ: {:.6f} unitless".format(gamma))
print("φ₁: {:.6f} ds".format(phi_1))
print("Δφ: {:.6f} s".format(delta_phi))
print("Dᵧ: {:.12f} m²·s¯¹".format(D_gamma))
print("Dᵩ: {:.12f} m²·s¯¹".format(D_delta_phi))


a = np.linspace(tau.nominal_value - 200, tau.nominal_value + 200, 1000)
def Sinusoidal_NK(x, a=fitting[0], b=fitting[1], c=fitting[2], d=0):  
    return a * np.sin(b * x + c) + d

# for i in a:
#     solutions = []
#     b = i
#     c = abs(int(round(i)))
#     for j in range(-c, c+1):
#         y = root(Sinusoidal_NK, j)
#         if y.success and (round(y.x[0], 6) not in solutions):
#             solutions.append(round(y.x[0], 3))
#     print(i, solutions)