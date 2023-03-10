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

#%% 1 minute data
dataset_list = ["thermal_1min_a.txt","thermal_1min_b.txt"]

def Sinusoidal(x, a, b, c, d):  return a * np.sin(b * x + c) + d
def DoubleSinusoidal(x, a, b, c, d, e, f, g):  return a * np.sin(b * x + c) + d * np.sin(e * x + f) + g
def Square(x, a, b, c, d):  return a * np.sign(np.sin(b * x + c)) + d

i = 0
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
tau, T_range = 300, 50

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[0.1, -0.00864, 0, 10, 0,  0, 54.8], maxfev=100000)
print(fitting)

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
# plt.plot(x2, Square(x2, fitting[0], fitting[1], tau/2, 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, 200/np.pi, fitting[1], tau/2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], tau/2, 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, 200/np.pi, fitting[1], tau/2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
# plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                # Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                # color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.5: First 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
# plt.savefig("Plots/Task2.4_1min_a.png", dpi=1000, bbox_inches='tight')
plt.show()

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_1min_a_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()



tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(209.497,1.171)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.12f} m????s????".format(D_gamma))
print("D???: {:.12f} m????s????".format(D_delta_phi))








i = 1
tau, T_range = 300, 50
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[0.1, -0.00864, 0, 10, 0,  0, 54.8], maxfev=100000)
print(fitting)

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, 0.5, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, 0.5, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_1min_b.png", dpi=1000, bbox_inches='tight')
plt.show()

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, 0.5, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, 0.5, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_1min_b_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()



tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(299.835,0.36)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.12f} m????s????".format(D_gamma))
print("D???: {:.12f} m????s????".format(D_delta_phi))


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











#%% 2 minute data
dataset_list = ["thermal_2min_a.txt","thermal_2min_b.txt"]

i = 0
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
tau, T_range = 600, 50

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[32.5, -0.005, -5, 10, 0, 0, 52.5], maxfev=100000)
print(fitting)

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
# plt.savefig("Plots/Task2.4_2min_a.png", dpi=1000, bbox_inches='tight')
plt.show()


plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, fitting[0], fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_2min_a_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()


tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(264.214,0.079)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.8f} m????s????".format(D_gamma))
print("D???: {:.10f} m????s????".format(D_delta_phi))

# a = np.linspace(tau.nominal_value - 200, tau.nominal_value + 200, 1000)
# def Sinusoidal_NK(x, a=fitting[0], b=fitting[1], c=fitting[2], d=0):  
#     return a * np.sin(b * x + c) + d

# for i in a:
#     solutions = []
#     b = i
#     c = abs(int(round(i)))
#     for j in range(-c, c+1):
#         y = root(Sinusoidal_NK, j)
#         if y.success and (round(y.x[0], 6) not in solutions):
#             solutions.append(round(y.x[0], 3))
#     print(i, solutions)







i = 1
tau, T_range = 600, 50
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[2.5, -0.005, -5, 10, 0.001,  2, 62.2], maxfev=100000)
print(fitting)

gamma = "{:.3f}".format(T_range/fitting[0])
delta_phi = "{:.3f}".format(np.abs(tau - 172.801))

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, 0.5, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, 0.5, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_2min_b.png", dpi=1000, bbox_inches='tight')
plt.show()

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, fitting[0], fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_2min_b_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()

tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(172.801,16.777)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.8f} m????s????".format(D_gamma))
print("D???: {:.10f} m????s????".format(D_delta_phi))

# a = np.linspace(tau.nominal_value - 200, tau.nominal_value + 200, 1000)
# def Sinusoidal_NK(x, a=fitting[0], b=fitting[1], c=fitting[2], d=0):  
#     return a * np.sin(b * x + c) + d

# for i in a:
#     solutions = []
#     b = i
#     c = abs(int(round(i)))
#     for j in range(-c, c+1):
#         y = root(Sinusoidal_NK, j)
#         if y.success and (round(y.x[0], 6) not in solutions):
#             solutions.append(round(y.x[0], 3))
#     print(i, solutions)





#%% 4 minute data
x2, y2 = np.loadtxt("thermal_4min_a.txt", unpack=True,skiprows=3)
order = [0, 2, 1, 4, 3, 5, 6, 7]

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], maxfev=100000)
tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(1213.546,1.946)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi/2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi/2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} s".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.8f} m????s????".format(D_gamma))
print("D???: {:.10f} m????s????".format(D_delta_phi))

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range.nominal_value, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range.nominal_value, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range.nominal_value, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                 color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                 Square(x2, T_range.nominal_value, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                 color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau.nominal_value*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
           loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
plt.savefig("Plots/Task2.3_4A.png", dpi=1000, bbox_inches='tight')
plt.show()

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau.nominal_value*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.3_4A_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()

# a = np.linspace(tau.nominal_value - 200, tau.nominal_value + 200, 1000)
# def Sinusoidal_NK(x, a=fitting[0], b=fitting[1], c=fitting[2], d=0):  
#     return a * np.sin(b * x + c) + d

# for i in a:
#     solutions = []
#     b = i
#     c = abs(int(round(i)))
#     for j in range(-c, c+1):
#         y = root(Sinusoidal_NK, j)
#         if y.success and (round(y.x[0], 6) not in solutions):
#             solutions.append(round(y.x[0], 3))
#     print(i, solutions)








x2, y2 = np.loadtxt("thermal_4min_b.txt", unpack=True,skiprows=3)
order = [0, 2, 1, 4, 3, 5, 6, 7]

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[10.3, -0.00264, 0, 10, 0.00032, 11.9, 52], maxfev=100000)

tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(1093.014,0.3)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi/2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi/2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.8f} m????s????".format(D_gamma))
print("D???: {:.10f} m????s????".format(D_delta_phi))

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range.nominal_value, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range.nominal_value, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range.nominal_value, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                 color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                 Square(x2, T_range.nominal_value, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                 color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau.nominal_value*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
           loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.3_4B.png", dpi=1000, bbox_inches='tight')
plt.show()


plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau.nominal_value*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.3_4B_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()

# a = np.linspace(tau.nominal_value - 200, tau.nominal_value + 200, 1000)
# def Sinusoidal_NK(x, a=fitting[0], b=fitting[1], c=fitting[2], d=0):  
#     return a * np.sin(b * x + c) + d

# for i in a:
#     solutions = []
#     b = i
#     c = abs(int(round(i)))
#     for j in range(-c, c+1):
#         y = root(Sinusoidal_NK, j)
#         if y.success and (round(y.x[0], 6) not in solutions):
#             solutions.append(round(y.x[0], 3))
#     print(i, solutions)








#%% 6 minute data
dataset_list = ["thermal_6min.txt"]

i = 0
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
tau, T_range = 1800, 50

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[18, -0.0018, -6.5, 10, 0, 0, 50], maxfev=100000)
print(fitting)

gamma = "{:.3f}".format(T_range/fitting[0])
delta_phi = "{:.3f}".format(np.abs(tau - 1421.635))

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, 18.9, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (????)')

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
# plt.savefig("Plots/Task2.4_6min.png", dpi=1000, bbox_inches='tight')
plt.show()

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, 18.9, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_6min_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()

tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(1421.635,4.923)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.12f} m????s????".format(D_gamma))
print("D???: {:.12f} m????s????".format(D_delta_phi))




#%% 8 minute data
dataset_list = ["thermal_8min.txt"]

i = 0
x2, y2 = np.loadtxt(dataset_list[i], unpack=True,skiprows=3)
tau, T_range = 2400, 50

fitting, covariance = curve_fit(DoubleSinusoidal, x2, y2, p0=[32, -0.0013, -7.4, 10, 0, 0, 50], maxfev=100000)
print(fitting)

gamma = "{:.3f}".format(T_range/fitting[0])
delta_phi = "{:.3f}".format(np.abs(tau - 1417.62))

plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, *fitting), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)


plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.1), prop=font)
# plt.savefig("Plots/Task2.4_8min.png", dpi=1000, bbox_inches='tight')
plt.show()


plt.plot(x2, y2, label='Data', **pointStyle)
plt.plot(x2, Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), ls="dotted", label='Mean Value', **lineStyleBoldP)
plt.plot(x2, DoubleSinusoidal(x2, fitting[0], fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Fitted Sine Wave', **lineStyleBoldR)
plt.plot(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Square Wave', **lineStyleBoldG)

plt.suptitle("Task 2.5: 'Back of the Envelope' Estimate of D", **titleFont)
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (????)')

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_8min_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()

tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(1417.621,18.452)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.12f} m????s????".format(D_gamma))
print("D???: {:.12f} m????s????".format(D_delta_phi))







#%% 16 minute data
dataset_list = ["thermal_16min.txt"]
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
# plt.title("T = " + str(tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
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
# plt.title("T = " + str(2*tau) + " ds; ?? = " + str(gamma) + "; ???? = " + str(delta_phi), **subtitleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)

plt.plot(x2, DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Sine Wave', **lineStyleR)
plt.plot(x2, Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), label='Actual Square Wave', **lineStyleG)

plt.fill_between(x2, DoubleSinusoidal(x2, *fitting), DoubleSinusoidal(x2, T_range, fitting[1], fitting[2], fitting[3], fitting[4], fitting[5], fitting[6]),
                color='red', alpha=0.2, label='Difference (??)')
plt.fill_between(x2, Square(x2, fitting[0], fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]), 
                Square(x2, T_range, fitting[1], fitting[2], 0) + Sinusoidal(x2, fitting[3], fitting[4], fitting[5], fitting[6]),
                color='green', alpha=0.2, label='Difference (????)')

plt.legend(loc="center left", bbox_to_anchor=(0.82, 0.2), prop=font)
# plt.savefig("Plots/Task2.4_16min_zoomed.png", dpi=1000, bbox_inches='tight')
plt.show()



tau, T_range, phi_1 = ufloat(0,0), ufloat(200/np.pi,0), ufloat(1852.415,152.329)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ??? a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",fitting[i],"??",np.sqrt(covariance[i, i]))
    ufloat_list.append(ufloat(fitting[i], np.sqrt(covariance[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi*2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi*2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("??: {:.6f} unitless".format(gamma))
print("?????: {:.6f} ds".format(phi_1))
print("????: {:.6f} s".format(delta_phi))
print("D???: {:.12f} m????s????".format(D_gamma))
print("D???: {:.12f} m????s????".format(D_delta_phi))


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