import numpy as np
import scipy as sp
import uncertainties as unc
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import matplotlib.font_manager as fnt
from scipy.optimize import curve_fit
from scipy import interpolate
from uncertainties import ufloat
from uncertainties import unumpy

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

#%% Task 2.3 - First ‘back of the envelope’ estimate of 𝑫.

x1, y1 = np.loadtxt("thermal_4min_a.txt", unpack=True,skiprows=3)
x2, y2 = np.loadtxt("thermal_4min_b.txt", unpack=True,skiprows=3)

def sinusodial(x, a, b, c, d):  return a * np.sin(b * x + c) + d

tau = 1200
sin1_fit, sin1_cov = curve_fit(sinusodial, x1, y1, p0=[10.3, -0.00264, 0, 52])
square_wave = -sin1_fit[0] * sp.signal.square(2 * np.pi * (0.5/tau) * x1) + sin1_fit[3]
square_wave_2 = -sin1_fit[0] * sp.signal.square(2 * np.pi * (0.5/tau) * x1)
sin2_fit, sin2_cov = curve_fit(sinusodial, x1, y1, p0=[10.3, -0.00264, 0, 52])

tau, T_range, phi_1 = ufloat(1200,0), ufloat(50,0), ufloat(1093.014,0.3)
r_inner, r_outer = ufloat(0.00250, 0.00005), ufloat(0.02057, 0.00001)
ufloat_list = []
print("Given the form: y ≡ a*sin(bx+c)+d*sin(ex+f)+g, we have")
for i in range(7):
    variables = ["a","b","c","d","e","f","g"]
    print(variables[i],": ",sin2_fit[i],"±",np.sqrt(sin2_cov[i, i]))
    ufloat_list.append(ufloat(sin2_fit[i], np.sqrt(sin2_cov[i, i])))

# gamma = "{:.3f}".format(T_range/fitting[0])
# delta_phi = "{:.3f}".format(np.abs(tau - 1093.014))
gamma = (T_range/ufloat_list[0])**-1
delta_phi = np.abs(tau - phi_1)/10
D_gamma = (np.pi/2)*(r_inner - r_outer)**2/(2*unumpy.log(gamma)**2)
D_delta_phi = (np.pi/2)*(r_inner - r_outer)**2/(2*delta_phi**2)

print("γ: {:.6f} unitless".format(gamma))
print("Δφ: {:.6f} s".format(delta_phi))
print("Dᵧ: {:.8f} m²·s¯¹".format(D_gamma))
print("Dᵩ: {:.10f} m²·s¯¹".format(D_delta_phi))

plt.figure()
plt.plot(x1, sinusodial(x1, *sin1_fit), **lineStyleBold, label="Sinusodial Fit")
plt.plot(x1, square_wave, **lineStyleBold1, label="Square Wave")
plt.plot(x1,y1, **lineStyle, label="Data")
plt.suptitle("Task 2.3: First 'Back of the Envelope' Estimate of D", **titleFont)
plt.title("T = " + str(tau*2) + " ds", **subtitleFont)
plt.xlabel("Time / ds", **axesFont)
plt.ylabel("Temperature / K", **axesFont)
plt.xticks(**ticksFont)
plt.yticks(**ticksFont)
plt.legend(loc="lower left", bbox_to_anchor=(0.87, 0.995), prop=font)
plt.savefig("Plots/Task2.3_4A.png", dpi=1000)
plt.show()
