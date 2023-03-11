#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as integrate
import scipy.special as special
import math
from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt

time_4min_a, temp_4min_a = np.loadtxt("4 MINUTES (A).txt", skiprows = 3, unpack=True)
y_square =  50 * signal.square(time_4min_a*8*np.pi/len(time_4min_a)) + 50
# np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
y_square_fundamental = 100 * np.sin(time_4min_a*8*np.pi/len(time_4min_a)) * 2 / np.pi + 50

# Fourier Series

# Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.
    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-75, max=100)    # Make the parameter objects for all the terms
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-75, max=100)
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)    # Construct the series
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))    
    return series

x, y = variables('x, y')
w, = parameters('w', value=2*np.pi/2200, min=2*np.pi/2400, max=2*np.pi/2000)

model_dict_1 = {y: fourier_series(x, f=w, n=1)}
model_dict_2 = {y: fourier_series(x, f=w, n=2)}
model_dict_3 = {y: fourier_series(x, f=w, n=3)}
fit_1 = Fit(model_dict_1, x=time_4min_a, y=temp_4min_a)
fit_2 = Fit(model_dict_2, x=time_4min_a, y=temp_4min_a)
fit_3 = Fit(model_dict_3, x=time_4min_a, y=temp_4min_a)
fit_result_1 = fit_1.execute()
fit_result_2 = fit_2.execute()
fit_result_3 = fit_3.execute()

print('Fourier Series n = 1:', model_dict_1)
print('n = 1 Fit Results:', fit_result_1)
print(' ')
print('Fourier Series n = 2:', model_dict_2)
print('n = 2 Fit Results:', fit_result_2)
print(' ')
print('Fourier Series n = 3:', model_dict_3)
print('n = 3 Fit Results:', fit_result_3)

# Plot the result
plt.plot(time_4min_a, y_square, color = 'orange', label = 'Period = 4 min, idealised outer data')
plt.plot(time_4min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
plt.plot(time_4min_a, temp_4min_a, color = 'blue', label = 'Period = 4 min, inner data set a')
plt.plot(time_4min_a, fit_1.model(x=time_4min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
plt.plot(time_4min_a, fit_2.model(x=time_4min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
plt.plot(time_4min_a, fit_3.model(x=time_4min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
plt.xlabel('Time (ds)')
plt.ylabel('Temperature (°C)')
plt.ylim(-20, 120)
plt.title('Fourier Series Approximations for Period = 4 min, Data Set a')
plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
plt.savefig('task 2.6 4min a Fourier series fit.png', dpi = 1000)
plt.show()


# # Find positions of peaks, amplitudes, and phase lags.

# peak_pos_outer, properties_outer = signal.find_peaks(y_square_fundamental, height = np.mean(temp_4min_a))
# # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

# # print('Positions of outer peaks = ', peak_pos_outer) # [625.0, 3125.0, 5624.0, 8124.0]
# # print('Heights of outer peaks = ', properties_outer) # {'peak_heights': array([114.9436979 , 114.94367905, 114.94366019, 114.94369161])}

# # Find peak positions and height:
# peak_pos_inner_1, properties_inner_1 = signal.find_peaks(fit_1.model(x=time_4min_a, **fit_result_1.params).y, height = np.mean(temp_4min_a))
# peak_pos_inner_2, properties_inner_2 = signal.find_peaks(fit_2.model(x=time_4min_a, **fit_result_2.params).y, height = np.mean(temp_4min_a))
# peak_pos_inner_3, properties_inner_3 = signal.find_peaks(fit_3.model(x=time_4min_a, **fit_result_3.params).y, height = np.mean(temp_4min_a))

# # Find amplitudes and phase lags
# amp_inner_1 = properties_inner_1['peak_heights'] - np.mean(temp_4min_a)
# amp_inner_2 = properties_inner_2['peak_heights'] - np.mean(temp_4min_a)
# amp_inner_3 = properties_inner_3['peak_heights'] - np.mean(temp_4min_a)

# phase_lag_1 = (peak_pos_inner_1 - peak_pos_outer) * np.pi / 1200
# phase_lag_2 = (peak_pos_inner_2 - peak_pos_outer) * np.pi / 1200
# phase_lag_3 = (peak_pos_inner_3 - peak_pos_outer) * np.pi / 1200


# # Calculate mean amplitudes and phase lags for each Fourier fit:
# print(' ')
# print('Mean amplitude n = 1: ', np.mean(amp_inner_1), '+/-', (np.amax(amp_inner_1) - np.amin(amp_inner_1)) / 2)
# print('Mean amplitude n = 2: ', np.mean(amp_inner_2), '+/-', (np.amax(amp_inner_2) - np.amin(amp_inner_2)) / 2)
# print('Mean amplitude n = 3: ', np.mean(amp_inner_3), '+/-', (np.amax(amp_inner_3) - np.amin(amp_inner_3)) / 2)

# print('Mean phase lag n = 1: ', np.mean(phase_lag_1), '+/-', (np.amax(phase_lag_1) - np.amin(phase_lag_1)) / 2)
# print('Mean phase lag n = 2: ', np.mean(phase_lag_2), '+/-', (np.amax(phase_lag_2) - np.amin(phase_lag_2)) / 2)
# print('Mean phase lag n = 3: ', np.mean(phase_lag_3), '+/-', (np.amax(phase_lag_3) - np.amin(phase_lag_3)) / 2)

# #%%

# from uncertainties import ufloat
# import uncertainties.umath as umath

# # task 2.6, 4 min a, calculating D:

# # Convert everything to SI units - KEEP TEMPERATURES IN °C
# d_r = ufloat(7.87e-3, 0.05e-5)
# w_1 = ufloat(2.620562e-02, 1.937889e-06) # e-02 since we converted deciseconds to seconds.
# w_2 = ufloat(2.624387e-02, 1.892346e-06)
# w_3 = ufloat(2.623502e-02, 1.549044e-06)
# gamma_1 = ufloat(0.18708992, 0.00000006)
# gamma_2 = ufloat(0.18795198, 0.00000008)
# gamma_3 = ufloat(0.19278198, 0.00000002)
# d_phase_1 = ufloat(3.1429, 0.0092)
# d_phase_2 = ufloat(3.1416, 0.0236)
# d_phase_3 = ufloat(3.0984, 0.0196)

# # Calculate thermal diffusivity values:
# D_tf_1 = w_1 * (d_r ** 2) / (2 * (umath.log(gamma_1) ** 2))
# D_pl_1 = w_1 * (d_r ** 2) / (2 * (d_phase_1 ** 2))
# D_tf_2 = w_2 * (d_r ** 2) / (2 * (umath.log(gamma_2) ** 2))
# D_pl_2 = w_2 * (d_r ** 2) / (2 * (d_phase_2 ** 2))
# D_tf_3 = w_3 * (d_r ** 2) / (2 * (umath.log(gamma_3) ** 2))
# D_pl_3 = w_3 * (d_r ** 2) / (2 * (d_phase_3 ** 2))

# print('D_tf for n = 1: ', D_tf_1)
# print('D_pl for n = 1: ', D_pl_1)
# print('D_tf for n = 2: ', D_tf_2)
# print('D_pl for n = 2: ', D_pl_2)
# print('D_tf for n = 3: ', D_tf_3)
# print('D_pl for n = 3: ', D_pl_3)
# # %%
