#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as integrate
import scipy.special as special
import math

#%%

# Task 2.6 - period = 4 mins, data a.

# from symfit import parameters, variables, sin, cos, Fit
# import numpy as np
# import matplotlib.pyplot as plt

# time_4min_a, temp_4min_a = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Thermal and Electrical Waves\Thermal waves data\thermal_4min_a.txt", skiprows = 3, unpack=True)

# y_square =  50 * signal.square(time_4min_a*8*np.pi/len(time_4min_a)) + 50
# # np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
# y_square_fundamental = 100 * np.sin(time_4min_a*8*np.pi/len(time_4min_a)) * 2 / np.pi + 50

# # Fourier Series

# # Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

# def fourier_series(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-75, max=100)
#     sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-75, max=100)
#     # Construct the series
    
#     series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
#                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    
#     return series

# x, y = variables('x, y')
# w, = parameters('w', value=2*np.pi/2200, min=2*np.pi/2400, max=2*np.pi/2000)

# model_dict_1 = {y: fourier_series(x, f=w, n=1)}
# model_dict_2 = {y: fourier_series(x, f=w, n=2)}
# model_dict_3 = {y: fourier_series(x, f=w, n=3)}


# # Make step function data
# # xdata = np.linspace(-np.pi, np.pi)
# # ydata = np.zeros_like(xdata)
# # ydata[xdata > 0] = 1
# # Define a Fit object for this model and data
# fit_1 = Fit(model_dict_1, x=time_4min_a, y=temp_4min_a)
# fit_2 = Fit(model_dict_2, x=time_4min_a, y=temp_4min_a)
# fit_3 = Fit(model_dict_3, x=time_4min_a, y=temp_4min_a)
# fit_result_1 = fit_1.execute()
# fit_result_2 = fit_2.execute()
# fit_result_3 = fit_3.execute()

# print('Fourier Series n = 1:', model_dict_1)
# print('n = 1 Fit Results:', fit_result_1)
# print(' ')
# print('Fourier Series n = 2:', model_dict_2)
# print('n = 2 Fit Results:', fit_result_2)
# print(' ')
# print('Fourier Series n = 3:', model_dict_3)
# print('n = 3 Fit Results:', fit_result_3)

# # Plot the result
# plt.plot(time_4min_a, y_square, color = 'orange', label = 'Period = 4 min, idealised outer data')
# plt.plot(time_4min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
# plt.plot(time_4min_a, temp_4min_a, color = 'blue', label = 'Period = 4 min, inner data set a')
# plt.plot(time_4min_a, fit_1.model(x=time_4min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
# plt.plot(time_4min_a, fit_2.model(x=time_4min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
# plt.plot(time_4min_a, fit_3.model(x=time_4min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
# plt.xlabel('Time (ds)')
# plt.ylabel('Temperature (°C)')
# plt.ylim(-20, 120)
# plt.title('Fourier Series Approximations for Period = 4 min, Data Set a')
# plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.savefig('task 2.6 4min a Fourier series fit.png', dpi = 1000)
# plt.show()


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


#%%

# Task 2.7 - period = 1 min, data a.
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import scipy.integrate as integrate
# import scipy.special as special
# import math
# from symfit import parameters, variables, sin, cos, Fit
# import numpy as np
# import matplotlib.pyplot as plt

# time_1min_a, temp_1min_a = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Thermal and Electrical Waves\Thermal waves data\thermal_1min_a.txt", skiprows = 3, unpack=True)

# y_square =  50 * signal.square(time_1min_a*8*np.pi/len(time_1min_a)) + 50
# # np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
# y_square_fundamental = 100 * np.sin(time_1min_a*8*np.pi/len(time_1min_a)) * 2 / np.pi + 50


# # Fourier Series

# # Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

# def fourier_series(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-75, max=100)
#     sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-75, max=100)
#     # Construct the series
    
#     series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
#                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    
#     return series

# x, y = variables('x, y')
# w, = parameters('w', value=2*np.pi/600, min=2*np.pi/700, max=2*np.pi/500)

# model_dict_1 = {y: fourier_series(x, f=w, n=1)}
# model_dict_2 = {y: fourier_series(x, f=w, n=2)}
# model_dict_3 = {y: fourier_series(x, f=w, n=3)}


# # Make step function data
# # xdata = np.linspace(-np.pi, np.pi)
# # ydata = np.zeros_like(xdata)
# # ydata[xdata > 0] = 1
# # Define a Fit object for this model and data
# fit_1 = Fit(model_dict_1, x=time_1min_a, y=temp_1min_a)
# fit_2 = Fit(model_dict_2, x=time_1min_a, y=temp_1min_a)
# fit_3 = Fit(model_dict_3, x=time_1min_a, y=temp_1min_a)
# fit_result_1 = fit_1.execute()
# fit_result_2 = fit_2.execute()
# fit_result_3 = fit_3.execute()

# print('Fourier Series n = 1:', model_dict_1)
# print('n = 1 Fit Results:', fit_result_1)
# print(' ')
# print('Fourier Series n = 2:', model_dict_2)
# print('n = 2 Fit Results:', fit_result_2)
# print(' ')
# print('Fourier Series n = 3:', model_dict_3)
# print('n = 3 Fit Results:', fit_result_3)

# # Plot the result
# plt.plot(time_1min_a, y_square, color = 'orange', label = 'Period = 1 min, idealised outer data')
# plt.plot(time_1min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
# plt.plot(time_1min_a, temp_1min_a, color = 'blue', label = 'Period = 1 min, inner data set a')
# plt.plot(time_1min_a, fit_1.model(x=time_1min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
# plt.plot(time_1min_a, fit_2.model(x=time_1min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
# plt.plot(time_1min_a, fit_3.model(x=time_1min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
# plt.xlabel('Time (ds)')
# plt.ylabel('Temperature (°C)')
# plt.ylim(-20, 120)
# plt.title('Fourier Series Approximations for Period = 1 min, Data Set a')
# plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.savefig('task 2.7 1min a Fourier series fit.png', dpi = 1000)
# plt.show()


# # Find positions of peaks, amplitudes, and phase lags.

# peak_pos_outer, properties_outer = signal.find_peaks(y_square_fundamental, height = np.mean(temp_1min_a))
# # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

# # print('Positions of outer peaks = ', peak_pos_outer) # [625.0, 3125.0, 5624.0, 8124.0]
# # print('Heights of outer peaks = ', properties_outer) # {'peak_heights': array([114.9436979 , 114.94367905, 114.94366019, 114.94369161])}

# # Find peak positions and height:
# peak_pos_inner_1, properties_inner_1 = signal.find_peaks(fit_1.model(x=time_1min_a, **fit_result_1.params).y, height = np.mean(temp_1min_a))
# peak_pos_inner_2, properties_inner_2 = signal.find_peaks(fit_2.model(x=time_1min_a, **fit_result_2.params).y, height = np.mean(temp_1min_a))
# peak_pos_inner_3, properties_inner_3 = signal.find_peaks(fit_3.model(x=time_1min_a, **fit_result_3.params).y, height = np.mean(temp_1min_a))

# # Find amplitudes and phase lags
# amp_inner_1 = properties_inner_1['peak_heights'] - np.mean(temp_1min_a)
# amp_inner_2 = properties_inner_2['peak_heights'] - np.mean(temp_1min_a)
# amp_inner_3 = properties_inner_3['peak_heights'] - np.mean(temp_1min_a)

# print(peak_pos_inner_1)
# print(peak_pos_inner_2)
# print(peak_pos_inner_3)
# print(peak_pos_outer)


# phase_lag_1 = (peak_pos_inner_1[1:] - peak_pos_outer[:-1]) * np.pi / 300
# phase_lag_2 = (peak_pos_inner_2[1:] - peak_pos_outer[:-1]) * np.pi / 300
# phase_lag_3 = (peak_pos_inner_3[1:] - peak_pos_outer[:-1]) * np.pi / 300

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

# # task 2.7, 1 min a, calculating D:

# # Convert everything to SI units - KEEP TEMPERATURES IN °C
# d_r = ufloat(7.87e-3, 0.05e-5)
# w_1 = ufloat(1.043449e-01, 1.047692e-05) # e-01 since units converted from decisecods to seconds. 
# w_2 = ufloat(1.043616e-01, 1.054891e-05)
# w_3 = ufloat(1.043602e-01, 1.058764e-05)
# gamma_1 = ufloat(0.02251114, 0.00000012)
# gamma_2 = ufloat(0.02255376, 0.00000002)
# gamma_3 = ufloat(0.02262250, 0.00000006)
# d_phase_1 = ufloat(5.3617, 0.0209)
# d_phase_2 = ufloat(5.3302, 0.0209)
# d_phase_3 = ufloat(5.3442, 0.0262)

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

# %%

# Task 2.7 - period = 2 min, data a.
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import scipy.integrate as integrate
# import scipy.special as special
# import math
# from symfit import parameters, variables, sin, cos, Fit
# import numpy as np
# import matplotlib.pyplot as plt

# time_2min_a, temp_2min_a = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Thermal and Electrical Waves\Thermal waves data\thermal_2min_a.txt", skiprows = 3, unpack=True)


# y_square =  50 * signal.square(time_2min_a*8*np.pi/len(time_2min_a)) + 50
# # np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
# y_square_fundamental = 100 * np.sin(time_2min_a*8*np.pi/len(time_2min_a)) * 2 / np.pi + 50


# # Fourier Series

# # Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

# def fourier_series(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-75, max=100)
#     sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-75, max=100)
#     # Construct the series
    
#     series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
#                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    
#     return series

# x, y = variables('x, y')
# w, = parameters('w', value = 2*np.pi/1200, min = 2*np.pi/1300, max = 2*np.pi/1100)

# model_dict_1 = {y: fourier_series(x, f=w, n=1)}
# model_dict_2 = {y: fourier_series(x, f=w, n=2)}
# model_dict_3 = {y: fourier_series(x, f=w, n=3)}


# # Make step function data
# # xdata = np.linspace(-np.pi, np.pi)
# # ydata = np.zeros_like(xdata)
# # ydata[xdata > 0] = 1
# # Define a Fit object for this model and data
# fit_1 = Fit(model_dict_1, x=time_2min_a, y=temp_2min_a)
# fit_2 = Fit(model_dict_2, x=time_2min_a, y=temp_2min_a)
# fit_3 = Fit(model_dict_3, x=time_2min_a, y=temp_2min_a)
# fit_result_1 = fit_1.execute()
# fit_result_2 = fit_2.execute()
# fit_result_3 = fit_3.execute()

# print('Fourier Series n = 1:', model_dict_1)
# print('n = 1 Fit Results:', fit_result_1)
# print(' ')
# print('Fourier Series n = 2:', model_dict_2)
# print('n = 2 Fit Results:', fit_result_2)
# print(' ')
# print('Fourier Series n = 3:', model_dict_3)
# print('n = 3 Fit Results:', fit_result_3)

# # Plot the result
# plt.plot(time_2min_a, y_square, color = 'orange', label = 'Period = 2 min, idealised outer data')
# plt.plot(time_2min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
# plt.plot(time_2min_a, temp_2min_a, color = 'blue', label = 'Period = 2 min, inner data set a')
# plt.plot(time_2min_a, fit_1.model(x=time_2min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
# plt.plot(time_2min_a, fit_2.model(x=time_2min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
# plt.plot(time_2min_a, fit_3.model(x=time_2min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
# plt.xlabel('Time (ds)')
# plt.ylabel('Temperature (°C)')
# plt.ylim(-20, 120)
# plt.title('Fourier Series Approximations for Period = 2 min, Data Set a')
# plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.savefig('task 2.7 2min a Fourier series fit.png', dpi = 1000)
# plt.show()


# # Find positions of peaks, amplitudes, and phase lags.

# peak_pos_outer, properties_outer = signal.find_peaks(y_square_fundamental, height = np.mean(temp_2min_a))
# # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

# # print('Positions of outer peaks = ', peak_pos_outer) # [625.0, 3125.0, 5624.0, 8124.0]
# # print('Heights of outer peaks = ', properties_outer) # {'peak_heights': array([114.9436979 , 114.94367905, 114.94366019, 114.94369161])}

# # Find peak positions and height:
# peak_pos_inner_1, properties_inner_1 = signal.find_peaks(fit_1.model(x=time_2min_a, **fit_result_1.params).y, height = np.mean(temp_2min_a))
# peak_pos_inner_2, properties_inner_2 = signal.find_peaks(fit_2.model(x=time_2min_a, **fit_result_2.params).y, height = np.mean(temp_2min_a))
# peak_pos_inner_3, properties_inner_3 = signal.find_peaks(fit_3.model(x=time_2min_a, **fit_result_3.params).y, height = np.mean(temp_2min_a))

# # Find amplitudes and phase lags
# amp_inner_1 = properties_inner_1['peak_heights'] - np.mean(temp_2min_a)
# amp_inner_2 = properties_inner_2['peak_heights'] - np.mean(temp_2min_a)
# amp_inner_3 = properties_inner_3['peak_heights'] - np.mean(temp_2min_a)

# print(peak_pos_inner_1)
# print(peak_pos_inner_2)
# print(peak_pos_inner_3)
# print(peak_pos_outer)

# phase_lag_1 = (peak_pos_inner_1 - peak_pos_outer) * np.pi / 600
# phase_lag_2 = (peak_pos_inner_2 - peak_pos_outer) * np.pi / 600
# phase_lag_3 = (peak_pos_inner_3 - peak_pos_outer) * np.pi / 600

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

# # task 2.7, 2 min a, calculating D:

# # Convert everything to SI units - KEEP TEMPERATURES IN °C
# d_r = ufloat(7.87e-3, 0.05e-5)
# w_1 = ufloat(5.235980e-02, 1.786207e-05) # e-02 since units converted from decisecods to seconds. 
# w_2 = ufloat(5.235288e-02, 1.788434e-05)
# w_3 = ufloat(5.234880e-02, 1.773368e-05)
# gamma_1 = ufloat(0.04896091, 0.00000002)
# gamma_2 = ufloat(0.04856552, 0.00000006)
# gamma_3 = ufloat(0.04791374, 0.00000006)
# d_phase_1 = ufloat(4.5396, 0.0273)
# d_phase_2 = ufloat(4.5632, 0.0265)
# d_phase_3 = ufloat(4.6011, 0.0261)

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

# %%

# Task 2.7 - period = 6 min, data.
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import scipy.integrate as integrate
# import scipy.special as special
# import math
# from symfit import parameters, variables, sin, cos, Fit
# import numpy as np
# import matplotlib.pyplot as plt

# time_6min_a, temp_6min_a = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Thermal and Electrical Waves\Thermal waves data\thermal_6min.txt", skiprows = 3, unpack=True)


# y_square =  50 * signal.square(time_6min_a*8*np.pi/len(time_6min_a)) + 50
# # np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
# y_square_fundamental = 100 * np.sin(time_6min_a*8*np.pi/len(time_6min_a)) * 2 / np.pi + 50


# # Fourier Series

# # Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

# def fourier_series_1(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-80, max=80)
#     sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-80, max=80)
#     # a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-80, max=80)
#     # sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-80, max=80)
#     # Construct the series
    
#     series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
#                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    
#     return series


# def fourier_series_2(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-1000000, max=1000000)
#     sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-1000000, max=1000000)
#     # c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-50, max=100)
#     # sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-150, max=150)
#     # Construct the series
    
#     series = c0 + sum(ci * cos(i * f * x) + di * sin(i * f * x)
#                      for i, (ci, di) in enumerate(zip(cos_c, sin_d), start=1))
    
#     return series


# def fourier_series_3(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     e0, *cos_e = parameters(','.join(['e{}'.format(i) for i in range(0, n + 1)]), min=-10000000, max=10000000)
#     sin_f = parameters(','.join(['f{}'.format(i) for i in range(1, n + 1)]), min=-10000000, max=10000000)
#     # Construct the series
    
#     series = e0 + sum(ei * cos(i * f * x) + fi * sin(i * f * x)
#                      for i, (ei, fi) in enumerate(zip(cos_e, sin_f), start=1))
    
#     return series


# x, y = variables('x, y')
# w, = parameters('w', value = 2*np.pi/3600, min = 2*np.pi/3700, max = 2*np.pi/3500)

# model_dict_1 = {y: fourier_series_1(x, f=w, n=1)}
# model_dict_2 = {y: fourier_series_2(x, f=w, n=2)}
# model_dict_3 = {y: fourier_series_3(x, f=w, n=3)}

# print(model_dict_1)
# print(model_dict_2)
# print(model_dict_3)

# # Make step function data
# # xdata = np.linspace(-np.pi, np.pi)
# # ydata = np.zeros_like(xdata)
# # ydata[xdata > 0] = 1
# # Define a Fit object for this model and data
# fit_1 = Fit(model_dict_1, x=time_6min_a, y=temp_6min_a)
# fit_2 = Fit(model_dict_2, x=time_6min_a, y=temp_6min_a)
# fit_3 = Fit(model_dict_3, x=time_6min_a, y=temp_6min_a)
# fit_result_1 = fit_1.execute()
# fit_result_2 = fit_2.execute()
# fit_result_3 = fit_3.execute()

# print('Fourier Series n = 1:', model_dict_1)
# print('n = 1 Fit Results:', fit_result_1)
# print(' ')
# print('Fourier Series n = 2:', model_dict_2)
# print('n = 2 Fit Results:', fit_result_2)
# print(' ')
# print('Fourier Series n = 3:', model_dict_3)
# print('n = 3 Fit Results:', fit_result_3)

# # Plot the result
# plt.plot(time_6min_a, y_square, color = 'orange', label = 'Period = 6 min, idealised outer data')
# plt.plot(time_6min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
# plt.plot(time_6min_a, temp_6min_a, color = 'blue', label = 'Period = 6 min, inner data set a')
# plt.plot(time_6min_a, fit_1.model(x=time_6min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
# plt.plot(time_6min_a, fit_2.model(x=time_6min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
# plt.plot(time_6min_a, fit_3.model(x=time_6min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
# plt.xlabel('Time (ds)')
# plt.ylabel('Temperature (°C)')
# plt.ylim(-20, 120)
# plt.title('Fourier Series Approximations for Period = 6 min, Data Set a')
# plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.savefig('task 2.7 6min a Fourier series fit.png', dpi = 1000)
# plt.show()


# # Find positions of peaks, amplitudes, and phase lags.

# peak_pos_outer, properties_outer = signal.find_peaks(y_square_fundamental, height = np.mean(temp_6min_a))
# # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

# # print('Positions of outer peaks = ', peak_pos_outer) # [625.0, 3125.0, 5624.0, 8124.0]
# # print('Heights of outer peaks = ', properties_outer) # {'peak_heights': array([114.9436979 , 114.94367905, 114.94366019, 114.94369161])}

# # Find peak positions and height:
# peak_pos_inner_1, properties_inner_1 = signal.find_peaks(fit_1.model(x=time_6min_a, **fit_result_1.params).y, height = np.mean(temp_6min_a))
# peak_pos_inner_2, properties_inner_2 = signal.find_peaks(fit_2.model(x=time_6min_a, **fit_result_2.params).y, height = np.mean(temp_6min_a))
# peak_pos_inner_3, properties_inner_3 = signal.find_peaks(fit_3.model(x=time_6min_a, **fit_result_3.params).y, height = np.mean(temp_6min_a))

# # Find amplitudes and phase lags
# amp_inner_1 = properties_inner_1['peak_heights'] - np.mean(temp_6min_a)
# amp_inner_2 = properties_inner_2['peak_heights'] - np.mean(temp_6min_a)
# amp_inner_3 = properties_inner_3['peak_heights'] - np.mean(temp_6min_a)

# print(peak_pos_inner_1)
# print(peak_pos_inner_2)
# print(peak_pos_inner_3)
# print(peak_pos_outer)


# phase_lag_1 = (peak_pos_inner_1 - peak_pos_outer) * np.pi / 1800
# phase_lag_2 = (peak_pos_inner_2 - peak_pos_outer) * np.pi / 1800
# phase_lag_3 = (peak_pos_inner_3 - peak_pos_outer) * np.pi / 1800

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

# # task 2.7, 6 min a, calculating D:

# # Convert everything to SI units - KEEP TEMPERATURES IN °C
# d_r = ufloat(7.87e-3, 0.05e-5)
# w_1 = ufloat(1.745075e-02, 1.224153e-06) # e-02 since units converted from decisecods to seconds. 
# w_2 = ufloat(1.744943e-02, 1.227824e-06)
# w_3 = ufloat(1.744801e-02, 6.651489e-07)
# gamma_1 = ufloat(0.35220290, 0.00000004)
# gamma_2 = ufloat(0.35408232, 0.00000002)
# gamma_3 = ufloat(0.37059926, 0.00000008)
# d_phase_1 = ufloat(2.4801, 0.0175)
# d_phase_2 = ufloat(2.4662, 0.0349)
# d_phase_3 = ufloat(2.5098, 0.0349)

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


# %%

# Task 2.7 - period = 8 min, data.

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.integrate as integrate
import scipy.special as special
import math
from symfit import parameters, variables, sin, cos, Fit
import numpy as np
import matplotlib.pyplot as plt

time_8min_a, temp_8min_a = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Thermal and Electrical Waves\Thermal waves data\thermal_8min.txt", skiprows = 3, unpack=True)


y_square =  50 * signal.square(time_8min_a*8*np.pi/len(time_8min_a)) + 50
# np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
y_square_fundamental = 100 * np.sin(time_8min_a*8*np.pi/len(time_8min_a)) * 2 / np.pi + 50


# Fourier Series

# Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

def fourier_series_1(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-1000000, max=1000000)
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-1000000, max=1000000)
    # a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-80, max=80)
    # sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-80, max=80)
    # Construct the series
    
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    
    return series


def fourier_series_2(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-1000000, max=1000000)
    sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-1000000, max=1000000)
    # c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-50, max=100)
    # sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-150, max=150)
    # Construct the series
    
    series = c0 + sum(ci * cos(i * f * x) + di * sin(i * f * x)
                     for i, (ci, di) in enumerate(zip(cos_c, sin_d), start=1))
    
    return series


def fourier_series_3(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    e0, *cos_e = parameters(','.join(['e{}'.format(i) for i in range(0, n + 1)]), min=-10000000, max=10000000)
    sin_f = parameters(','.join(['f{}'.format(i) for i in range(1, n + 1)]), min=-10000000, max=10000000)
    # Construct the series
    
    series = e0 + sum(ei * cos(i * f * x) + fi * sin(i * f * x)
                     for i, (ei, fi) in enumerate(zip(cos_e, sin_f), start=1))
    
    return series


def fourier_series_10(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    g0, *cos_g = parameters(','.join(['g{}'.format(i) for i in range(0, n + 1)]), min=-10000000000, max=10000000000)
    sin_h = parameters(','.join(['h{}'.format(i) for i in range(1, n + 1)]), min=-10000000000, max=10000000000)
    # c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-50, max=100)
    # sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-150, max=150)
    # Construct the series
    
    series = g0 + sum(gi * cos(i * f * x) + hi * sin(i * f * x)
                     for i, (gi, hi) in enumerate(zip(cos_g, sin_h), start=1))
    
    return series



x, y = variables('x, y')
w, = parameters('w', value = 2*np.pi/4800, min = 2*np.pi/4900, max = 2*np.pi/4700)

model_dict_1 = {y: fourier_series_1(x, f=w, n=1)}
model_dict_2 = {y: fourier_series_2(x, f=w, n=2)}
model_dict_3 = {y: fourier_series_3(x, f=w, n=3)}
model_dict_10 = {y: fourier_series_10(x, f=w, n=10)}

print(model_dict_1)
print(model_dict_2)
print(model_dict_3)
print(model_dict_10)

# Make step function data
# xdata = np.linspace(-np.pi, np.pi)
# ydata = np.zeros_like(xdata)
# ydata[xdata > 0] = 1
# Define a Fit object for this model and data
fit_1 = Fit(model_dict_1, x=time_8min_a, y=temp_8min_a)
fit_2 = Fit(model_dict_2, x=time_8min_a, y=temp_8min_a)
fit_3 = Fit(model_dict_3, x=time_8min_a, y=temp_8min_a)
fit_10 = Fit(model_dict_10, x=time_8min_a, y=temp_8min_a)
fit_result_1 = fit_1.execute()
fit_result_2 = fit_2.execute()
fit_result_3 = fit_3.execute()
fit_result_10 = fit_10.execute()

print('Fourier Series n = 1:', model_dict_1)
print('n = 1 Fit Results:', fit_result_1)
print(' ')
print('Fourier Series n = 2:', model_dict_2)
print('n = 2 Fit Results:', fit_result_2)
print(' ')
print('Fourier Series n = 3:', model_dict_3)
print('n = 3 Fit Results:', fit_result_3)
print(' ')
print('Fourier Series n = 10:', model_dict_10)
print('n = 10 Fit Results:', fit_result_10)

# Plot the result
plt.plot(time_8min_a, y_square, color = 'orange', label = 'Period = 8 min, idealised outer data')
plt.plot(time_8min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
plt.plot(time_8min_a, temp_8min_a, color = 'blue', label = 'Period = 8 min, inner data set a')
plt.plot(time_8min_a, fit_1.model(x=time_8min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
plt.plot(time_8min_a, fit_2.model(x=time_8min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
plt.plot(time_8min_a, fit_3.model(x=time_8min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
plt.plot(time_8min_a, fit_10.model(x=time_8min_a, **fit_result_10.params).y, color = 'violet', label = 'Fourier series n = 10, inner')
plt.xlabel('Time (ds)')
plt.ylabel('Temperature (°C)')
plt.ylim(-20, 120)
plt.title('Fourier Series Approximations for Period = 8 min, Data Set a')
plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
plt.savefig('task 2.7 8min a Fourier series fit.png', dpi = 1000)
plt.show()


# Find positions of peaks, amplitudes, and phase lags.

peak_pos_outer, properties_outer = signal.find_peaks(y_square_fundamental, height = np.mean(temp_8min_a))
# Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

# print('Positions of outer peaks = ', peak_pos_outer) # [625.0, 3125.0, 5624.0, 8124.0]
# print('Heights of outer peaks = ', properties_outer) # {'peak_heights': array([114.9436979 , 114.94367905, 114.94366019, 114.94369161])}

# Find peak positions and height:
peak_pos_inner_1, properties_inner_1 = signal.find_peaks(fit_1.model(x=time_8min_a, **fit_result_1.params).y, height = np.mean(temp_8min_a))
peak_pos_inner_2, properties_inner_2 = signal.find_peaks(fit_2.model(x=time_8min_a, **fit_result_2.params).y, height = np.mean(temp_8min_a))
peak_pos_inner_3, properties_inner_3 = signal.find_peaks(fit_3.model(x=time_8min_a, **fit_result_3.params).y, height = np.mean(temp_8min_a))
peak_pos_inner_10, properties_inner_10 = signal.find_peaks(fit_10.model(x=time_8min_a, **fit_result_10.params).y, height = np.mean(temp_8min_a))

# Find amplitudes and phase lags
amp_inner_1 = properties_inner_1['peak_heights'] - np.mean(temp_8min_a)
amp_inner_2 = properties_inner_2['peak_heights'] - np.mean(temp_8min_a)
amp_inner_3 = properties_inner_3['peak_heights'] - np.mean(temp_8min_a)
amp_inner_10 = properties_inner_10['peak_heights'] - np.mean(temp_8min_a)

print(peak_pos_inner_1)
print(peak_pos_inner_2)
print(peak_pos_inner_3)
print(peak_pos_inner_10)
print(peak_pos_outer)

phase_lag_1 = (peak_pos_inner_1 - peak_pos_outer) * np.pi / 2400
phase_lag_2 = (peak_pos_inner_2 - peak_pos_outer) * np.pi / 2400
phase_lag_3 = (peak_pos_inner_3 - peak_pos_outer) * np.pi / 2400
phase_lag_10 = (peak_pos_inner_10 - peak_pos_outer) * np.pi / 2400

# Calculate mean amplitudes and phase lags for each Fourier fit:
print(' ')
print('Mean amplitude n = 1: ', np.mean(amp_inner_1), '+/-', (np.amax(amp_inner_1) - np.amin(amp_inner_1)) / 2)
print('Mean amplitude n = 2: ', np.mean(amp_inner_2), '+/-', (np.amax(amp_inner_2) - np.amin(amp_inner_2)) / 2)
print('Mean amplitude n = 3: ', np.mean(amp_inner_3), '+/-', (np.amax(amp_inner_3) - np.amin(amp_inner_3)) / 2)
print('Mean amplitude n = 10: ', np.mean(amp_inner_10), '+/-', (np.amax(amp_inner_10) - np.amin(amp_inner_10)) / 2)

print('Mean phase lag n = 1: ', np.mean(phase_lag_1), '+/-', (np.amax(phase_lag_1) - np.amin(phase_lag_1)) / 2)
print('Mean phase lag n = 2: ', np.mean(phase_lag_2), '+/-', (np.amax(phase_lag_2) - np.amin(phase_lag_2)) / 2)
print('Mean phase lag n = 3: ', np.mean(phase_lag_3), '+/-', (np.amax(phase_lag_3) - np.amin(phase_lag_3)) / 2)
print('Mean phase lag n = 10: ', np.mean(phase_lag_10), '+/-', (np.amax(phase_lag_10) - np.amin(phase_lag_10)) / 2)

#%%

from uncertainties import ufloat
import uncertainties.umath as umath

# task 2.7, 8 min a, calculating D:

# Convert everything to SI units - KEEP TEMPERATURES IN °C
d_r = ufloat(7.87e-3, 0.05e-5)
w_1 = ufloat(1.310618e-02, 1.331324e-06) # e-02 since units converted from decisecods to seconds. 
w_2 = ufloat(1.311917e-02, 1.306500e-06)
w_3 = ufloat(1.312091e-02, 6.651489e-07)
w_10 = ufloat(1.311379e-02, 5.697898e-07)
gamma_1 = ufloat(0.56487488, 0.00000002)
gamma_2 = ufloat(0.57497840, 0.00000006)
gamma_3 = ufloat(0.59995116, 0.00000008)
gamma_10 = ufloat(0.61117048, 0.00000010)
d_phase_1 = ufloat(1.8431, 0.0117)
d_phase_2 = ufloat(1.8244, 0.0209)
d_phase_3 = ufloat(1.9965, 0.0223)
d_phase_10 = ufloat(2.0391, 0.0170)

# Calculate thermal diffusivity values:
D_tf_1 = w_1 * (d_r ** 2) / (2 * (umath.log(gamma_1) ** 2))
D_pl_1 = w_1 * (d_r ** 2) / (2 * (d_phase_1 ** 2))
D_tf_2 = w_2 * (d_r ** 2) / (2 * (umath.log(gamma_2) ** 2))
D_pl_2 = w_2 * (d_r ** 2) / (2 * (d_phase_2 ** 2))
D_tf_3 = w_3 * (d_r ** 2) / (2 * (umath.log(gamma_3) ** 2))
D_pl_3 = w_3 * (d_r ** 2) / (2 * (d_phase_3 ** 2))
D_tf_10 = w_10 * (d_r ** 2) / (2 * (umath.log(gamma_10) ** 2))
D_pl_10 = w_10 * (d_r ** 2) / (2 * (d_phase_10 ** 2))

print('D_tf for n = 1: ', D_tf_1)
print('D_pl for n = 1: ', D_pl_1)
print('D_tf for n = 2: ', D_tf_2)
print('D_pl for n = 2: ', D_pl_2)
print('D_tf for n = 3: ', D_tf_3)
print('D_pl for n = 3: ', D_pl_3)
print('D_tf for n = 10: ', D_tf_10)
print('D_pl for n = 10: ', D_pl_10)

#%%

# Task 2.7 - period = 16 min, data.

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# import scipy.integrate as integrate
# import scipy.special as special
# import math
# from symfit import parameters, variables, sin, cos, Fit
# import numpy as np
# import matplotlib.pyplot as plt

# time_16min_a, temp_16min_a = np.loadtxt(r"C:\Users\ginta\OneDrive\HIROKI\Imperial College London\Year 2\Y2 Practical lab\Thermal and Electrical Waves\Thermal waves data\thermal_16min.txt", skiprows = 3, unpack=True)

# y_square =  50 * signal.square(time_16min_a*8*np.pi/len(time_16min_a)) + 50
# # np.mean(temp_4min_a) + ((temp_4min_a.max() - temp_4min_a.min()) / 2)
# y_square_fundamental = 100 * np.sin(time_16min_a*8*np.pi/len(time_16min_a)) * 2 / np.pi + 50


# # Fourier Series

# # Source https://symfit.readthedocs.io/en/stable/examples/ex_fourier_series.html

# def fourier_series_1(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-1000000, max=1000000)
#     sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-1000000, max=1000000)
#     # a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]), min=-80, max=80)
#     # sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]), min=-80, max=80)
#     # Construct the series
    
#     series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
#                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    
#     return series


# def fourier_series_2(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-1000000, max=1000000)
#     sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-1000000, max=1000000)
#     # c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-50, max=100)
#     # sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-150, max=150)
#     # Construct the series
    
#     series = c0 + sum(ci * cos(i * f * x) + di * sin(i * f * x)
#                      for i, (ci, di) in enumerate(zip(cos_c, sin_d), start=1))
    
#     return series


# def fourier_series_3(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     e0, *cos_e = parameters(','.join(['e{}'.format(i) for i in range(0, n + 1)]), min=-10000000, max=10000000)
#     sin_f = parameters(','.join(['f{}'.format(i) for i in range(1, n + 1)]), min=-10000000, max=10000000)
#     # Construct the series
    
#     series = e0 + sum(ei * cos(i * f * x) + fi * sin(i * f * x)
#                      for i, (ei, fi) in enumerate(zip(cos_e, sin_f), start=1))
    
#     return series


# def fourier_series_10(x, f, n=0):
#     """
#     Returns a symbolic fourier series of order `n`.

#     :param n: Order of the fourier series.
#     :param x: Independent variable
#     :param f: Frequency of the fourier series
#     """
#     # Make the parameter objects for all the terms
#     g0, *cos_g = parameters(','.join(['g{}'.format(i) for i in range(0, n + 1)]), min=-10000000000, max=10000000000)
#     sin_h = parameters(','.join(['h{}'.format(i) for i in range(1, n + 1)]), min=-10000000000, max=10000000000)
#     # c0, *cos_c = parameters(','.join(['c{}'.format(i) for i in range(0, n + 1)]), min=-50, max=100)
#     # sin_d = parameters(','.join(['d{}'.format(i) for i in range(1, n + 1)]), min=-150, max=150)
#     # Construct the series
    
#     series = g0 + sum(gi * cos(i * f * x) + hi * sin(i * f * x)
#                      for i, (gi, hi) in enumerate(zip(cos_g, sin_h), start=1))
    
#     return series



# x, y = variables('x, y')
# w, = parameters('w', value = 2*np.pi/9600, min = 2*np.pi/9700, max = 2*np.pi/9500)

# model_dict_1 = {y: fourier_series_1(x, f=w, n=1)}
# model_dict_2 = {y: fourier_series_2(x, f=w, n=2)}
# model_dict_3 = {y: fourier_series_3(x, f=w, n=3)}
# model_dict_10 = {y: fourier_series_10(x, f=w, n=10)}

# print(model_dict_1)
# print(model_dict_2)
# print(model_dict_3)
# print(model_dict_10)

# # Make step function data
# # xdata = np.linspace(-np.pi, np.pi)
# # ydata = np.zeros_like(xdata)
# # ydata[xdata > 0] = 1
# # Define a Fit object for this model and data
# fit_1 = Fit(model_dict_1, x=time_16min_a, y=temp_16min_a)
# fit_2 = Fit(model_dict_2, x=time_16min_a, y=temp_16min_a)
# fit_3 = Fit(model_dict_3, x=time_16min_a, y=temp_16min_a)
# fit_10 = Fit(model_dict_10, x=time_16min_a, y=temp_16min_a)
# fit_result_1 = fit_1.execute()
# fit_result_2 = fit_2.execute()
# fit_result_3 = fit_3.execute()
# fit_result_10 = fit_10.execute()

# print('Fourier Series n = 1:', model_dict_1)
# print('n = 1 Fit Results:', fit_result_1)
# print(' ')
# print('Fourier Series n = 2:', model_dict_2)
# print('n = 2 Fit Results:', fit_result_2)
# print(' ')
# print('Fourier Series n = 3:', model_dict_3)
# print('n = 3 Fit Results:', fit_result_3)
# print(' ')
# print('Fourier Series n = 10:', model_dict_10)
# print('n = 10 Fit Results:', fit_result_10)

# # Plot the result
# plt.plot(time_16min_a, y_square, color = 'orange', label = 'Period = 16 min, idealised outer data')
# plt.plot(time_16min_a, y_square_fundamental, color = 'gray', label = 'Fourier series n = 1, outer')
# plt.plot(time_16min_a, temp_16min_a, color = 'blue', label = 'Period = 16 min, inner data set a')
# plt.plot(time_16min_a, fit_1.model(x=time_16min_a, **fit_result_1.params).y, color = 'lime', label = 'Fourier series n = 1, inner')
# plt.plot(time_16min_a, fit_2.model(x=time_16min_a, **fit_result_2.params).y, color = 'cyan', label = 'Fourier series n = 2, inner')
# plt.plot(time_16min_a, fit_3.model(x=time_16min_a, **fit_result_3.params).y, color = 'red', label = 'Fourier series n = 3, inner')
# plt.plot(time_16min_a, fit_10.model(x=time_16min_a, **fit_result_10.params).y, color = 'violet', label = 'Fourier series n = 10, inner')
# plt.xlabel('Time (ds)')
# plt.ylabel('Temperature (°C)')
# plt.ylim(-20, 120)
# plt.title('Fourier Series Approximations for Period = 16 min, Data Set a')
# plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
# plt.savefig('task 2.7 16min a Fourier series fit.png', dpi = 1000)
# plt.show()


# # Find positions of peaks, amplitudes, and phase lags.

# peak_pos_outer, properties_outer = signal.find_peaks(y_square_fundamental, height = np.mean(temp_16min_a))
# # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html#scipy.signal.peak_widths

# # print('Positions of outer peaks = ', peak_pos_outer) # [625.0, 3125.0, 5624.0, 8124.0]
# # print('Heights of outer peaks = ', properties_outer) # {'peak_heights': array([114.9436979 , 114.94367905, 114.94366019, 114.94369161])}

# # Find peak positions and height:
# peak_pos_inner_1, properties_inner_1 = signal.find_peaks(fit_1.model(x=time_16min_a, **fit_result_1.params).y, height = np.mean(temp_16min_a))
# peak_pos_inner_2, properties_inner_2 = signal.find_peaks(fit_2.model(x=time_16min_a, **fit_result_2.params).y, height = np.mean(temp_16min_a))
# peak_pos_inner_3, properties_inner_3 = signal.find_peaks(fit_3.model(x=time_16min_a, **fit_result_3.params).y, height = np.mean(temp_16min_a))
# peak_pos_inner_10, properties_inner_10 = signal.find_peaks(fit_10.model(x=time_16min_a, **fit_result_10.params).y, height = np.mean(temp_16min_a))

# # Find amplitudes and phase lags
# amp_inner_1 = properties_inner_1['peak_heights'] - np.mean(temp_16min_a)
# amp_inner_2 = properties_inner_2['peak_heights'] - np.mean(temp_16min_a)
# amp_inner_3 = properties_inner_3['peak_heights'] - np.mean(temp_16min_a)
# amp_inner_10 = properties_inner_10['peak_heights'] - np.mean(temp_16min_a)

# print(peak_pos_inner_1)
# print(peak_pos_inner_2)
# print(peak_pos_inner_3)
# print(peak_pos_inner_10)
# print(peak_pos_outer)

# phase_lag_1 = (peak_pos_inner_1 - peak_pos_outer) * np.pi / 4800
# phase_lag_2 = (peak_pos_inner_2 - peak_pos_outer) * np.pi / 4800
# phase_lag_3 = (peak_pos_inner_3 - peak_pos_outer) * np.pi / 4800
# phase_lag_10 = (peak_pos_inner_10 - peak_pos_outer) * np.pi / 4800

# # Calculate mean amplitudes and phase lags for each Fourier fit:
# print(' ')
# print('Mean amplitude n = 1: ', np.mean(amp_inner_1), '+/-', (np.amax(amp_inner_1) - np.amin(amp_inner_1)) / 2)
# print('Mean amplitude n = 2: ', np.mean(amp_inner_2), '+/-', (np.amax(amp_inner_2) - np.amin(amp_inner_2)) / 2)
# print('Mean amplitude n = 3: ', np.mean(amp_inner_3), '+/-', (np.amax(amp_inner_3) - np.amin(amp_inner_3)) / 2)
# print('Mean amplitude n = 10: ', np.mean(amp_inner_10), '+/-', (np.amax(amp_inner_10) - np.amin(amp_inner_10)) / 2)

# print('Mean phase lag n = 1: ', np.mean(phase_lag_1), '+/-', (np.amax(phase_lag_1) - np.amin(phase_lag_1)) / 2)
# print('Mean phase lag n = 2: ', np.mean(phase_lag_2), '+/-', (np.amax(phase_lag_2) - np.amin(phase_lag_2)) / 2)
# print('Mean phase lag n = 3: ', np.mean(phase_lag_3), '+/-', (np.amax(phase_lag_3) - np.amin(phase_lag_3)) / 2)
# print('Mean phase lag n = 10: ', np.mean(phase_lag_10), '+/-', (np.amax(phase_lag_10) - np.amin(phase_lag_10)) / 2)

# #%%

# from uncertainties import ufloat
# import uncertainties.umath as umath

# task 2.7, 8 min a, calculating D:

# Convert everything to SI units - KEEP TEMPERATURES IN °C
# d_r = ufloat(7.87e-3, 0.05e-5)
# w_1 = ufloat(1.745075e-02, 1.224153e-06) # e-02 since units converted from decisecods to seconds. 
# w_2 = ufloat(1.744943e-02, 1.227824e-06)
# w_3 = ufloat(1.744801e-02, 6.651489e-07)
# w_10 = ufloat(1.744801e-02, 6.651489e-07)
# gamma_1 = ufloat(0.35220290, 0.00000004)
# gamma_2 = ufloat(0.35408232, 0.00000002)
# gamma_3 = ufloat(0.37059926, 0.00000008)
# gamma_10 = ufloat(0.37059926, 0.00000008)
# d_phase_1 = ufloat(2.4801, 0.0175)
# d_phase_2 = ufloat(2.4662, 0.0349)
# d_phase_3 = ufloat(2.5098, 0.0349)
# d_phase_10 = ufloat(2.5098, 0.0349)

# # Calculate thermal diffusivity values:
# D_tf_1 = w_1 * (d_r ** 2) / (2 * (umath.log(gamma_1) ** 2))
# D_pl_1 = w_1 * (d_r ** 2) / (2 * (d_phase_1 ** 2))
# D_tf_2 = w_2 * (d_r ** 2) / (2 * (umath.log(gamma_2) ** 2))
# D_pl_2 = w_2 * (d_r ** 2) / (2 * (d_phase_2 ** 2))
# D_tf_3 = w_3 * (d_r ** 2) / (2 * (umath.log(gamma_3) ** 2))
# D_pl_3 = w_3 * (d_r ** 2) / (2 * (d_phase_3 ** 2))
# D_tf_10 = w_10 * (d_r ** 2) / (2 * (umath.log(gamma_10) ** 2))
# D_pl_10 = w_10 * (d_r ** 2) / (2 * (d_phase_10 ** 2))

# print('D_tf for n = 1: ', D_tf_1)
# print('D_pl for n = 1: ', D_pl_1)
# print('D_tf for n = 2: ', D_tf_2)
# print('D_pl for n = 2: ', D_pl_2)
# print('D_tf for n = 3: ', D_tf_3)
# print('D_pl for n = 3: ', D_pl_3)
# print('D_tf for n = 10: ', D_tf_10)
# print('D_pl for n = 10: ', D_pl_10)


#%%

# Plot D vs. n

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([1, 2, 3])
# x_8 = np.array([1, 2, 3, 10])

# # D_TF:
# D_TF_1 = np.array([2.2452e-7, 2.2478e-7, 2.2514e-7])
# D_TF_1_unc = np.array([4e-11, 4e-11, 4e-11])
# D_TF_2 = np.array([1.7817e-7, 1.7720e-7, 1.7561e-7])
# D_TF_2_unc = np.array([6e-11, 6e-11, 6e-11])
# D_TF_4 = np.array([2.8886e-7, 2.9087e-7, 2.9980e-7])
# D_TF_4_unc = np.array([4e-11, 4e-11, 4e-11])
# D_TF_6 = np.array([4.9626e-7, 5.0132e-7, 5.4839e-7])
# D_TF_6_unc = np.array([7e-11, 7e-11, 7e-11])
# D_TF_8 = np.array([1.2442e-6, 1.3265e-6, 1.5567e-6, 1.6751e-6])
# D_TF_8_unc = np.array([2e-10, 2e-10, 2e-10, 2e-10])

# # D_PL:
# D_PL_1 = np.array([1.124e-7, 1.138e-7, 1.132e-7])
# D_PL_1_unc = np.array([9e-10, 9e-10, 9e-10])
# D_PL_2 = np.array([7.87e-8, 7.79e-8, 7.66e-8])
# D_PL_2_unc = np.array([9e-10, 9e-10, 9e-10])
# D_PL_4 = np.array([8.22e-8, 8.23e-8, 8.46e-8])
# D_PL_4_unc = np.array([5e-10, 12e-10, 11e-10])
# D_PL_6 = np.array([8.79e-8, 8.88e-8, 8.58e-8])
# D_PL_6_unc = np.array([12e-10, 25e-10, 24e-10])
# D_PL_8 = np.array([1.195e-7, 1.221e-7, 1.019e-7, 9.77e-8])
# D_PL_8_unc = np.array([15e-10, 28e-10, 23e-10, 16e-10])

# plt.errorbar(x, D_TF_1, yerr = D_TF_1_unc, fmt = '-x', label = '$D_{TF}$, 1 min')
# plt.errorbar(x, D_TF_2, yerr = D_TF_2_unc, fmt = '-x', label = '$D_{TF}$, 2 min')
# plt.errorbar(x, D_TF_4, yerr = D_TF_4_unc, fmt = '-x', label = '$D_{TF}$, 4 min')
# plt.errorbar(x, D_TF_6, yerr = D_TF_6_unc, fmt = '-x', label = '$D_{TF}$, 6 min')
# plt.errorbar(x_8, D_TF_8, yerr = D_TF_8_unc, fmt = '-x', label = '$D_{TF}$, 8 min')
# plt.errorbar(x, D_PL_1, yerr = D_PL_1_unc, fmt = '-x', label = '$D_{PL}$, 1 min')
# plt.errorbar(x, D_PL_2, yerr = D_PL_2_unc, fmt = '-x', label = '$D_{PL}$, 2 min')
# plt.errorbar(x, D_PL_4, yerr = D_PL_4_unc, fmt = '-x', label = '$D_{PL}$, 4 min')
# plt.errorbar(x, D_PL_6, yerr = D_PL_6_unc, fmt = '-x', label = '$D_{PL}$, 6 min')
# plt.errorbar(x_8, D_PL_8, yerr = D_PL_8_unc, fmt = '-x', label = '$D_{PL}$, 8 min')
# plt.title('Comparision of $D_{TF}$ and $D_{PL}$ for Various Periods')
# plt.xlabel('Fourier Series Order n')
# plt.ylabel('Thermal Diffusivity ($m^2$ $s^-1$)')
# plt.legend(loc = 'upper left')
# plt.savefig('task 2.7 D vs n.png', dpi = 1000)
# plt.show()

# %%

# Plot D vs. n

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.array([1, 2, 4, 6, 8])

# # D_TF:
# D_TF_1 = np.array([2.2452e-7, 1.7817e-7, 2.8886e-7, 4.9626e-7, 1.2442e-6])
# D_TF_1_unc = np.array([4e-11, 6e-11, 4e-11, 7e-11, 2e-10])
# D_TF_2 = np.array([2.4782e-7, 1.7720e-7, 2.9087e-7, 5.0132e-7, 1.3265e-6])
# D_TF_2_unc = np.array([4e-11, 6e-11, 4e-11, 7e-11, 2e-10])
# D_TF_3 = np.array([2.2514e-7, 1.7561e-7, 2.9980e-7, 5.4839e-7, 1.5567e-6])
# D_TF_3_unc = np.array([4e-11, 6e-11, 4e-11, 7e-11, 2e-10])

# # D_PL:
# D_PL_1 = np.array([1.124e-7, 7.87e-8, 8.22e-8, 8.79e-8, 1.195e-7])
# D_PL_1_unc = np.array([9e-10, 9e-10, 5e-10, 12e-10, 15e-10])
# D_PL_2 = np.array([1.138e-7, 7.79e-8, 8.23e-8, 8.88e-8, 1.221e-7])
# D_PL_2_unc = np.array([9e-10, 9e-10, 12e-10, 25e-10, 28e-10])
# D_PL_3 = np.array([1.132e-7, 7.66e-8, 8.46e-8, 8.58e-8, 1.019e-7])
# D_PL_3_unc = np.array([11e-10, 9e-10, 11e-10, 24e-10, 23e-10])

# plt.errorbar(x, D_TF_1, yerr = D_TF_1_unc, fmt = '-x', label = '$D_{TF}$, n = 1')
# plt.errorbar(x, D_TF_2, yerr = D_TF_2_unc, fmt = '-x', label = '$D_{TF}$, n = 2')
# plt.errorbar(x, D_TF_3, yerr = D_TF_3_unc, fmt = '-x', label = '$D_{TF}$, n = 3')

# plt.errorbar(x, D_PL_1, yerr = D_PL_1_unc, fmt = '-x', label = '$D_{PL}$, n = 1')
# plt.errorbar(x, D_PL_2, yerr = D_PL_2_unc, fmt = '-x', label = '$D_{PL}$, n = 2')
# plt.errorbar(x, D_PL_3, yerr = D_PL_3_unc, fmt = '-x', label = '$D_{PL}$, n = 3')

# plt.axhline(y = 1.3e-7, linestyle = '--', color = 'black', label = 'Reference D')

# plt.title('Comparision of $D_{TF}$ and $D_{PL}$ for Various Periods')
# plt.xlabel('Period τ (mins)')
# plt.ylabel('Thermal Diffusivity D ($m^2$ $s^-1$)')
# plt.legend(loc = 'upper left')
# plt.savefig('task 2.7 D vs period.png', dpi = 1000)
# plt.show()

# %%