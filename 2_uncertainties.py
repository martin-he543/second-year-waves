import uncertainties as unc
from uncertainties import ufloat
import uncertainties.unumpy as unp
import numpy as np

x = ufloat(2.50, 0.05)
y = ufloat(20.57, 0.01)

a = ufloat(50, 0.01)
b = ufloat(18.9, 0.1)

print(x-y)
print((np.pi*(y-x)**2)/(120*12**2))
print((np.pi*(y-x)**2)/(2*0.972861**2))
