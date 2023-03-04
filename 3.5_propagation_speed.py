import numpy as np
import matplotlib.pyplot as plt

t1, V1 = np.loadtxt("Data/Extra/SAVE51/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
t2, V2 = np.loadtxt("Data/Extra/SAVE52/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
t1_max = t1[np.argmax(V1)]
t2_max = t2[np.argmax(V2)]
print(4/(t2_max - t1_max))

plt.plot(t1, V1, label="V1")
plt.plot(t2, V2, label="V2")
plt.show()

t1, V1 = np.loadtxt("Data/Extra/SAVE54/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
t1, V1 = np.loadtxt("Data/Extra/SAVE55/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
t1_max = t1[np.argmax(V1)]
t2_max = t2[np.argmax(V2)]
print(4/(t2_max - t1_max))

plt.plot(t1, V1, label="V1")
plt.plot(t2, V2, label="V2")
plt.show()

t1, V1 = np.loadtxt("Data/Extra0/SAVE15/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
t1, V1 = np.loadtxt("Data/Extra0/SAVE16/WFM.CSV", delimiter=",", skiprows=1, unpack=True)
t1_max = t1[np.argmax(V1)]
t2_max = t2[np.argmax(V2)]
print(4/(t2_max - t1_max))

plt.plot(t1, V1, label="V1")
plt.plot(t2, V2, label="V2")
plt.show()

