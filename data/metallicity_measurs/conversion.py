import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/home/porrassa/Downloads/metall_conv.txt')

plt.figure()
plt.scatter(data[:, 0], data[:, 1])

aa = np.polyfit(data[:, 0], data[:, 1], deg=1)
print(aa)

xx_arr = np.linspace(-1.1, 0.1)
plt.plot(xx_arr, aa[1] + aa[0]*xx_arr)

arr_metall = np.array([0.05, 0.02, 0.008, 0.004, 0.0004])
print(8.69 + np.log10(arr_metall/0.02))

plt.show()