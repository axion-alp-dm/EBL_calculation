import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

aaa = np.loadtxt('../Swire_library/Ell13_template_norm.sed')

plt.figure()
position5500 = abs(aaa[:,0]-5500).argmin()
position2200 = abs(aaa[:,0]-2200).argmin()
print(aaa[position5500,1])
aaa[:, 1] *= aaa[position5500, 1] / aaa[position2200, 1]
old_spectrum_spline = UnivariateSpline(aaa[:, 0], aaa[:, 1], k=1)

plt.plot(aaa[:,0], aaa[:,1])
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'Wavelength [A]')
plt.ylabel(r'lambda * F_lambda [erg cm-2 s-1 A-1]')

plt.show()
'''
aaa = np.array([3.98, 16., 4.0, 15., 4.1,
3.88, 15.8, 3.6, 15.4, 3.7,
3.78, 12.9, 3.3, 12.8, 3.4,
3.68, 10.6, 3.7, 10.9, 3.9,
3.58, 14.0, 3.0, 13.9, 3.1,
3.48, 12.7, 3.1, 12.6, 3.3,
3.38, 12.9, 3.0, 12.7, 3.2,
3.28, 11.6, 3.1, 11.6, 3.3,
3.17, 15.1, 3.0, 14.9, 3.2,
3.07, 18.5, 3.0, 18.2, 3.3,
2.98, 17.1, 3.2, 16.6, 3.4,
2.88, 19.0, 3.5, 18.2, 3.7,
2.54, 22.3, 4.2, 20.9, 4.4,
2.44, 20.5, 4.6, 19.2 ,4.9,
2.34, 23.6, 4.9, 21.7 ,5.1,
2.24, 29.2, 5.3, 26.8, 5.6,
2.14, 35.4, 5.9, 32.2, 6.2,
2.03, 38.0, 6.8, 34.4, 7.2,
1.93, 40.1, 7.8, 36.3, 8.2,
1.83, 42.4, 8.7, 42.8, 9.0,
1.73 ,53.1, 10.1, 47.3, 10.6,
1.63, 58.3, 11.8, 51.4, 12.2,
1.53, 59.9, 12.7, 52.8, 13.3,
1.43, 58.1, 13.1, 51.2, 13.7])
print(np.shape(aaa)[0]/5.)
aaa = np.reshape(aaa, (24, 5))
print(aaa)
print(aaa[:,0])
print(aaa[:,1])
print(aaa[:,2])
'''
