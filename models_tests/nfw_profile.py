import numpy as np
from scipy.integrate import simpson

import astropy.units as u
from astropy.constants import c
from astropy.constants import h as h_plank

def nfw_density(dd_gc, rho_sun=0.42, r_sun=8., r_s=21.):
    rho0 = rho_sun * (r_sun/r_s) * (1. + r_sun/r_s)**2.
    return rho0 / (dd_gc/r_s) / (1. + dd_gc/r_s)**2.
xx_int = np.geomspace(1e-10, 300, num=50000)
r_sun = 8.
rr_from_ss = np.sqrt(xx_int**2. + r_sun**2. - 2.* r_sun * xx_int
                     * np.cos(0*np.pi/180)*np.cos(180*np.pi/180))
aaa_042 = simpson(nfw_density(rr_from_ss), x=xx_int * u.kpc.to(u.cm))
print(aaa_042)

for i in [0.40, 0.42, 0.45]:
    aaa = simpson(nfw_density(rr_from_ss, rho_sun=i),
                  x=xx_int * u.kpc.to(u.cm))
    print(i, aaa, aaa/aaa_042, 14.5318*aaa/aaa_042)


print('\nSky patches')

coord_vector = [[180., 0., 0.],
                [345.41, 85.74, 10 * 10],
                [271.45, 28.41, 10 * 10],
                [66.27, -57.69, 3  * 10],
                [98.81, -62.03, 3  * 10],
                [20.89, 47.72, 28  * 10],
                [357.91, 55.25, 30 * 10],
                [301.11, 40.02, 29 * 10],
                [48.34, 30.86,  48 * 10],
                [349.46, 67.87, 99 * 10],
                [33.78, -62.32, 30 * 10],
                [54.33, -60.76, 15 * 10],
                [61.88, 38.45,  30 * 10],
                [73.08, -76.15, 63 * 30],
                [350.96, -65.06, 104*30],
                [275.02, -61.69, 15*30],
                [92.71, -59.91, 3*30],
                [98.06, -60.23, 3*30],
                [59.51, 61.34, 3*30],
                [57.26, 60.26, 3*30]
                ]

xx_int = np.geomspace(1e-10, 300, num=50000)
Dfact = []
Dfact_mean = 0.
coord_vector = np.array(coord_vector, dtype=float)
weights = sum(coord_vector[:, 2])

for ni, coord in enumerate(coord_vector):
    rr_from_ss = np.sqrt(
        xx_int ** 2. + r_sun ** 2. - 2. * r_sun * xx_int
        * np.cos(coord[0] * np.pi / 180)
        * np.cos(coord[1] * np.pi / 180))

    Dfact.append(simpson(nfw_density(rr_from_ss),
                         x=xx_int * u.kpc.to(u.cm)))
    Dfact_mean += (simpson(nfw_density(rr_from_ss),
                           x=xx_int * u.kpc.to(u.cm)) * coord[2])
    print(coord, Dfact[ni])

aa = np.where(max(Dfact) == Dfact)[0][0]
print('max')
print(coord_vector[aa], Dfact[aa])

aa = np.where(min(Dfact) == Dfact)[0][0]
print('min')
print(coord_vector[aa], Dfact[aa])
print(Dfact[np.where(max(Dfact) == Dfact)[0][0]]
      /Dfact[np.where(min(Dfact) == Dfact)[0][0]])

print('Dfactor weighted: ', Dfact_mean/weights)
