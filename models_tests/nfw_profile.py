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
aaa = simpson(nfw_density(rr_from_ss), x=xx_int * u.kpc.to(u.cm))
print(aaa)


aaa = (c/128./np.pi/u.sr / u.micron
       * (1*u.eV)**2
       * (1e-10*u.GeV**-1)**2
       *1.11e22* u.GeV * u.cm ** -2
       ).to(u.nW*u.sr**-1*u.m**-2)
print(aaa)
