import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, \
    interpn, RegularGridInterpolator
from fast_interp import interp2d, interp3d

from astropy.io import fits
from astropy import units as u
from astropy.constants import c, L_sun
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebl_codes import dust_absorption_models as dust_abs

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=5)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

os.chdir('..')
print(os.listdir())
f_tir = 3e9
chary = fits.open('data/ssp_synthetic_spectra/chary2001/chary_elbaz.fits')

ir_wv = chary[1].data.field('LAMBDA')[0]
ir_freq = c.value / ir_wv * 1e6

ir_lum = (chary[1].data.field('NULNUINLSUN')[0]
          * (L_sun.to(u.erg / u.s).value / f_tir))

# Cap the dust reemisison to the wavelength where there is
# proper reemission, not the whole possible spectrum
# ir_lum[ir_wv < 3.5, :] = 1e-43

l_tir = np.log(10) * simpson(
    ir_lum[::-1], x=np.log10(ir_freq)[::-1], axis=0)

sort_order = np.argsort(l_tir)

ir_lum *= (1 / ir_freq[:, np.newaxis])
ir_lum[ir_lum < 1e-43] = 1e-43

l_tir = np.log10(np.log(10) * simpson(
                ir_lum[::-1], x=np.log10(ir_freq)[::-1], axis=0))
# ----------------------------------------------------

data = fits.open('outputs/dust_reem/Z.fits')
aaa = data[1].data

yyy = np.column_stack((
    aaa['nuLnu[Z=6.99103]'],
    aaa['nuLnu[Z=6.99103]'], aaa['nuLnu[Z=7.99103]'],
    aaa['nuLnu[Z=8.29205999]'], aaa['nuLnu[Z=8.69]'],
    aaa['nuLnu[Z=9.08794001]']))
yyy = (yyy * (L_sun.to(u.erg / u.s)).value
       * (aaa['wavelength'] * 1e-9 / c.value)[:, np.newaxis]
       )

yyy = np.log10(yyy)

# ---------------------------------------------------------------------
plt.figure(figsize=(10, 8))

metalls = [6.99103, 7.99103, 8.29205999, 8.69, 9.08794001]
metalls_Ztot = 10**(np.array(metalls) - 12)/4.89778819e-04*0.02
metalls_Ztot = [r'$\leq$'+str(0.0004), 0.004,  0.008,  0.02,
                r'$\geq$'+str(0.05)]
print(metalls_Ztot)
for iii in range(len(metalls)):
    plt.plot(aaa['wavelength']*1e-3, yyy[:, iii+1],
             label=metalls_Ztot[iii],
             c=plt.cm.CMRmap(iii / len(metalls_Ztot)))

plt.legend(title='Z')

plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel(r'log$_{10}$(L$_{\nu}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{Hz}^{-1}$ M$_{\odot}^{-1}$])')

plt.xscale('log')
# plt.yscale('log')

plt.xlim(1, 1e4)
# plt.ylim(16, 25.5)

plt.savefig('outputs/dust_reem/bosaZ_plot.pdf',
            bbox_inches='tight')
plt.savefig('outputs/dust_reem/bosaZ_plot.png',
            bbox_inches='tight')
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.plot(ir_wv, np.log10(ir_lum[:, 0]),
         label=r'$\leq$%.2f' % l_tir[0], c=plt.cm.CMRmap(0/8.))
for ni, i in enumerate(range(15, np.shape(ir_lum)[1]-1, 15)):
    plt.plot(ir_wv, np.log10(ir_lum[:, i]),
             label='%.2f' % l_tir[i],
             c=plt.cm.CMRmap((ni+1) / 8.)
             )
plt.plot(ir_wv, np.log10(ir_lum[:, -1]),
         label=r'$\geq$%.2f' % l_tir[-1], c=plt.cm.CMRmap(7/8.))

plt.xlabel('Wavelength ($\mu$m)')
plt.ylabel(r'log$_{10}(L_{\nu}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{Hz}^{-1}$ M$_{\odot}^{-1}$])')

plt.legend(ncols=1, loc=1, title=r'log$_{10}\left(L_{\nu}\right)$',
           framealpha=1)

plt.xscale('log')

plt.xlim(0.1, 3e5)
plt.ylim(16, 25.5)

plt.savefig('outputs/dust_reem/chary2001_plot.pdf',
            bbox_inches='tight')
plt.savefig('outputs/dust_reem/chary2001_plot.png',
            bbox_inches='tight')

plt.show()