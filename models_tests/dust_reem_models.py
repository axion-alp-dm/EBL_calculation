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
from astropy.constants import c, L_sun, k_B
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebl_codes import dust_absorption_models as dust_abs

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=22)
plt.rc('axes', titlesize=22)
plt.rc('axes', labelsize=22)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=10)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=7)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

os.chdir('..')
print(os.listdir())

plt.figure()
lambda_array = np.geomspace(1., 1e3, num=100)
frew_array = c.value / lambda_array * 1e6 * u.Hz
print(frew_array)

def bb_plank(T):
    xx = h_plank*frew_array/k_B/T/u.K
    return (15. / np.pi**4. / frew_array
            * xx**4. / (np.exp(xx) - 1.))
yyy = bb_plank(50.) + bb_plank(150.)
plt.loglog(lambda_array, yyy)
print(simpson(yyy, x=frew_array))
print(bb_plank(50)[-1])
# plt.show()
f_tir = 3e9
chary = fits.open('data/ssp_synthetic_spectra/chary2001/chary_elbaz.fits')

ir_wv = chary[1].data.field('LAMBDA')[0]
ir_freq = c.value / ir_wv * 1e6

ir_lum = (np.log10(chary[1].data.field('NULNUINLSUN')[0])
          - np.log10(f_tir))

aaa = np.zeros((np.shape(ir_lum)[0], np.shape(ir_lum)[1] + 2))
aaa[:, 1:-1] = ir_lum
aaa[:, 0] = ir_lum[:, 0] - 3.
aaa[:, -1] = ir_lum[:, -1] + 3.

ir_lum = aaa

# Cap the dust reemisison to the wavelength where there is
# proper reemission, not the whole possible spectrum
# ir_lum[ir_wv < 3.5, :] = 1e-43

l_tir = np.log(10) * simpson(
    10**ir_lum[::-1], x=np.log10(ir_freq)[::-1], axis=0)
l_tir *= L_sun.to(u.erg / u.s).value

sort_order = np.argsort(l_tir)

ir_lum -= np.log10(ir_freq[:, np.newaxis])
ir_lum += np.log10(L_sun.to(u.erg / u.s).value)
ir_lum[ir_lum < 1e-43] = 1e-43

l_tir = np.log10(l_tir)
print('l_tir', l_tir)
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
fig, ax1 = plt.subplots(figsize=(7.5, 5))

metalls = [6.99103, 7.99103, 8.29205999, 8.69, 9.08794001]
# metalls_Ztot = 10**(np.array(metalls) - 12)/4.89778819e-04*0.02
# metalls_Ztot = [r'$\leq$'+str(0.0004), 0.004,  0.008,  0.02,
#                 r'$\geq$'+str(0.05)]
metalls_Ztot = [0.0004, 0.004,  0.008,  0.02, 0.05]
print(metalls)
print(metalls_Ztot)
print(metalls_Ztot)
for iii in range(len(metalls)):
    plt.plot(aaa['wavelength']*1e-3, yyy[:, iii+1],
             label=metalls_Ztot[iii],
             c=plt.cm.CMRmap(iii / len(metalls_Ztot)))

handles, labels = ax1.get_legend_handles_labels()
leg1 = plt.legend(handles[::-1], labels[::-1], title='Z')

plt.xlabel('Wavelength (µm)')
plt.ylabel(r'log$_{10}$(L$_{\nu}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{Hz}^{-1}$ M$_{\odot}^{-1}$])')

plt.xscale('log')
# plt.yscale('log')

plt.xlim(1, 1e4)
# plt.ylim(16, 25.5)

plt.yticks(np.arange(15, 22, 1),
           labels=['15', '', '17', '', '19', '', '21'])

plt.savefig('outputs/dust_reem/bosaZ_plot.pdf',
            bbox_inches='tight')
plt.savefig('outputs/dust_reem/bosaZ_plot.png',
            bbox_inches='tight', dpi=500)
# ---------------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(8.5, 6))

plt.plot(ir_wv, ir_lum[:, 0], ls='--',
         label=r'%.1f' % l_tir[0], c=plt.cm.CMRmap(0/6.))
for ni, i in enumerate(range(1, np.shape(ir_lum)[1]-2, 20)[:-1]):
    plt.plot(ir_wv, ir_lum[:, i],
             label='%.1f' % l_tir[i],
             c=plt.cm.CMRmap(ni / 6.)
             )
# plt.plot(ir_wv, ir_lum[:, -3],
#          label=r'%.1f' % l_tir[-3], c=plt.cm.CMRmap(7/8.))
plt.plot(ir_wv, ir_lum[:, -2],
         label=r'%.1f' % l_tir[-2], c=plt.cm.CMRmap(5/6.))
plt.plot(ir_wv, ir_lum[:, -1], ls='--',
         label=r'%.1f' % l_tir[-1], c=plt.cm.CMRmap(5/6.))

plt.xlabel('Wavelength (µm)')
plt.ylabel(r'log$_{10}(L_{\nu}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{Hz}^{-1}$ M$_{\odot}^{-1}$])')

handles, labels = ax1.get_legend_handles_labels()
leg1 = plt.legend(handles[::-1], labels[::-1],
                  ncol=1, loc=1,
                  title=r'log$_{10}\left(L_\mathrm{TIR}\right)$',
                  framealpha=1, fontsize=16, title_fontsize=17)

# plt.text(x=0.2, y=26.5,
#          s=r'$f_\mathrm{TIR} = 3\times10^9$',
#          backgroundcolor='white',
#          bbox=dict(facecolor='w', alpha=1, edgecolor='gainsboro'),
#          fontsize=20)

# plt.text(x=0.15, y=26.5,
#          s=r'$\lambda_\mathrm{TIR, cut}=5.5\,$µm',
#          backgroundcolor='white',
#          bbox=dict(facecolor='w', alpha=1, edgecolor='lightgrey'),
#          fontsize=16)
#
# plt.text(x=20, y=13.5,
#          s=r'$f_\mathrm{TIR} = 3\times10^9$',
#          backgroundcolor='white',
#          bbox=dict(facecolor='w', alpha=1, edgecolor='gainsboro'),
#          fontsize=18)

plt.text(x=0.8, y=26,
         s=r'$f_\mathrm{TIR} = 3\times10^9$'
         '\n'
         r'$\lambda_\mathrm{TIR, cut}=5.5\,$µm'
         ,
         backgroundcolor='white',
         bbox=dict(facecolor='w', alpha=1, edgecolor='lightgrey'),
         fontsize=16,
         horizontalalignment='center',
)

plt.xscale('log')

plt.xlim(0.1, 3e5)
# plt.ylim(16, 25.5)

plt.axvspan(1e-2, 5.5, color='grey', alpha=0.1, zorder=0, ls='')

aaa = np.arange(13, 29, 1)
bbb = []

for ni in range(len(aaa)):
    if ni % 2 == 0:
        bbb.append(str(int(aaa[ni])))
    else:
        bbb.append('')

plt.yticks(aaa, labels=bbb)

plt.savefig('outputs/dust_reem/chary2001_plot.pdf',
            bbox_inches='tight')
plt.savefig('outputs/dust_reem/chary2001_plot.png',
            bbox_inches='tight', dpi=500)

plt.show()