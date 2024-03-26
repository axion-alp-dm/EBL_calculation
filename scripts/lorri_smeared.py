# IMPORTS -----------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebltable.ebl_from_model import EBL

all_size = 34
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)
# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

direct_name = str('outputs/figures/paper/')
print(direct_name)

lorri_trans = np.loadtxt('data/lorri_transmitance.txt')
lorri_trans[:, 0] = lorri_trans[:, 0] * 1e-3
lorri_trans[:, 1] = lorri_trans[:, 1] * 1e-2  #/ max(lorri_trans[:, 1])
lorri_spline = UnivariateSpline(lorri_trans[:, 0], lorri_trans[:, 1],
                                k=1, s=0, ext=1)

waves_ebl = np.geomspace(3e-1, 1, num=500)

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)

h=0.7
omegaM=0.3
omegaBar=0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                                    Ob0=omegaBar, Tcmb0=2.7255)


def avg_lmbd(lambda_array, f_nu, lorri_pass=lorri_spline):
    yyy = lorri_pass(lambda_array) * f_nu(lambda_array)
    yyy_num = simpson(y=yyy, x=lambda_array)
    yyy_den = simpson(y=yyy / lambda_array, x=lambda_array)
    return yyy_num / yyy_den


def avg_lmbd_v2(f_nu,
                lorri_pass=lorri_trans[:, 1],
                lambda_array=lorri_trans[:, 0]):
    yyy = lorri_pass * f_nu(lambda_array)
    yyy_num = simpson(y=yyy, x=lambda_array)
    yyy_den = simpson(y=yyy / lambda_array, x=lambda_array)
    return yyy_num / yyy_den

def axion_contr(lmbd, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = c.value / lmbd * 1e6 * u.s**-1

    z_star = (axion_mass / (2. * h_plank.to(u.eV * u.s) * freq)
              - 1.)

    ebl_axion_cube = (
            ((cosmo.Odm(0.) * cosmo.critical_density0
              * c ** 3. / (64. * np.pi * u.sr)
              * axion_gayy ** 2. * axion_mass ** 2.
              * freq
              / cosmo.H(z_star)
              ).to(u.nW * u.m ** -2 * u.sr ** -1)
             ).value
            * (z_star > 0.))
    return ebl_axion_cube


# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 8))
fig, ax = plt.subplots()
plt.tight_layout()
fig2, ax2 = plt.subplots()
plt.tight_layout()
axion_mass_array = np.linspace(3.5, 5., num=40)

mean_lambda = np.zeros(len(axion_mass_array))
mean_lambda_only = np.zeros(len(axion_mass_array))

for i in range(len(mean_lambda)):
    total_yy = (axion_contr(waves_ebl,
                            axion_mass_array[i], 1e-10)
                + spline_cuba(waves_ebl)
                )

    ax.loglog(waves_ebl, total_yy)
    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda[i] = avg_lmbd_v2(f_nu=total_spline)

    total_yy = (axion_contr(waves_ebl,
                            axion_mass_array[i], 1e-10))
    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_only[i] = avg_lmbd_v2(f_nu=total_spline)

ax2.plot(axion_mass_array, mean_lambda, marker='.', label='CUBA + axion')
ax2.plot(axion_mass_array, mean_lambda_only, marker='.', label='Axion')
ax.plot(waves_ebl, spline_cuba(waves_ebl), 'k')

ax2.legend()
ax2.set_ylim(0.4, 1.)
ax2.axhline(0.608, ls='--')
ax2.set_xlabel('axion mass (eV)')
ax2.set_ylabel(r'<$\lambda$>')

plt.figure()
plt.tight_layout()
plt.plot(lorri_trans[:, 0], lorri_trans[:, 1], marker='.')
plt.xlabel('wavelenght (microns)')
plt.ylabel(r'responsivity')

plt.show()
