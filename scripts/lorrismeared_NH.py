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
from data.cb_measurs.import_cb_measurs import import_cb_data

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

direct_name = str('lorri_smeared_host/')
print(direct_name)

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

lorri_trans = np.loadtxt('data/lorri_transmitance.txt')
lorri_trans[:, 0] = lorri_trans[:, 0] * 1e-3
lorri_trans[:, 1] = lorri_trans[:, 1] * 1e-2  # / max(lorri_trans[:, 1])
lorri_spline = UnivariateSpline(lorri_trans[:, 0], lorri_trans[:, 1],
                                k=1, s=0, ext=1)

waves_ebl = np.geomspace(0.1, 1.3, num=20000)

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)

h = 0.7
omegaM = 0.3
omegaBar = 0.0222 / 0.7 ** 2.
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

    freq = c.value / lmbd * 1e6 * u.s ** -1

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


D_factor = 2.20656e22 * u.GeV * u.cm ** -2


def host_axion_contr(xx, mass, gay, v_dispersion=220.):
    sigma = (2. * 2.48 / mass
             * (v_dispersion * u.km * u.s ** -1 / c).to(1))
    nuInu_values = (
            14.53 * mass ** 3. * (gay*1e10) ** 2.
            * (D_factor / (1.11e22 * u.GeV * u.cm ** -2)).to(1)
            * np.exp(-0.5 * ((xx - 2.48/mass) / sigma) ** 2.))

    return nuInu_values


axion_mass_array = np.geomspace(2., 9., num=750)
axion_gayy_array = np.geomspace(1e-11, 1e-7, num=700)
values_gay_array_NH = np.zeros(
    (len(axion_mass_array), len(axion_gayy_array)))

np.save('outputs/' + direct_name + '/axion_mass', axion_mass_array)
np.save('outputs/' + direct_name + '/axion_gayy', axion_gayy_array)

for na, aa in enumerate(axion_mass_array):
    if na % 25 == 0:
        print(na)
    # if na > 50:
    #     break

    for nb, bb in enumerate(axion_gayy_array):
        total_yy = (host_axion_contr(waves_ebl, aa, bb)
                    + spline_cuba(waves_ebl)
                    )
        total_spline = UnivariateSpline(
            waves_ebl, total_yy, s=0, k=1)
        mean_lambda = avg_lmbd_v2(f_nu=total_spline)

        values_gay_array_NH[na, nb] += (
                ((21.98 - total_spline(mean_lambda)) / 1.23) ** 2.)

np.save('outputs/' + direct_name + '/CUBA' + 'lorri_smeared',
        values_gay_array_NH)

fig, ax1 = plt.subplots()
upper_lims_all, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=1300.,
    ax1=ax1, plot_measurs=True)
total_yy = (host_axion_contr(waves_ebl, 3.837, 2e-11)
                    + spline_cuba(waves_ebl))
total_spline = UnivariateSpline(
            waves_ebl, total_yy, s=0, k=1)
mean_lambda = avg_lmbd_v2(f_nu=total_spline)
print(mean_lambda)
plt.axvline(mean_lambda, c='k')
plt.axhline(21.98, c='b')
plt.axhline(21.98+1.23, alpha=0.5, c='b')
plt.axhline(21.98-1.23, alpha=0.5, c='b')
plt.loglog(waves_ebl, total_yy, marker='x')
plt.xlim(0.3, 1.)
plt.show()


plt.figure()
plt.contour(axion_mass_array, axion_gayy_array,
            values_gay_array_NH.T - np.min(values_gay_array_NH),
            levels=[5.99], origin='lower',
            colors='cyan', zorder=1e10, linewidths=10, alpha=0.9)
nh_contours = plt.contour(axion_mass_array, axion_gayy_array,
                          values_gay_array_NH.T - np.min(values_gay_array_NH),
                          levels=[2.30], origin='lower',
                          colors='r', zorder=1e10, linewidths=4, alpha=0.9)
plt.xscale('log')
plt.yscale('log')
# plt.show()

# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 8))
fig, ax = plt.subplots()
plt.tight_layout()
fig2, ax2 = plt.subplots()
plt.tight_layout()
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
