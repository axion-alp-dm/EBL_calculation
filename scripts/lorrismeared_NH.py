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

direct_name = str('lorri_smeared_new_total_220_value1116_zoom/')
print(direct_name)

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

lorri_trans = np.loadtxt('data/lorri_qe_v2.txt')
lorri_trans[:, 0] = lorri_trans[:, 0] * 1e-3
lorri_trans[:, 1] = lorri_trans[:, 1]  # * 1e-2  # / max(lorri_trans[:, 1])
spline_lorri = UnivariateSpline(lorri_trans[:, 0], lorri_trans[:, 1],
                                s=0, k=1, ext=1)

waves_ebl = np.geomspace(0.1, 1.3, num=20000)

pivot_vw = (simpson(lorri_trans[:, 1] * lorri_trans[:, 0],
                    x=lorri_trans[:, 0])
            / simpson(lorri_trans[:, 1] / lorri_trans[:, 0],
                      x=lorri_trans[:, 0])
            ) ** 0.5
print('pivot wv: ', pivot_vw)

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


def avg_flux(f_nu, trans_spline, lambda_array):
    yyy = trans_spline(lambda_array) / lambda_array
    yyy_num = simpson(y=yyy * f_nu(lambda_array), x=lambda_array)
    yyy_den = simpson(y=yyy, x=lambda_array)
    return yyy_num / yyy_den


def avg_lmbd_v1(f_nu, trans_spline, lambda_array):
    yyy = trans_spline(lambda_array) * f_nu(lambda_array)
    yyy_num = simpson(y=yyy, x=lambda_array)
    yyy_den = simpson(y=yyy / lambda_array, x=lambda_array)
    return yyy_num / yyy_den


def cosmic_axion_contr(lmbd, mass, gayy):
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
            14.53 * mass ** 3. * (gay * 1e10) ** 2.
            * (D_factor / (1.11e22 * u.GeV * u.cm ** -2)).to(1)
            * (v_dispersion / 220.) ** -1
            * np.exp(-0.5 * ((xx - 2.48 / mass) / sigma) ** 2.))

    return nuInu_values


mean_lmbd_cuba = avg_lmbd_v1(
    f_nu=UnivariateSpline(
        waves_ebl,
        nuInu['cuba'] * waves_ebl * 1e-6 / c.value,
        s=0, k=1),
    trans_spline=spline_lorri,
    lambda_array=waves_ebl)

print('pivot only cuba: ', mean_lmbd_cuba)
plt.figure()
plt.plot(waves_ebl, spline_lorri(waves_ebl))
plt.plot(lorri_trans[:, 0], spline_lorri(lorri_trans[:, 0]))

print('avg flux of cuba: ',
      avg_flux(
          f_nu=UnivariateSpline(
              waves_ebl,
              nuInu['cuba'] * pivot_vw * 1e-6 / c.value,
              s=0, k=1),
          trans_spline=spline_lorri,
          lambda_array=waves_ebl)
      * c.value / mean_lmbd_cuba * 1e6)

axion_mass_array = np.linspace(2, 9, num=10000)
axion_gayy_array = np.geomspace(1.e-11, 1e-7, num=500)
# axion_mass_array = np.geomspace(8, 1e7, num=600)
# axion_gayy_array = np.geomspace(2.e-12, 2e-10, num=200)

values_gay_array_NH = np.zeros(
    (len(axion_mass_array), len(axion_gayy_array)))

np.save('outputs/' + direct_name + '/axion_mass', axion_mass_array)
np.save('outputs/' + direct_name + '/axion_gayy', axion_gayy_array)
fig2, ax2 = plt.subplots()
fig1, ax1 = plt.subplots()
upper_lims_all, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax1, plot_measurs=True)
fig3, ax3 = plt.subplots(figsize=(12, 8))
ax4 = ax3.twinx()
import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax3, plot_measurs=True)

for na, aa in enumerate(axion_mass_array):
    if na % 25 == 0:
        print(na)

    for nb, bb in enumerate(axion_gayy_array):
        total_yy = (
                host_axion_contr(waves_ebl, aa, bb,
                                 v_dispersion=220)
                + cosmic_axion_contr(waves_ebl, aa, bb)
                + spline_cuba(waves_ebl)
        )
        intensity_points = total_yy * waves_ebl * 1e-6 / c.value
        total_spline = UnivariateSpline(
            waves_ebl, intensity_points, s=0, k=1)
        mean_lambda = avg_lmbd_v1(f_nu=total_spline,
                                  trans_spline=spline_lorri,
                                  lambda_array=waves_ebl)
        mean_flux = avg_flux(f_nu=total_spline,
                             trans_spline=spline_lorri,
                             lambda_array=waves_ebl
                             ) * c.value / mean_lambda * 1e6

        values_gay_array_NH[na, nb] += (
                ((11.16 * pivot_vw / mean_lambda
                  - mean_flux
                  ) / (1.65 * pivot_vw / mean_lambda)
                 ) ** 2.)

        # if na % 1 == 0 and nb % 1 == 0:
        # print('%.4f  %.4e  %.7e %.6f'
        #       % (aa, bb, mean_flux, mean_lambda))

        # plt.figure(fig1)
        # next_c = next(ax1._get_lines.prop_cycler)['color']
        # plt.axvline(mean_lambda, c=next_c, ls='--', alpha=0.7)
        # plt.loglog(waves_ebl, total_yy, c=next_c,
        #            label=aa)
        #
        # plt.figure(fig3)
        # next_c = next(ax3._get_lines.prop_cycler)['color']
        # ax3.axvline(mean_lambda, c=next_c, ls='--', alpha=0.7, lw=2)
        # ax3.plot(waves_ebl, total_yy, c=next_c,
        #            label=aa, lw=2)
        #
        # plt.figure(fig2)
        # plt.loglog(waves_ebl,
        #            total_spline(waves_ebl)
        #            * spline_lorri(waves_ebl) / waves_ebl,
        #            ls='-', c=next_c)

plt.figure(fig1)
plt.xlim(0.1, 1.5)
# plt.legend()
plt.plot(lorri_trans[:, 0], lorri_trans[:, 1], marker='.')

plt.figure(fig3)
plt.xlim(0.3, 1.)
# ax3.legend()
ax4.plot(lorri_trans[:, 0], lorri_trans[:, 1], lw=2, c='k')
ax3.set_yscale('log')
ax4.set_yscale('linear')
ax4.set_ylim(0., 0.7)
ax3.set_ylim(2., 1300)
ax3.axvline(0.608, c='k', alpha=0.7, lw=2)
ax3.set_xticks(ticks=[0.3, 0.4, 0.6, 1.],
               labels=['0.3', '0.4', '0.6', '1.'])
ax4.tick_params(axis='y', labelsize=26)
ax3.tick_params(axis='x', pad=10)

ax4.set_ylabel('Quantum efficiency of LORRI', fontsize=26, labelpad=10)
ax3.set_xlabel(r'Wavelength ($\mu$m)', fontsize=34)

# plt.savefig('outputs/figures_paper/axiondecay.pdf',
#             bbox_inches='tight')
# plt.savefig('outputs/figures_paper/axiondecay.png',
#             bbox_inches='tight')

plt.figure(fig2)

plt.loglog(lorri_trans[:, 0],
           lorri_trans[:, 1] / lorri_trans[:, 0],
           ls='--', c='k')
plt.xlim(0.1, 1.5)
plt.legend()
np.save('outputs/' + direct_name + '/CUBA' + 'lorri_smeared',
        values_gay_array_NH)
# plt.show()

fig, ax1 = plt.subplots()
import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=1300.,
    ax1=ax1, plot_measurs=True)
total_yy = (host_axion_contr(waves_ebl, 3.837, 2e-11)
            + spline_cuba(waves_ebl))
total_spline = UnivariateSpline(
    waves_ebl, total_yy * waves_ebl * 1e-6 / c.value, s=0, k=1)
mean_lambda = avg_lmbd_v1(f_nu=total_spline,
                          trans_spline=spline_lorri,
                          lambda_array=waves_ebl)
mean_flux = avg_flux(f_nu=total_spline,
                     trans_spline=spline_lorri,
                     lambda_array=waves_ebl)
baseline_flux = simpson(y=lorri_trans[:, 1] / lorri_trans[:, 0],
                        x=lorri_trans[:, 0])
print(baseline_flux)
print(mean_lambda, mean_flux * c.value / mean_lambda * 1e6)
print(baseline_flux * mean_flux * c.value / mean_lambda * 1e6,
      simpson(y=total_spline(lorri_trans[:, 0]) * lorri_trans[:, 1]
                / lorri_trans[:, 0],
              x=lorri_trans[:, 0]) * c.value / mean_lambda * 1e6,
      )
print(mean_flux, 21.98 * pivot_vw * 1e-6 / c.value,
      (21.98 - 1.23) * pivot_vw * 1e-6 / c.value,
      (21.98 + 1.23) * pivot_vw * 1e-6 / c.value)
plt.axvline(mean_lambda, c='k')
plt.axhline(21.98, c='b')
plt.axhline(21.98 + 1.23, alpha=0.5, c='b')
plt.axhline(21.98 - 1.23, alpha=0.5, c='b')
plt.loglog(waves_ebl, total_yy, marker='x')
plt.xlim(0.3, 1.)
# plt.show()

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
plt.show()

# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 8))
fig, ax = plt.subplots()
plt.tight_layout()
fig2, ax2 = plt.subplots(figsize=(10, 8))
plt.tight_layout()
mean_lambda_host = np.zeros(len(axion_mass_array))
mean_lambda_cosmic = np.zeros(len(axion_mass_array))
mean_lambda_total = np.zeros(len(axion_mass_array))
mean_lambda_only = np.zeros(len(axion_mass_array))
for i in range(len(mean_lambda_cosmic)):
    total_yy = (cosmic_axion_contr(waves_ebl,
                            axion_mass_array[i], 1e-10)
                + spline_cuba(waves_ebl)
                ) * waves_ebl * 1e-6 / c.value

    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_cosmic[i] = avg_lmbd_v1(f_nu=total_spline,
                                 trans_spline=spline_lorri,
                                 lambda_array=waves_ebl)
    #-----------------
    total_yy = (host_axion_contr(waves_ebl, axion_mass_array[i], 1e-10,
                                 v_dispersion=220)
                + spline_cuba(waves_ebl)
                ) * waves_ebl * 1e-6 / c.value

    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_host[i] = avg_lmbd_v1(f_nu=total_spline,
                                 trans_spline=spline_lorri,
                                 lambda_array=waves_ebl)
    #-----------------
    total_yy = (host_axion_contr(waves_ebl, axion_mass_array[i], 1e-10,
                                 v_dispersion=220)
                + cosmic_axion_contr(waves_ebl,
                            axion_mass_array[i], 1e-10)
                + spline_cuba(waves_ebl)
                ) * waves_ebl * 1e-6 / c.value

    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_total[i] = avg_lmbd_v1(f_nu=total_spline,
                                 trans_spline=spline_lorri,
                                 lambda_array=waves_ebl)
    #-----------------

    total_yy = (cosmic_axion_contr(waves_ebl,
                            axion_mass_array[i], 1e-10)
                ) * waves_ebl * 1e-6 / c.value
    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_only[i] = avg_lmbd_v1(f_nu=total_spline,
                                      trans_spline=spline_lorri,
                                      lambda_array=waves_ebl)

ax2.plot(axion_mass_array, mean_lambda_cosmic,# marker='.',
         label='CUBA + cosmic', lw=2)
ax2.plot(axion_mass_array, mean_lambda_host, #marker='.',
         label='CUBA + host', lw=2)
ax2.plot(axion_mass_array, mean_lambda_total, #marker='.',
         label='CUBA + cosmic + host', lw=2)
# ax2.plot(axion_mass_array, mean_lambda_only, #marker='.',
#          label='Cosmic axion', lw=2)
ax.plot(waves_ebl, spline_cuba(waves_ebl), 'k')

# ax2.plot(2.48/lorri_trans[:, 0],
#          lorri_trans[:, 1]*0.5 + 0.4,
#          c='grey', alpha=0.8)

ax2.legend(fontsize=22)
ax2.set_ylim(0.45, 0.75)
ax2.set_xlim(2., 9.)
ax2.axhline(0.608, ls='--', c='grey')
ax2.annotate('pivot\nwavelength', xy=(3., 0.584), c='grey',
             fontsize=18, horizontalalignment='center')
ax2.set_xlabel('Axion mass (eV)')
ax2.tick_params(axis='y', pad=12)
ax2.set_ylabel(r'$\langle\lambda\rangle$ ($\mu$m)')

plt.figure()
plt.tight_layout()
plt.plot(lorri_trans[:, 0], lorri_trans[:, 1], marker='.')
plt.xlabel('wavelenght (microns)')
plt.ylabel(r'responsivity')
fig2.savefig('outputs/figures_paper/meanlambda_mass.pdf',
            bbox_inches='tight')
fig2.savefig('outputs/figures_paper/meanlambda_mass.png',
            bbox_inches='tight')

# plt.show()
