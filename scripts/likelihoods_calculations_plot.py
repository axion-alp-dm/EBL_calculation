# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model

from emissivity_data.emissivity_read_data import emissivity_data
from ebl_measurements.import_cb_measurs import import_cb_data
from sfr_data.sfr_read import *

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

from ebltable.ebl_from_model import EBL

all_size = 22
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=all_size)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)
# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

direct_name = str('final_outputs_NOlims 2023-10-05 08:05:19'
                  # + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# Configuration file reading and data input/output ---------#
with open('outputs/' + direct_name + '/input_data.yml', 'r') as file:
    config_data = yaml.safe_load(file)

ebl_class = EBL_model.input_yaml_data_into_class(config_data)
# ebl_class.logging_prints = True

waves_ebl = np.logspace(-1, 1)
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

colors = ['b', 'r', 'k', 'k']
f_color = ['dodgerblue', 'C1']
models = ['solid', 'solid', 'dashed', 'dotted']

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)

# FIGURE: COB fit ------------------------------------------------
fig_ebl = plt.figure(figsize=(10, 7.5))
axes_ebl = fig_ebl.gca()

plt.yscale('log')
plt.xscale('log')

plt.ylim(0.1, 120)
plt.xlim(0.1, 10.)

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

plt.plot(waves_ebl, nuInu['finke2022'], ls='--', color='k', lw=2.)
plt.plot(waves_ebl, nuInu['cuba'], ls='dotted', color='k', lw=2.)

# COB measurements that we are going to use
upper_lims_ebldata, igl_ebldata = import_cb_data(
    lambda_min_total=0.1, lambda_max_total=5.,
    plot_measurs=True, ax1=axes_ebl)

upper_lims_ebldata_woNH = upper_lims_ebldata[
    upper_lims_ebldata['ref'] != r'NH/LORRI (Lauer+ \'22)']

plt.legend([plt.Line2D([], [], linewidth=2,
                       linestyle=models[i], color=colors[i])
            for i in range(4)],
           ['Model A', 'Model B', 'Finke22', 'CUBA'],
           loc=8,
           title=r'EBL models')

# FIGURE: sfr fit ------------------------------------------------
fig_sfr = plt.figure(figsize=(12, 8))
axes_sfr = fig_sfr.gca()

sfr_data = sfr_data_dict('sfr_data/')
plot_sfr_data(sfr_data)

legend1 = plt.legend(title='Measurements', loc=1, fontsize=18,
                     title_fontsize=20, framealpha=1)

legend2 = plt.legend([plt.Line2D([], [], linewidth=2,
                                 linestyle=models[i], color=colors[i])
                      for i in range(3)],
                     ['Model A', 'Model B', 'MD14'],
                     loc=3, bbox_to_anchor=(0.1, 0.),
                     fontsize=18,
                     title=r'Models', title_fontsize=20
                     )

axes_sfr.add_artist(legend1)

x_sfr = np.linspace(0, 10)

axes_sfr.plot(x_sfr, ebl_class.sfr_function(
    'lambda x, ci : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[2])**ci[3])',
    x_sfr, [0.015, 2.7, 2.9, 5.6]),
              color='k', linestyle='--', lw=2)
# axes_sfr.plot(x_sfr, ebl_class.sfr_function(
#     'lambda x, ci : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[2])**ci[3])',
#     x_sfr, [0.0092, 2.79, 3.10, 6.97]),
#               color='k', linestyle='dotted', lw=2)

plt.yscale('log')

plt.xlim(0, 10)

plt.xlabel('redshift z')
plt.ylabel(r'sfr (M$_{\odot}$ / yr / Mpc$^{3}$)')

# FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS ------------------
fig_emiss_z, axes_emiss_z = plt.subplots(3, 3, figsize=(15, 15))

z_array = np.linspace(0, 10)

for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                               0.44, 0.55, 0.79,
                               1.22, 2.2, 3.6]):
    plt.subplot(3, 3, n_lambda + 1)
    emissivity_data(directory='emissivity_data/',
                    z_min=None, z_max=None,
                    lambda_min=ll - 0.05, lambda_max=ll + 0.05,
                    take1ref=None, plot_fig=True)

    plt.annotate(r'%r$\,\mu m$' % ll, xy=(7, 3e34))

    plt.xlim(min(z_array), max(z_array))
    plt.ylim(1e33, 3e35)

    plt.yscale('log')

plt.subplot(3, 3, 8)
plt.xlabel(r'redshift z', fontsize=22)

plt.subplot(3, 3, 4)
plt.ylabel(r'$_{\nu} \varepsilon_{_{\nu} \,\,(\mathrm{W\, / \, Mpc}^3)}$',
           fontsize=30)

plt.subplot(3, 3, 3)
plt.legend([plt.Line2D([], [], linewidth=2,
                       linestyle=models[i], color=colors[i])
            for i in range(2)],
           ['Model A', 'Model B'],
           loc=1, fontsize=16)

ax = [plt.subplot(3, 3, i) for i in [2, 3, 5, 6, 8, 9]]
for a in ax:
    a.set_yticklabels([])

ax = [plt.subplot(3, 3, i + 1) for i in range(6)]
for a in ax:
    a.set_xticklabels([])

ax = [plt.subplot(3, 3, i + 1) for i in range(6, 8)]
for a in ax:
    a.set_xticks([0, 2, 4, 6, 8])

ax = [plt.subplot(3, 3, i) for i in range(1, 7)]
for a in ax:
    a.set_xticks([0, 2, 4, 6, 8, 10])

a = plt.subplot(3, 3, 9)
a.set_xticks([0, 2, 4, 6, 8, 10])

emiss_data = emissivity_data(directory='emissivity_data/')
freq_emiss = c.value / (emiss_data['lambda'] * 1e-6)

# EMISSIVIY AT Z=0 FIGURE ----------------------------------------
fig_emiss = plt.figure(figsize=(12, 8))
axes_emiss = fig_emiss.gca()

data_emiss = emissivity_data(directory='emissivity_data/', z_max=0.1)

axes_emiss.errorbar(x=data_emiss['lambda'], y=data_emiss['eje'],
                    yerr=(data_emiss['eje_n'], data_emiss['eje_p']),
                    linestyle='', marker='o')

plt.legend([plt.Line2D([], [], linewidth=2,
                       linestyle=models[i], color=colors[i])
            for i in range(2)],
           ['Model A', 'Model B'],
           loc=8, fontsize=22, title='Models')

plt.yscale('log')
plt.xscale('log')

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$_{\nu} \varepsilon_{_{\nu} \,\,(\mathrm{W\, / \, Mpc}^3)}$',
           fontsize=30)

plt.xlim(0.09, 10)
plt.ylim(1e33, 1e35)

for nkey, key in enumerate(config_data['ssp_models']):

    values_sfr = config_data['ssp_models'][key]['sfr_params']
    values_cov = config_data['ssp_models'][key]['cov_matrix']
    values_cov = np.array(values_cov).reshape(len(values_sfr), len(values_sfr))

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])


    def fit_igl(lambda_igl, params):
        config_data['ssp_models'][key]['sfr_params'] = params.copy()
        return ebl_class.ebl_ssp_individualData(
            yaml_data=config_data['ssp_models'][key],
            x_data=lambda_igl)


    def fit_emiss(x_all, params):
        lambda_emiss, z_emiss = x_all
        freq_emissions = np.log10(c.value / lambda_emiss * 1e6)

        config_data['ssp_models'][key]['sfr_params'] = params.copy()

        ebl_class.emiss_ssp_calculation(config_data['ssp_models'][key])

        return 10 ** (freq_emissions
                      + ebl_class.emiss_ssp_spline(freq_emissions,
                                                   z_emiss)
                      - 7)


    def sfr(x, params):
        return ebl_class.sfr_function(
            config_data['ssp_models'][key]['sfr'], x, params)


    # FIGURE: cob fit
    axes_ebl.plot(waves_ebl,
                  10 ** ebl_class.ebl_ssp_spline(freq_array_ebl, 0.,
                                                 grid=False),
                  color=colors[nkey], lw=2)

    # y, y_cov = propagate(lambda pars:
    #                      fit_igl(waves_ebl, pars),
    #                      values_sfr, values_cov)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # axes_ebl.fill_between(waves_ebl, y - yerr_prop, y + yerr_prop,
    #                       facecolor=f_color[nkey], alpha=0.5)
    # print(yerr_prop)


    # FIGURE: SFR
    plt.figure(fig_sfr)
    axes_sfr.plot(x_sfr, sfr(x_sfr, values_sfr), '-',
                  color=colors[nkey], lw=2)

    y, y_cov = propagate(lambda pars:
                         sfr(x_sfr, pars),
                         values_sfr, values_cov)
    yerr_prop = np.diag(y_cov) ** 0.5
    plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
                     facecolor=f_color[nkey], alpha=0.5)
    print(yerr_prop)

    # FIGURE: emissivities fit
    plt.figure(fig_emiss_z)
    for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                                   0.44, 0.55, 0.79,
                                   1.22, 2.2, 3.6]):
        plt.subplot(3, 3, n_lambda + 1)

        plt.plot(z_array,
                 (c.value / (ll * 1e-6))
                 * 10 ** ebl_class.emiss_ssp_spline(
                     np.log10(c.value / ll * 1e6) * np.ones(
                         len(z_array)),
                     z_array)
                 * 1e-7,
                 linestyle='-', color=colors[nkey], lw=2)

        y, y_cov = propagate(lambda pars:
                             fit_emiss((ll * np.ones(len(z_array)), z_array),
                                       pars),
                             values_sfr, values_cov)
        yerr_prop = np.diag(y_cov) ** 0.5
        plt.fill_between(z_array, y - yerr_prop, y + yerr_prop,
                         facecolor=f_color[nkey], alpha=0.5)

        # print(yerr_prop)

    # FIGURE: emissivities at z=0
    # axes_emiss.plot(waves_ebl,
    #                 10 ** (freq_array_ebl
    #                        + ebl_class.emiss_ssp_spline(
    #                             freq_array_ebl, np.zeros(len(waves_ebl)))
    #                        - 7),
    #                 linestyle='-', color=colors[nkey], lw=2)

    # y, y_cov = propagate(
    #     lambda pars:
    #     fit_emiss((waves_ebl, np.zeros(len(waves_ebl))), pars),
    #     values_sfr, values_cov)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # plt.fill_between(waves_ebl, y - yerr_prop, y + yerr_prop,
    #                  facecolor=f_color[nkey], alpha=0.5)
    # print(yerr_prop)

# -------------------------------------------------------------
fig_ebl.savefig('outputs/' + direct_name + '/ebl' + '.png',
                bbox_inches='tight')
fig_ebl.savefig('outputs/' + direct_name + '/ebl' + '.pdf',
                bbox_inches='tight')

fig_sfr.savefig('outputs/' + direct_name + '/sfr' + '.png',
                bbox_inches='tight')
fig_sfr.savefig('outputs/' + direct_name + '/sfr' + '.pdf',
                bbox_inches='tight')

fig_emiss_z.subplots_adjust(wspace=0, hspace=0)
fig_emiss_z.savefig(
    'outputs/' + direct_name + '/emiss_redshift' + '.png',
    bbox_inches='tight')
fig_emiss_z.savefig(
    'outputs/' + direct_name + '/emiss_redshift' + '.pdf',
    bbox_inches='tight')

fig_emiss.savefig('outputs/' + direct_name + '/luminosities' + '.png',
                  bbox_inches='tight')
fig_emiss.savefig('outputs/' + direct_name + '/luminosities' + '.pdf',
                  bbox_inches='tight')
