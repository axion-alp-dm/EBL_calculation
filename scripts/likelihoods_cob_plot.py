# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model

from data.emissivity_measurs.emissivity_read_data import emissivity_data
from data.cb_measurs.import_cb_measurs import import_cb_data
from data.sfr_measurs.sfr_read import *
from data.metallicity_measurs.import_metall import import_met_data

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

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

direct_name = str('outputs/final_outputs_Zevol_fixezZsolar '
                       '2024-06-05 09:10:05/')
print(direct_name)

# Configuration file reading and data input/output ---------#
with open(direct_name + '/input_data.yml', 'r') as file:
    config_data = yaml.safe_load(file)

ebl_class = EBL_model.input_yaml_data_into_class(config_data)
# ebl_class.logging_prints = True

waves_ebl = np.logspace(-1, 1)
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

colors = ['b', 'tab:orange']
f_color = ['dodgerblue', 'C1']

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

handles_cob, labels_cob = [], []

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

# COB measurements that we are going to use
import_cb_data(
    lambda_min_total=0.1, lambda_max_total=5.,
    plot_measurs=True, ax1=axes_ebl)

# FIGURE: sfr fit ------------------------------------------------
fig_sfr = plt.figure(figsize=(12, 9))
axes_sfr = fig_sfr.gca()

x_sfr = np.linspace(0, 10)

sfr_data = sfr_data_dict()
plot_sfr_data(sfr_data)

handles, labels = axes_sfr.get_legend_handles_labels()
handles = [h[0] for h in handles]

legend1 = plt.legend(handles, labels,
                     title='Measurements', loc=1, fontsize=20,
                     title_fontsize=24, framealpha=0.8)
axes_sfr.add_artist(legend1)

handles_sfr, labels_sfr = [], []

plt.yscale('log')

plt.xlim(0, 10)

plt.xlabel('redshift z')
plt.ylabel(r'$\rho_{\star}$ (M$_{\odot}$ / yr / Mpc$^{3}$)')


fig_Z, ax_met = plt.subplots(figsize=(8, 8))
plt.yscale('log')
aa = import_met_data(ax=ax_met)
plt.xlim(0, 10)

plt.xlabel('redshift z')
plt.ylabel('Z')


# FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS ------------------
fig_emiss_z, axes_emiss_z = plt.subplots(3, 3, figsize=(12, 12))

z_array = np.linspace(0, 10)

for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                               0.44, 0.55, 0.79,
                               1.22, 2.2, 3.6]):
    plt.subplot(3, 3, n_lambda + 1)
    emissivity_data(z_min=None, z_max=None,
                    lambda_min=ll - 0.05, lambda_max=ll + 0.05,
                    take1ref=None, plot_fig=True)

    if n_lambda != 8:
        plt.annotate(r'%r$\,\mu m$' % ll, xy=(5, 1e35), fontsize=28)

    plt.xlim(min(z_array), max(z_array))
    plt.ylim(1e33, 3e35)

    plt.yscale('log')

handles_emiss, labels_emiss = [], []

plt.subplot(3, 3, 8)
plt.xlabel(r'redshift z', fontsize=34)

plt.subplot(3, 3, 4)
plt.ylabel(r'$_{\nu} \varepsilon_{_{\nu} \,\,(\mathrm{W\, / \, Mpc}^3)}$',
           fontsize=40)

plt.subplot(3, 3, 9)
plt.annotate(r'3.6$\,\mu m$', xy=(6, 1e34), fontsize=28)

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

emiss_data = emissivity_data()
freq_emiss = c.value / (emiss_data['lambda'] * 1e-6)


for nkey, key in enumerate(config_data['ssp_models']):

    values_sfr = config_data['ssp_models'][key]['sfr_params']
    values_metall = config_data['ssp_models'][key]['args_metall']
    values_sfr = np.concatenate((values_sfr, values_metall))
    print(values_sfr)
    values_cov = config_data['ssp_models'][key]['cov_matrix']
    values_cov = np.array(values_cov).reshape(
        8, 8)

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


    def metall(x, params):
        return ebl_class.metall_mean(
            function_input=config_data['ssp_models'][key]['metall_formula'],
            zz_array=x,
            args=params[4:])

    # FIGURE: cob fit
    axes_ebl.plot(waves_ebl,
                  10 ** ebl_class.ebl_ssp_spline(freq_array_ebl, 0.,
                                                 grid=False),
                  color=colors[nkey], lw=2)

    labels_cob.append(config_data['ssp_models'][key]['name'])
    handles_cob.append(plt.Line2D([], [], linewidth=2,
                                  linestyle='-',
                                  color=colors[nkey]))

    # y, y_cov = propagate(lambda pars:
    #                      fit_igl(waves_ebl, pars),
    #                      values_sfr, values_cov)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # axes_ebl.fill_between(waves_ebl, y - yerr_prop, y + yerr_prop,
    #                       facecolor=f_color[nkey], alpha=0.5)
    # print(y)
    # print(yerr_prop)
    # print()

    # FIGURE: SFR
    plt.figure(fig_sfr)
    axes_sfr.plot(x_sfr, sfr(x_sfr, values_sfr), '-',
                  color=colors[nkey], lw=2)

    labels_sfr.append(config_data['ssp_models'][key]['name'])
    handles_sfr.append(plt.Line2D([], [], linewidth=3,
                                  linestyle='-',
                                  color=colors[nkey]))

    y, y_cov = propagate(lambda pars:
                         sfr(x_sfr, pars),
                         values_sfr, values_cov)
    yerr_prop = np.diag(y_cov) ** 0.5
    plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
                     facecolor=f_color[nkey], alpha=0.5)
    print(y)
    print(yerr_prop)

    # Fig Z
    plt.figure(fig_Z)
    plt.plot(x_sfr, metall(x_sfr, params=values_sfr))

    # y, y_cov = propagate(lambda pars:
    #                      metall(x_sfr, pars),
    #                      values_sfr, values_cov)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
    #                  facecolor=f_color[nkey], alpha=0.5)
    # print(y)
    # print(yerr_prop)

    # FIGURE: emissivities fit
    # plt.figure(fig_emiss_z)
    # for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
    #                                0.44, 0.55, 0.79,
    #                                1.22, 2.2, 3.6]):
    #     plt.subplot(3, 3, n_lambda + 1)
    #
    #     plt.plot(z_array,
    #              (c.value / (ll * 1e-6))
    #              * 10 ** ebl_class.emiss_ssp_spline(
    #                  np.log10(c.value / ll * 1e6) * np.ones(
    #                      len(z_array)),
    #                  z_array)
    #              * 1e-7,
    #              linestyle='-', color=colors[nkey], lw=2)
    #
    # labels_emiss.append(config_data['ssp_models'][key]['name'])
    # handles_emiss.append(plt.Line2D([], [], linewidth=2,
    #                                 linestyle='-',
    #                                 color=colors[nkey]))
    #
    # y, y_cov = propagate(lambda pars:
    #                      fit_emiss((ll * np.ones(len(z_array)), z_array),
    #                                pars),
    #                      values_sfr, values_cov)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # plt.fill_between(z_array, y - yerr_prop, y + yerr_prop,
    #                  facecolor=f_color[nkey], alpha=0.5)

    # print(yerr_prop)

# -------------------------------------------------------------
plt.figure(fig_ebl)

plt.plot(waves_ebl, nuInu['finke2022'], ls='--', color='magenta', lw=2.)
labels_cob.append('Finke22')
handles_cob.append(plt.Line2D([], [], linewidth=2,
                              linestyle='-',
                              color='magenta'))

plt.plot(waves_ebl, nuInu['cuba'], ls='dotted', color='k', lw=2.)
labels_cob.append('CUBA')
handles_cob.append(plt.Line2D([], [], linewidth=2,
                              linestyle='-',
                              color='k'))

plt.legend(handles_cob, labels_cob,
           loc=8,
           title=r'Models')


plt.figure(fig_sfr)

axes_sfr.plot(x_sfr, ebl_class.sfr_function(
    'lambda x, ci : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[2])**ci[3])',
    x_sfr, [0.015, 2.7, 2.9, 5.6]),
              color='k', linestyle='--', lw=2)

labels_sfr.append('MD14')
handles_sfr.append(plt.Line2D([], [], linewidth=2,
                              linestyle='--',
                              color='k'))

legend2 = plt.legend(handles_sfr, labels_sfr,
                     loc=3, bbox_to_anchor=(0.01, 0.),
                     fontsize=26, title='Models',title_fontsize=28
                     )
axes_sfr.add_artist(legend2)


plt.figure(fig_emiss_z)
plt.subplot(3, 3, 9)
plt.legend(handles_emiss, labels_emiss,
           loc=1, fontsize=24)

# Save the figures
# fig_ebl.savefig(direct_name + '/ebl' + '.png',
#                 bbox_inches='tight')
# fig_ebl.savefig(direct_name + '/ebl' + '.pdf',
#                 bbox_inches='tight')
#
fig_sfr.savefig(direct_name + '/sfr' + '.png',
                bbox_inches='tight')
fig_sfr.savefig(direct_name + '/sfr' + '.pdf',
                bbox_inches='tight')
#
# fig_Z.savefig(direct_name + '/Zev' + '.png',
#                 bbox_inches='tight')
# fig_Z.savefig(direct_name + '/Zev' + '.pdf',
#                 bbox_inches='tight')

# fig_emiss_z.subplots_adjust(wspace=0, hspace=0)
# fig_emiss_z.savefig(
#     direct_name + '/emiss_redshift' + '.png',
#     bbox_inches='tight')
# fig_emiss_z.savefig(
#     direct_name + '/emiss_redshift' + '.pdf',
#     bbox_inches='tight')
plt.show()

