# IMPORTS --------------------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data
from astropy.constants import c

from data.emissivity_measurs.emissivity_read_data import emissivity_data

from ebltable.ebl_from_model import EBL

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
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


# FIGURE: EBL FOR DIFFERENT MODELS -----------------------------------

fig = plt.figure(figsize=(15, 8))
axes = fig.gca()

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'g', 'orange', 'grey', 'purple']
j = 0

# We initialize the class with the input file
config_data = read_config_file(
    'scripts/input_files/input_data_Finke.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=True)

# Axion component calculation
# ebl_class.ebl_axion_calculation(
#     axion_mass=float(config_data['axion_params']['axion_mass']),
#     axion_gayy=float(config_data['axion_params']['axion_gayy'])
#     )
# plt.plot(waves_ebl,
#          10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0., grid=False),
#          linestyle=models[3], color='k')

# Intrahalo component calculation
# ebl_class.ebl_intrahalo_calculation(float(
#                                       config_data['ihl_params']['A_ihl']),
#                                     float(
#                                     config_data['ihl_params']['alpha']))
# plt.plot(waves_ebl, 10 ** ebl_class.ebl_ihl_spline(
# freq_array_ebl, 0., grid=False),
# linestyle=models[2], color='k')


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

# SSPs component calculation (all models listed in the input file)
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    ebl_class.ebl_sum_contributions()

    plt.figure(fig)
    plt.plot(waves_ebl, 10 ** ebl_class.ebl_total_spline(
        freq_array_ebl, 0., grid=False),
             linestyle=models[0], color=colors[nkey])
    plt.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0., grid=False),
             linestyle=models[1], color=colors[nkey])

    ebl_class.logging_prints = False

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

    labels_emiss.append(config_data['ssp_models'][key]['name'])
    handles_emiss.append(plt.Line2D([], [], linewidth=2,
                                    linestyle='-',
                                    color=colors[nkey]))
plt.figure(fig)
ax = plt.gca()

import_cb_data(plot_measurs=True, ax1=ax)
np.savetxt('outputs/data_pegasemetall.txt', np.column_stack((waves_ebl,
                                            10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0., grid=False))))
# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)
plt.loglog(waves_ebl, spline_finke(waves_ebl), c='orange', label='Finke22 '
                                                                 'model A')
plt.loglog(waves_ebl, spline_cuba(waves_ebl), c='fuchsia', label='CUBA')


plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
                      title=r'Measurements')
legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
                                  color='k') for i in range(4)],
                      ['Total', 'SSP', 'IHL', 'Axion decay'
                               '\n(example)'
                               '\n'
                               r'    m$_a = 1$ eV'
                               '\n'
                               r'    g$_{a\gamma} = 5 \cdot 10^{-10}$ GeV$^{-1}$'], loc=3,
                      title=r'Components')
legend33 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
                                  color=colors[i])
                       for i in range(len(config_data['ssp_models']))],
                      [config_data['ssp_models'][key]['name']
                       for key in config_data['ssp_models']],
                      title=r'SSP models', bbox_to_anchor=(1.04, 0.1),
                      loc="lower left")
axes.add_artist(legend11)
axes.add_artist(legend22)
axes.add_artist(legend33)

plt.xlim([.1, 200])
plt.ylim(1e-2, 100)
# plt.grid("major")
# plt.grid("minor")
plt.grid(True, which="both", ls="-")
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)


# SFRs FOR SSP MODELS ------------------------------
plt.figure(figsize=(12, 8))

z_data = np.linspace(float(config_data['redshift_array']['zmin']),
                     float(config_data['redshift_array']['zmax']),
                     num=500)


for nkey, key in enumerate(config_data['ssp_models']):
    plt.plot(z_data, ebl_class.sfr_function(
        function_input=config_data['ssp_models'][key]['sfr'],
        xx_array=z_data,
        params=config_data['ssp_models'][key]['sfr_params']),
             label=config_data['ssp_models'][key]['name'])


plt.legend()
plt.yscale('log')
plt.ylabel('sfr(z)')
plt.xlabel('z')


plt.show()
