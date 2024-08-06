# IMPORTS --------------------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator

from ebl_codes.EBL_class import EBL_model
from astropy.constants import c

from data.cb_measurs.import_cb_measurs import import_cb_data
from data.emissivity_measurs.emissivity_read_data import emissivity_data
from data.sfr_measurs.sfr_read import *
from data.metallicity_measurs.import_metall import import_met_data

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

input_file_dir = ('scripts/input_files/')

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


models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'g', 'orange', 'grey',
          'purple', 'k', 'cyan', 'brown']

linstyles_ssp = ['solid', '--', 'dotted', '-.']

markers = ['.', 'x', '+', '*', '^', '>', '<']


# We initialize the class with the input file
config_data = read_config_file(input_file_dir + 'input_dust_reem.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=True)

# FIGURE: METALLICITIES FOR DIFFERENT MODELS ---------------------------
fig_met, ax_met = plt.subplots(figsize=(8, 8))
plt.yscale('log')
aa = import_met_data(ax=ax_met)
# aa = import_met_data(ax=ax_met, z_sun=0.014)

plt.xlabel('redshift z')
plt.ylabel('Z')

# FIGURE: COB FOR DIFFERENT MODELS -------------------------------------
fig_cob, ax_cob = plt.subplots(figsize=(10, 8))

waves_ebl = np.logspace(-1, 3, num=500)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))


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

z_array = np.linspace(0., 10.)

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

# SFRs FOR SSP MODELS ------------------------------
fig_sfr, ax_sfr = plt.subplots(figsize=(12, 8))

z_data = np.linspace(float(config_data['redshift_array']['zmin']),
                     float(config_data['redshift_array']['zmax']),
                     num=500)

sfr_data = sfr_data_dict()
plot_sfr_data(sfr_data)

plt.yscale('log')
plt.ylabel('sfr(z)')
plt.xlabel('z')

# SSP SYNTHETIC SPECTRA

fig_ssp, ax_ssp = plt.subplots(figsize=(10, 8))
plt.xscale('log')
plt.title('More transparency, less metallicity')
# plt.yscale('log')
# plt.ylim(0., 30.)

xx_amstrongs = np.logspace(2, 6, 2000)

ax_ssp.set_xlabel('Wavelength [A]')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{\AA}^{-1}$ M$_{\odot}^{-1}$])')

previous_ssp = []
labels_ssp1 = []
handles_ssp1 = []
labels_ssp2 = []
handles_ssp2 = []

dict_kernels = {}

# SSPs component calculation (all models listed in the input file)
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])

    kernel_emiss = ebl_class.emiss_ssp_calculation(config_data['ssp_models'][
                                                       key])
    kernel_spline = RegularGridInterpolator(
        points=(np.log10(ebl_class._lambda_array),
                ebl_class._log_t_ssp_intcube[0, 0, :]),
        values=kernel_emiss[:, 0, :],
        method='linear',
        bounds_error=False, fill_value=1e-43
    )
    dict_kernels[key] = kernel_spline

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    print(10 ** ebl_class.ebl_ssp_spline(
        np.log10(c.value/0.608*1e6), 0., grid=False),
          21.98 - 10 ** ebl_class.ebl_ssp_spline(
        np.log10(c.value/0.608*1e6), 0., grid=False))

    ax_cob.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0., grid=False),
                linestyle='-', color=colors[nkey % len(colors)],
                lw=3,
                # markersize=16, marker=markers[nkey]
                )

    ebl_class.logging_prints = True

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
                 linestyle='-', marker=markers[nkey],
                 color=colors[nkey % len(colors)], lw=2)

    labels_emiss.append(
        config_data['ssp_models'][key]['name'])
    handles_emiss.append(plt.Line2D([], [], linewidth=2,
                                    linestyle='-',
                                    color=colors[nkey % len(colors)]))

    ax_met.plot(z_array,
                ebl_class.metall_mean(
                    function_input=config_data['ssp_models'][key][
                        'metall_formula'],
                    zz_array=z_array,
                    args=config_data['ssp_models'][key]['args_metall']),
                label=config_data['ssp_models'][key]['name'],
                color=colors[nkey % len(colors)])

    ax_sfr.plot(z_data, ebl_class.sfr_function(
        function_input=config_data['ssp_models'][key]['sfr'],
        xx_array=z_data,
        params=config_data['ssp_models'][key]['sfr_params']),
                label=config_data['ssp_models'][key]['name'],
                color=colors[nkey % len(colors)])

    color_ssp = ['b', 'orange', 'k', 'r', 'green', 'grey', 'limegreen',
                 'purple', 'brown']

    if config_data['ssp_models'][key]['path_SSP'] not in previous_ssp:
        previous_ssp.append(config_data['ssp_models'][key]['path_SSP'])
        labels_ssp2.append(
            config_data['ssp_models'][key]['path_SSP'].replace(
                'data/ssp_synthetic_spectra/', ''))
        handles_ssp2.append(
            plt.Line2D([], [], linewidth=2,
                       linestyle=linstyles_ssp[len(previous_ssp) - 1],
                       color='k'))
        list_met = ebl_class._ssp_metall
        list_met = np.sort(list_met)
        for n_met, met in enumerate(list_met):
            for i, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
                ax_ssp.plot(
                    xx_amstrongs,
                    ebl_class.ssp_lumin_spline(
                        xi=(
                            np.log10(c.value / xx_amstrongs * 1e10),
                            age, np.log10(met)),
                    ),
                    linestyle=linstyles_ssp[len(previous_ssp) - 1],
                    color=color_ssp[i],
                    alpha=float(n_met) / len(list_met) * 1.1
                )
                if n_met == 0 and len(previous_ssp) == 1:
                    labels_ssp1.append(age)
                    handles_ssp1.append(
                        plt.Line2D([], [], linewidth=2, linestyle='-',
                                   color=color_ssp[i]))

plt.figure(fig_cob)
import_cb_data(plot_measurs=True, ax1=ax_cob, lambda_max_total=1000)

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)
ax_cob.plot(waves_ebl, spline_finke(waves_ebl),
            c='orange', label='Finke22 model A')
ax_cob.plot(waves_ebl, spline_cuba(waves_ebl),
            c='fuchsia', label='CUBA')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

# legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
#                       title=r'Measurements')
# legend22 = plt.legend([
#     plt.Line2D([], [], linewidth=2, linestyle=models[i],
#                color='k') for i in range(4)],
#     ['Total', 'SSP', 'IHL', 'Axion decay'
#                             '\n(example)'
#                             '\n'
#                             r'    m$_a = 1$ eV'
#                             '\n'
#                             r'    g$_{a\gamma} = 5 \cdot 10^{-10}$ GeV$^{'
#                             r'-1}$'], loc=4,
#     title=r'Components')
legend33 = ax_cob.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
                                     color=colors[i])
                          for i in range(len(config_data['ssp_models']))],
                         [config_data['ssp_models'][key]['name']
                          for key in config_data['ssp_models']],
                         title=r'SSP models',  # bbox_to_anchor=(1.04, 0.1),
                         loc=1
                         )
# axes.add_artist(legend11)
# axes.add_artist(legend22)
ax_cob.add_artist(legend33)

plt.xlim([.1, 1000])
plt.ylim(1e-2, 100)

ax_sfr.legend()
ax_met.legend()
print(previous_ssp)

plt.figure(fig_ssp)
legend11 = plt.legend(handles_ssp1, labels_ssp1,
                      loc=1,
                      title=r'Age log10(years)')
legend22 = plt.legend(handles_ssp2, labels_ssp2,
                      loc=4,
                      )
ax_ssp.add_artist(legend11)
ax_ssp.add_artist(legend22)

np.savetxt('outputs/data_pegasemetall.txt',
           np.column_stack((
               waves_ebl,
               10 ** ebl_class.ebl_ssp_spline(freq_array_ebl, 0.,
                                              grid=False))))

# Save the figures
fig_cob.savefig(input_file_dir + '/ebl_bare' + '.png',
                bbox_inches='tight')
fig_cob.savefig(input_file_dir + '/ebl_bare' + '.pdf',
                bbox_inches='tight')

fig_sfr.savefig(input_file_dir + '/sfr_bare' + '.png',
                bbox_inches='tight')
fig_sfr.savefig(input_file_dir + '/sfr_bare' + '.pdf',
                bbox_inches='tight')

fig_met.savefig(input_file_dir + '/Zev_bare' + '.png',
                bbox_inches='tight')
fig_met.savefig(input_file_dir + '/Zev_bare' + '.pdf',
                bbox_inches='tight')

fig_ssp.savefig(input_file_dir + '/ssp_bare' + '.png',
                bbox_inches='tight')
fig_ssp.savefig(input_file_dir + '/ssp_bare' + '.pdf',
                bbox_inches='tight')

fig_emiss_z.subplots_adjust(wspace=0, hspace=0)
fig_emiss_z.savefig(
    input_file_dir + '/emiss_redshift_bare' + '.png',
    bbox_inches='tight')
fig_emiss_z.savefig(
    input_file_dir + '/emiss_redshift_bare' + '.pdf',
    bbox_inches='tight')
fig_mean, ax_mean = plt.subplots()
plt.xscale('log')
plt.yscale('log')
i = 0

for i, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
    plt.loglog(
        ebl_class._lambda_array,
        (dict_kernels['SB99_Raue']((np.log10(ebl_class._lambda_array), age))),
                       ls='--', c=colors[i], marker=markers[0], label=age)
    plt.loglog(
        ebl_class._lambda_array,
         dict_kernels['SB99_dustFinke'](
                    (np.log10(ebl_class._lambda_array), age)),
        ls='-', c=colors[i], alpha=0.5, marker=markers[1])
    i += 1
plt.legend()

plt.figure()
i = 0

for i, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
    plt.loglog(
        ebl_class._lambda_array,
        (dict_kernels['SB99_Raue']((np.log10(ebl_class._lambda_array), age))
         / dict_kernels['SB99_dustFinke'](
                    (np.log10(ebl_class._lambda_array), age))),
        ls='--', c=colors[i], label=age)
    i += 1
plt.legend()

plt.show()
