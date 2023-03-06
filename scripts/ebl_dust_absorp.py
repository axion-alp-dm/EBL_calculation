# IMPORTS --------------------------------------------#
import os
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.EBL_measurs_plot import plot_ebl_measurement_collection

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
if not os.path.exists("../outputs/"):
    os.makedirs("../outputs/")


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


def input_yaml_data_into_class(yaml_data, log_prints=False):
    z_array = np.linspace(float(yaml_data['redshift_array']['zmin']),
                          float(yaml_data['redshift_array']['zmax']),
                          yaml_data['redshift_array']['zsteps'])

    lamb_array = np.logspace(np.log10(float(
        yaml_data['wavelenght_array']['lmin'])),
        np.log10(float(
            yaml_data['wavelenght_array']['lmax'])),
        yaml_data['wavelenght_array']['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     h=float(yaml_data['cosmology_params']['cosmo'][0]),
                     omegaM=float(yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'],
                     log_prints=log_prints)


# FIGURE: EBL FOR DIFFERENT MODELS -----------------------------------

fig = plt.figure(figsize=(15, 8))
axes = fig.gca()
plt.title('Popstar different metallicities')

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

models = ['solid', 'dashed', 'dotted']  # , 'dashdot']
colors = ['b', 'r', 'g']
j = 0

model_finke = np.loadtxt('ebl_codes/EBL_intensity_total_z0.00.dat')
print(np.shape(model_finke))
plt.plot(model_finke[:, 0]/1e4, model_finke[:, 1], '-k', label='Finke Model A')

# We initialize the class with the input file
config_data = read_config_file('scripts/input_data_change_metallicities.yml')
ebl_class = input_yaml_data_into_class(config_data, log_prints=True)

# SSPs component calculation (all models listed in the input file)
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])
    init = time.process_time()
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    ebl_class.ebl_sum_contributions()
    print('Time for SSP calculation: %.2fs' % (time.process_time() - init))

    if config_data['ssp_models'][key]['dust_abs_models'] == ['finke2022_2']:
        linestyle = '-'
    elif config_data['ssp_models'][key]['dust_abs_models'] == ['finke2022']:
        linestyle = '--'
    else:
        linestyle = 'dotted'

    # if config_data['ssp_models'][key]['ssp_type'] == 'SB99':
    #     color = colors[0]

    if config_data['ssp_models'][key]['file_name'] \
            == 'spneb_kro_0.15_100_z0200_t':
            color = colors[0]
    elif config_data['ssp_models'][key]['file_name']\
            == 'spneb_kro_0.15_100_z0040_t':
            color = colors[1]
    else:
            color = colors[2]

    plt.figure(fig)
    plt.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0., grid=False),
             linestyle=linestyle, color=color)

    ebl_class.logging_prints = False
plt.figure(fig)
plot_ebl_measurement_collection('ebl_measurements/EBL_measurements.yml')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
                      title=r'Measurements')
# legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
#                                   color=colors[i])
#                        for i in range(2)],
#                       ['SB99 with small ages', 'SB99 only old ages'],
#                       loc=8,
#                       title=r'SSP model')
# # legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
# #                                   color=colors[i])
# #                        for i in range(len(colors))],
# #                       ['SB99', 'PopStar09 cut', 'PopStar09 full'],
# #                       loc=8,
# #                       title=r'SSP model')
# legend33 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
#                                   color='k') for i in [0, 2]],  # range(3)],
#                       ['Finke model A', 'Kneiske02', 'Razzaque14 + z Finke'],
#                       title=r'Dust absorption model',
#                       bbox_to_anchor=(1.04, 0.1),
#                       loc="lower left")

legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
                                  color=colors[i])
                       for i in range(3)],
                      ['0.02', '0.004', '0.0001'],
                      loc=8,
                      title=r'Metallicity')
legend33 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
                                  color='k') for i in [0, 2]],  # range(3)],
                      ['Finke model A', 'Kneiske02', 'Razzaque14 + z Finke'],
                      title=r'Dust absorption model',
                      bbox_to_anchor=(1.04, 0.1),
                      loc="lower left")
axes.add_artist(legend11)
axes.add_artist(legend22)
axes.add_artist(legend33)

plt.xlim([.09, 30])
plt.ylim(1e-1, 100)
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)

# SFRs FOR SSP MODELS ------------------------------
plt.figure(figsize=(12, 8))

z_data = np.linspace(float(config_data['redshift_array']['zmin']),
                     float(config_data['redshift_array']['zmax']),
                     num=500)


def sfr(zz, str_sfr, params):
    return eval(str_sfr)(params, zz)


for nkey, key in enumerate(config_data['ssp_models']):
    plt.plot(z_data, sfr(z_data, config_data['ssp_models'][key]['sfr'],
                         config_data['ssp_models'][key]['sfr_params']),
             label=config_data['ssp_models'][key]['name'])

plt.legend()
plt.yscale('log')
plt.ylabel('sfr(z)')
plt.xlabel('z')

plt.show()
