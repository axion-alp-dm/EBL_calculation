# IMPORTS --------------------------------------------#
import os
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


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


if not os.path.exists("../outputs/"):
    # if the directory for outputs is not present, create it.
    os.makedirs("../outputs/")


def input_yaml_data_into_class(yaml_data):
    z_array = np.linspace(float(yaml_data['redshift_array']['zmin']),
                          float(yaml_data['redshift_array']['zmax']),
                          yaml_data['redshift_array']['zsteps'])

    lamb_array = np.logspace(np.log10(float(yaml_data['wavelenght_array']['lmin'])),
                             np.log10(float(yaml_data['wavelenght_array']['lmax'])),
                             yaml_data['wavelenght_array']['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     h=float(yaml_data['cosmology_params']['cosmo'][0]),
                     omegaM=float(yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'])


# FIGURE: EBL FOR DIFFERENT MODELS -----------------------------------

fig = plt.figure(figsize=(15, 8))
axes = fig.gca()

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'g', 'orange']
j = 0

# We initialize the class with the input file
config_data = read_config_file('input_data.yml')
ebl_class = input_yaml_data_into_class(config_data)

# Axion component calculation
ebl_class.ebl_axion_calculation(mass=float(config_data['axion_params']['axion_mass']),
                                gamma=float(config_data['axion_params']['axion_gamma']))
plt.plot(waves_ebl, 10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0., grid=False), linestyle=models[3],
         color='k')

# Intrahalo component calculation
ebl_class.ebl_intrahalo_calculation(float(config_data['ihl_params']['A_ihl']),
                                    float(config_data['ihl_params']['alpha']))
plt.plot(waves_ebl, 10 ** ebl_class.ebl_intra_spline(freq_array_ebl, 0., grid=False), linestyle=models[2],
         color='k')

# SSPs component calculation (all models listed in the input file)
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

    plt.plot(waves_ebl, 10 ** ebl_class.ebl_total_spline(freq_array_ebl, 0., grid=False),
             linestyle=models[0], color=colors[nkey])
    plt.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(freq_array_ebl, 0., grid=False),
             linestyle=models[1], color=colors[nkey])

    ebl_class.logging_prints = False

plot_ebl_measurement_collection('../ebl_measurements/EBL_measurements.yml')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title=r'Measurements')
legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i], color='k') for i in range(4)],
                      ['Total', 'SSP', 'IHL', 'Axion decay'], loc=3, title=r'Components')
legend33 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-', color=colors[i])
                       for i in range(len(config_data['ssp_models']))],
                      [config_data['ssp_models'][key]['name'] for key in config_data['ssp_models']],
                      title=r'SSP models', bbox_to_anchor=(1.04, 0.1), loc="lower left")
axes.add_artist(legend11)
axes.add_artist(legend22)
axes.add_artist(legend33)

plt.xlim([.1, 1E3])
plt.ylim(1e-2, 100)
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)


# SFRs FOR SSP MODELS ------------------------------
plt.figure(figsize=(12, 8))

z_data = np.linspace(float(config_data['redshift_array']['zmin']),
                     float(config_data['redshift_array']['zmax']),
                     num=500)


def sfr(zz, str_sfr, params):
    return eval(str_sfr)(params, zz)


for nkey, key in enumerate(config_data['ssp_models']):
    plt.plot(z_data, sfr(z_data, config_data['ssp_models'][key]['sfr'], config_data['ssp_models'][key]['sfr_params']),
             label=config_data['ssp_models'][key]['name'])


plt.legend()
plt.yscale('log')
plt.ylabel('sfr(z)')
plt.xlabel('z')


plt.show()
