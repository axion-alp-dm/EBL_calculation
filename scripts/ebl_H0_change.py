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

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


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


def input_yaml_data_into_class(yaml_data, log_prints=False):
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
                     z_max=yaml_data['z_intmax'],
                     log_prints=log_prints)


# FIGURE: EBL FOR DIFFERENT MODELS -----------------------------------

# We initialize the class with the input file
config_data = read_config_file('scripts/input_data.yml')
ebl_class = input_yaml_data_into_class(config_data, log_prints=True)

h0_parameters = [50, 70, 100]

fig = plt.figure(figsize=(15, 8))
axes = fig.gca()

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'g', 'orange']
alphas = np.linspace(0.2, 1., num=len(h0_parameters))
j = 0

for n_h0, h0 in enumerate(h0_parameters):
    ebl_class.change_H0(new_H0=h0)
    ebl_class.ebl_all_calculations(log10_Aihl=float(config_data['ihl_params']['A_ihl']),
                                   alpha=float(config_data['ihl_params']['alpha']),
                                   axion_mass=float(config_data['axion_params']['axion_mass']),
                                   axion_gamma=float(config_data['axion_params']['axion_gamma'])
                                   )

    for nkey, key in enumerate(config_data['ssp_models']):
        print()
        print('SSP model: ', config_data['ssp_models'][key]['name'])

        ebl_class.change_ssp_contribution(config_data['ssp_models'][key])

        plt.plot(waves_ebl, 10 ** ebl_class.ebl_total_spline(freq_array_ebl, 0., grid=False),
                 linestyle=models[0], color=colors[nkey], alpha=alphas[n_h0])
        plt.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(freq_array_ebl, 0., grid=False),
                 linestyle=models[1], color=colors[nkey], alpha=alphas[n_h0])

        ebl_class.logging_prints = False

    plt.plot(waves_ebl, 10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0., grid=False),
             linestyle=models[3], color='k', alpha=alphas[n_h0])
    plt.plot(waves_ebl, 10 ** ebl_class.ebl_ihl_spline(freq_array_ebl, 0., grid=False),
             linestyle=models[2], color='k', alpha=alphas[n_h0])

plot_ebl_measurement_collection('ebl_measurements/EBL_measurements.yml')

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
legend44 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-', color='b', alpha=alphas[i])
                       for i in range(len(h0_parameters))],
                      [h0_parameters[i] for i in range(len(h0_parameters))],
                      title=r'H$_0$', loc=4)
axes.add_artist(legend11)
axes.add_artist(legend22)
axes.add_artist(legend33)

plt.xlim([.1, 1E3])
plt.ylim(1e-2, 100)
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)

plt.show()
