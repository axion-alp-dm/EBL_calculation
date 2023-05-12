# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from ebl_codes.EBL_class import EBL_model
from emissivity_data.emissivity_read_data import emissivity_data
from ebl_measurements.EBL_measurs_plot import plot_ebl_measurement_collection

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model

all_size = 18
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

init_time = time.process_time()
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

    lamb_array = np.logspace(np.log10(float(
        yaml_data['wavelenght_array']['lmin'])),
        np.log10(float(
            yaml_data['wavelenght_array']['lmax'])),
        yaml_data['wavelenght_array']['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     h=float(yaml_data['cosmology_params']['cosmo'][0]),
                     omegaM=float(
                         yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(
                         yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'],
                     log_prints=log_prints)


config_data = read_config_file('scripts/input_data_iminuit_test.yml')
ebl_class = input_yaml_data_into_class(config_data, log_prints=True)

fig = plt.figure(figsize=(12, 8))
axes = fig.gca()

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))
z_array = np.zeros(len(waves_ebl))

colors = ['b', 'r', 'g', 'purple']
models = ['solid', 'dashed', 'dotted']

for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])
    ebl_class.emiss_ssp_calculation(config_data['ssp_models'][key])
    ebl_class.emiss_axion_calculation(config_data['axion_params'][
                                          'axion_mass'],
                                      config_data['axion_params'][
                                          'axion_gamma'])
    ebl_class.emiss_sum_contributions()

    # if config_data['ssp_models'][key]['dust_abs_models'] == ['finke2022_2']:
    #     linestyle = '-'
    # elif config_data['ssp_models'][key]['dust_abs_models'] == ['finke2022']:
    #     linestyle = '--'
    # else:
    #     linestyle = 'dotted'

    if config_data['ssp_models'][key]['file_name'] \
            == '0.0001':
        color = colors[0]
    elif config_data['ssp_models'][key]['file_name'] \
            == '0.008':
        color = colors[1]
    elif config_data['ssp_models'][key]['file_name'] \
            == '0.004':
        color = colors[2]
    else:  # '0.02'
        color = colors[3]

    plt.plot(waves_ebl,
             10 ** freq_array_ebl
             * 10 ** ebl_class.emiss_ssp_spline(
                 freq_array_ebl, z_array)
             * 1e-7,
             linestyle='--', color=color)
    plt.plot(waves_ebl,
             10 ** freq_array_ebl
             * 10 ** ebl_class.emiss_axion_spline(
                 freq_array_ebl, z_array, grid=False)
             * 1e-7,
             linestyle='dotted', color=color)
    plt.plot(waves_ebl,
             10 ** freq_array_ebl
             * 10 ** ebl_class.emiss_total_spline(
                 freq_array_ebl, z_array, grid=False)
             * 1e-7,
             linestyle='-', color=color)

plt.title('pegase, z=0, sfr MD14 formula, FinkeA params')
data_emiss = emissivity_data(z_max=1)
plt.errorbar(x=data_emiss['lambda'], y=data_emiss['e.j_e'],
             yerr=data_emiss['e.j_e_n'], linestyle='', marker='o')

plt.yscale('log')
plt.xscale('log')

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{L}_{\nu}$ (W / Mpc$^3$)')

# plt.xlim(0.09, 10)
# plt.ylim(1e33, 1e35)

legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
                                  color=colors[i])
                       for i in range(4)],
                      ['0.0001', '0.008', '0.004', '0.02'],
                      loc=8,
                      title=r'Metallicity')
legend33 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
                                  color='k') for i in [0, 2]],
                      # range(3)],
                      ['Finke model A', 'Kneiske02',
                       'Razzaque14 + z Finke'],
                      title=r'Dust absorption model',
                      # bbox_to_anchor=(1.04, 0.1),
                      loc=1)
axes.add_artist(legend22)
axes.add_artist(legend33)

# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
# plt.savefig('outputs/luminosities_sfrFinke22A' + '.png')

plt.show()
