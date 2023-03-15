# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.EBL_measurs_plot import plot_ebl_measurement_collection

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

from ebl_codes.EBL_class import EBL_model

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=24)
plt.rc('axes', titlesize=30)
plt.rc('axes', labelsize=30)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('legend', fontsize=30)
plt.rc('figure', titlesize=24)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True)
plt.rc('ytick.major', size=10, width=2, right=True)
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


config_data = read_config_file('scripts/input_data.yml')
ebl_class = input_yaml_data_into_class(config_data, log_prints=True)

# FIGURE: MEASUREMENTS
fig = plt.figure(figsize=(27, 15))
axes = fig.gca()

plot_ebl_measurement_collection('ebl_measurements/EBL_measurements.yml')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ sr)')

legend11 = plt.legend(bbox_to_anchor=(1.04, 1),
                      loc="upper left", title=r'Measurements')
axes.add_artist(legend11)

plt.xlim([.1, 10])
plt.ylim(1, 100)
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)

plt.savefig('outputs/ebl_measurements')

plt.figure()
x_sfr = np.linspace(0, 10)
m1 = [0.015, 2.7, 2.9, 5.6]
sfr = (lambda mi, x: eval(
    'lambda ci, x : ci[0] * (1 + x)**ci[1]'
    ' / (1 + ((1+x)/ci[2])**ci[3])')(mi, x))
plt.plot(x_sfr, sfr(m1, x_sfr), color='green', label='Madau&Dickinson +\'14')

plt.yscale('log')
plt.xlabel('z')
plt.ylabel('sfr(z)')

plt.xlim(0, 10)
plt.legend()

plt.show()
