# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from emissivity_data.emissivity_read_data import emissivity_data
from ebl_measurements.EBL_measurs_plot import plot_ebl_measurement_collection
from ebl_measurements.read_ebl_biteau import dictionary_datatype
from sfr_data.sfr_read import *

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from ebltable.ebl_from_model import EBL

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
# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
direct_name = str('chi2_params' +
                  time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime()))
if not os.path.exists("outputs/"):
    # if the directory for outputs is not present, create it.
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    # if the directory for outputs is not present, create it.
    os.makedirs('outputs/' + direct_name)
print(direct_name)


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
                     omegaM=float(
                         yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(
                         yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'],
                     log_prints=log_prints)


def chi2_upperlims(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


def chi2_measurs(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2.)


def gamma_from_rest(mass, gay):
    return (((mass * u.eV) ** 3. * (gay * u.GeV ** -1) ** 2.
             / 32. / h_plank.to(u.eV * u.s)).to(u.s ** -1).value)


config_data = read_config_file('scripts/input_data_iminuit_test.yml')
ebl_class = input_yaml_data_into_class(config_data)
#
# axion_mac2 = np.logspace(np.log10(2), np.log10(30), num=500)
# axion_gay = np.logspace(np.log10(2e-11), np.log10(5e-10), num=500)

axion_mac2 = np.logspace(-1, 4, num=500)
axion_gay = np.logspace(-12, -8, num=500)

np.save('outputs/' + direct_name + '/axion_params',
        np.column_stack((axion_mac2, axion_gay)))

upper_lims_ebldata = dictionary_datatype(
    'ebl_measurements/optical_data_2023', 'UL', lambda_max=5.)
upper_lims_ebldata_woNH = dictionary_datatype(
    'ebl_measurements/optical_data_2023', 'UL', lambda_max=5.,
    obs_not_taken=['lauer2022.ecsv'])

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)

values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))
values_gay_array_NH = np.zeros((len(axion_mac2), len(axion_gay)))

for na, aa in enumerate(axion_mac2):
    if na % 25 == 0:
        print(na)
    for nb, bb in enumerate(axion_gay):
        ebl_class.change_axion_contribution(
            aa, gamma_from_rest(aa, bb))
        values_gay_array[na, nb] += 2. * chi2_upperlims(
            x_model=(10 ** ebl_class.ebl_axion_spline(
                np.log10(3e8 / (upper_lims_ebldata['lambda'] * 1e-6)),
                0.,
                grid=False)
                     + spline_finke(upper_lims_ebldata['lambda'])),
            x_obs=upper_lims_ebldata['nuInu'],
            err_obs=((upper_lims_ebldata['nuInu_errp']
                      + upper_lims_ebldata['nuInu_errp']) / 2.))
        # values_gay_array_NH[na, nb] += 2. * chi2_upperlims(
        #     x_model=10 ** ebl_class.ebl_total_spline(
        #         np.log10(3e8 / (upper_lims_ebldata_woNH['lambda']
        #                         * 1e-6)),
        #         0.,
        #         grid=False),
        #     x_obs=upper_lims_ebldata_woNH['nuInu'],
        #     err_obs=(upper_lims_ebldata_woNH['nuInu_errn'] +
        #              upper_lims_ebldata_woNH['nuInu_errp']) / 2.)
        values_gay_array_NH[na, nb] += (
                ((16.7 - (10 ** ebl_class.ebl_axion_spline(
                    np.log10(3e8 / (0.608 * 1e-6)), 0.,
                    grid=False)
                          + spline_finke(0.608))
                  ) / 1.47) ** 2.)

np.save('outputs/' + direct_name + '/params_UL', values_gay_array)
np.save('outputs/' + direct_name + '/params_measur', values_gay_array_NH)
print(direct_name)
