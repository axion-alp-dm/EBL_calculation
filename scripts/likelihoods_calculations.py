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
direct_name = str('final_outputs_NOlims'
                  + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


def chi2_upperlims(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


def chi2_measurs(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2.)


waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))


config_data = read_config_file(
    'scripts/input_files/input_data_paper.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)


# COB measurements that we are going to use
upper_lims_ebldata, igl_ebldata = import_cb_data(
    lambda_min_total=0.1, lambda_max_total=5.,
    plot_measurs=False)

upper_lims_ebldata_woNH = upper_lims_ebldata[
    upper_lims_ebldata['ref'] != r'NH/LORRI (Lauer+ \'22)']


# FIGURE: sfr fit ------------------------------------------------
sfr_data = sfr_data_dict('sfr_data/')

# FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS ------------------

emiss_data = emissivity_data(directory='emissivity_data/')
freq_emiss = c.value / (emiss_data['lambda'] * 1e-6)


# MINIMIZATION OF CHI2 OF SSPs
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])


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


    combined_likelihood = (LeastSquares(igl_ebldata['lambda'],
                                        igl_ebldata['nuInu'],
                                        igl_ebldata['1 sigma'],
                                        fit_igl)
                           + LeastSquares((emiss_data['lambda'],
                                           emiss_data['z']),
                                          emiss_data['eje'],
                                          (emiss_data['eje_n']
                                           + emiss_data['eje_p']) / 2.,
                                          fit_emiss)
                           + LeastSquares(sfr_data[:, 0],
                                          sfr_data[:, 3],
                                          (sfr_data[:, 4]
                                           + sfr_data[:, 5]) / 2.,
                                          sfr))

    init_time = time.process_time()

    m = Minuit(combined_likelihood,
               config_data['ssp_models'][key]['sfr_params'])
    # m.limits = [[0.001, 0.02], [2., 3.5], [1., 5.5], [4., 8.]]
    print(m.params)

    m.migrad()  # finds minimum of least_squares function
    m.hesse()  # accurately computes uncertainties

    outputs_file = open('outputs/' + direct_name + '/z_fits_info', 'a+')
    outputs_file.write(str(key) + '\n')
    outputs_file.write('SSP model: '
                       + str(config_data['ssp_models'][key]['name'])
                       + '\n')
    outputs_file.write(str(m.params) + '\n')
    outputs_file.write(str(m.values) + '\n')
    outputs_file.write(str(m.covariance) + '\n')
    outputs_file.write('\n\n\n')
    outputs_file.close()

    print(m.params)
    print(m.values)
    print(m.covariance)

    print('Fit: %.2fs' % (time.process_time() - init_time))
    init_time = time.process_time()

    print(config_data['ssp_models'][key]['sfr_params'])

    config_data['ssp_models'][key]['sfr_params'] = [m.params[0].value,
                                                    m.params[1].value,
                                                    m.params[2].value,
                                                    m.params[3].value]
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

    np.save('outputs/' + direct_name + '/' + key + 'spline',
            ebl_class.ebl_ssp_spline)

    ccc = []
    for i in range(len(config_data['ssp_models'][key]['sfr_params']) ** 2):
        ccc.append(float(m.covariance.flatten()[i]))
    config_data['ssp_models'][key]['cov_matrix'] = ccc


outputs_file = open('outputs/' + direct_name + '/input_data.yml', 'w')
yaml.dump(config_data, outputs_file,
          default_flow_style=False, allow_unicode=True)
outputs_file.close()
