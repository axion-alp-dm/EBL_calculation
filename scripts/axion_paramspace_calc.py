# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.import_cb_measurs import import_cb_data

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank

from ebltable.ebl_from_model import EBL

all_size = 24
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
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
direct_name = str('new_axionparams'
                  + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# Choose the max and minimum wavelengthS of the data that we import
lambda_min_total = 0.05  # [microns]
lambda_max_total = 5.  # [microns]

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

# Parameter space for axion characteristics and rest of necessary arrays
axion_mac2 = np.geomspace(0.3, 30., num=500)
axion_gay = np.geomspace(1e-12, 1e-7, num=500)
np.save('outputs/' + direct_name + '/axion_mass', axion_mac2)
np.save('outputs/' + direct_name + '/axion_gayy', axion_gay)

waves_ebl = np.geomspace(0.01, 10, num=int(1e4))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


def spline_pegase0001(lambda_array):
    method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                          grid=False)


def spline_starburst(lambda_array):
    method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                          grid=False)


list_working_models = {
    # 'ModelA': {'label': 'Model A', 'callable_func': spline_starburst,
    #            'color': 'b', 'linewidth': 3},
    'ModelB': {'label': 'Model B', 'callable_func': spline_pegase0001,
               'color': 'tab:orange', 'linewidth': 3},
    'Finke22': {'label': 'Finke+22', 'callable_func': spline_finke,
                'color': 'magenta', 'linewidth': 2},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
             'color': 'k', 'linewidth': 2}
}


def chi2_upperlims(x_model, x_obs, err_obs):
    """

    :param x_model:
    :param x_obs:
    :param err_obs:
    :return:
    """
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


# Configuration of EBL class from input yaml file
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


config_data = read_config_file(
    'scripts/input_files/input_data_paper.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)

upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=None, plot_measurs=False)

aaa = []
for working_model_name in list_working_models.keys():
    print(working_model_name)

    working_model = list_working_models[
        working_model_name]['callable_func']

    values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))
    values_gay_array_NH = np.zeros((len(axion_mac2), len(axion_gay)))
    values_nuInu = np.zeros((len(axion_mac2), len(axion_gay)))

    for na, aa in enumerate(axion_mac2):
        if na % 25 == 0:
            print(na)

        for nb, bb in enumerate(axion_gay):
            ebl_class.ebl_axion_calculation(aa, bb)

            values_gay_array[na, nb] += 2. * chi2_upperlims(
                x_model=(10 ** ebl_class.ebl_axion_spline(
                    np.log10(c.value / (upper_lims_all['lambda'] * 1e-6)),
                    0.,
                    grid=False)
                         + working_model(upper_lims_all['lambda'])),
                x_obs=upper_lims_all['nuInu'],
                err_obs=upper_lims_all['1 sigma'])

            values_gay_array_NH[na, nb] += (
                    ((16.7 - (10 ** ebl_class.ebl_axion_spline(
                        np.log10(c.value / (0.608 * 1e-6)), 0.,
                        grid=False)
                              + working_model(0.608))
                      ) / 1.47) ** 2.)

            values_nuInu[na, nb] += (10 ** ebl_class.ebl_axion_spline(
                np.log10(c.value / (0.608 * 1e-6)), 0., grid=False)
                                     + working_model(0.608))

    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_nuInu', values_nuInu)
    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_UL', values_gay_array)
    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_measur',
            values_gay_array_NH)
    aaa.append([working_model_name, list_working_models[
        working_model_name]['label']])
    print()

np.save('outputs/' + direct_name + '/list_models', aaa)
print(direct_name)
