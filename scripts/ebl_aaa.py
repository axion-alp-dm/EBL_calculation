# IMPORTS --------------------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model

# Manuel s repository and code
from ebltable.ebl_from_model import EBL


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


config_data = read_config_file('scripts/input_data_iminuit_test.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)

# z_array = np.linspace(0, 40, num=200)
# plt.figure()
# plt.plot(z_array, ebl_class.sfr_function(
#     'lambda x, ci : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[2])**ci[3])',
#     z_array,
#     [0.015, 2.7, 2.9, 5.6]))
#
# sfr_capelluti = np.loadtxt('sfr_data/capelluti_sfr_sims.txt')
# plt.plot(sfr_capelluti[:, 0], sfr_capelluti[:, 1], '.')
#
# zmadau_array = np.linspace(0, sfr_capelluti[0, 0]-0.1, num=50)
# z_total = np.append(zmadau_array, sfr_capelluti[:, 0])
# sfr_total = np.append(ebl_class.sfr_function(
#     'lambda x, ci : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[2])**ci[3])',
#     zmadau_array,
#     [0.015, 2.7, 2.9, 5.6]), sfr_capelluti[:, 1])
#
# plt.plot(z_total, sfr_total)
# sfr_spline = UnivariateSpline(z_total, sfr_total, s=0, k=1)
#
# plt.yscale('log')


waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
# spline_cuba = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)

for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    ebl_class.ebl_axion_calculation(1.,  5e-23)
    ebl_class.ebl_sum_contributions()

    plt.figure()
    plt.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0.,
        grid=False))
    plt.plot(waves_ebl, 10**ebl_class.ebl_axion_spline(
        freq_array_ebl, 0.,
        grid=False))
    plt.plot(waves_ebl, 10**ebl_class.ebl_total_spline(
        freq_array_ebl, 0.,
        grid=False))
    plt.plot(waves_ebl, spline_finke(waves_ebl))

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    ebl_class.ebl_sum_contributions()

    plt.plot(waves_ebl, 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0.,
        grid=False), '--')

    plt.plot(waves_ebl, 10**ebl_class.ebl_total_spline(
        freq_array_ebl, 0.,
        grid=False), '--')

    plt.xscale('log')
    plt.yscale('log')
    plt.show()
