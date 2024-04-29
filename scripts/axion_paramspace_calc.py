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
from data.cb_measurs.import_cb_measurs import import_cb_data

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

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
direct_name = str('CIB_Finke'
                  # + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# Choose the max and minimum wavelengths of the data that we import
lambda_min_total = 0.1  # [microns]
lambda_max_total = 1500  # [microns]

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

# Parameter space for axion characteristics and rest of necessary arrays
axion_mac2 = np.geomspace(1e-3, 0.8, num=500)
axion_gay = np.geomspace(5e-10, 5e-5, num=500)
np.save('outputs/' + direct_name + '/axion_mass', axion_mac2)
np.save('outputs/' + direct_name + '/axion_gayy', axion_gay)

waves_ebl = np.geomspace(1., 1300, num=int(1e5))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
# spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


# We initialize the class with the input file
# config_data = read_config_file('outputs/final_outputs_Zevol_fixezZsolar '
#                        '2024-04-11 13:41:34/' + 'input_data.yml')
# ebl_class = EBL_model.input_yaml_data_into_class(config_data,
#                                                  log_prints=True)
# ebl_class.ebl_ssp_calculation(
#     config_data['ssp_models']['SB99_dustFinke'])
#
# def spline_starburst(lambda_array):
#     return 10 ** ebl_class.ebl_ssp_spline(
#         np.log10(c.value * 1e6 / lambda_array), 0.,
#                           grid=False)



list_working_models = {
    # 'ModelA': {'label': 'Our model', 'callable_func': spline_starburst,
    #            'color': 'b', 'linewidth': 3},
    'Finke22': {'label': 'Finke+22', 'callable_func': spline_finke,
                'color': 'magenta', 'linewidth': 2},
    # 'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
    #          'color': 'k', 'linewidth': 2}
}


def chi2_upperlims(x_model, x_obs, err_obs):
    """

    :param x_model:
    :param x_obs:
    :param err_obs:
    :return:
    """
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


h = 0.7
omegaM = 0.3
omegaBar = 0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                      Ob0=omegaBar, Tcmb0=2.7255)
def axion_contr(lmbd, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = (c / lmbd).to(u.s**-1)

    z_star = (axion_mass / (2. * h_plank.to(u.eV * u.s) * freq)
              - 1.)

    ebl_axion_cube = (
            ((cosmo.Odm(0.) * cosmo.critical_density0
              * c ** 3. / (64. * np.pi * u.sr)
              * axion_gayy ** 2. * axion_mass ** 2.
              * freq
              / cosmo.H(z_star)
              ).to(u.nW * u.m ** -2 * u.sr ** -1)
             ).value
            * (z_star > 0.))
    return ebl_axion_cube

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

    for na, aa in enumerate(axion_mac2):
        if na % 25 == 0:
            print(na)

        for nb, bb in enumerate(axion_gay):

            values_gay_array[na, nb] += chi2_upperlims(
                x_model=(axion_contr(upper_lims_all['lambda'],
                                     mass=aa, gayy=bb)
                         + working_model(upper_lims_all['lambda'])),
                x_obs=upper_lims_all['nuInu'],
                err_obs=upper_lims_all['1 sigma'])

    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_UL', values_gay_array)

    aaa.append([working_model_name, list_working_models[
        working_model_name]['label']])
    print()

np.save('outputs/' + direct_name + '/list_models', aaa)
print(direct_name)
