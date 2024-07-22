# IMPORTS -----------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebl_codes.EBL_class import EBL_model
from ebltable.ebl_from_model import EBL
from data.cb_measurs.import_cb_measurs import import_cb_data

all_size = 34
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
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

lorri_trans = np.loadtxt('data/lorri_qe_v2.txt')
lorri_trans[:, 0] = lorri_trans[:, 0] * 1e-3
lorri_trans[:, 1] = lorri_trans[:, 1]  # * 1e-2  # / max(lorri_trans[:, 1])
spline_lorri = UnivariateSpline(lorri_trans[:, 0], lorri_trans[:, 1],
                                s=0, k=1, ext=1)

waves_ebl = np.geomspace(0.05, 5., num=200)
freq_ebl = np.log10(c.value / waves_ebl * 1e6)

pivot_vw = (simpson(lorri_trans[:, 1] * lorri_trans[:, 0],
                    x=lorri_trans[:, 0])
            / simpson(lorri_trans[:, 1] / lorri_trans[:, 0],
                      x=lorri_trans[:, 0])
            ) ** 0.5
print('pivot wv: ', pivot_vw)

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


def avg_flux(f_nu, trans_spline, lambda_array):
    yyy = trans_spline(lambda_array) / lambda_array
    yyy_num = simpson(y=yyy * f_nu(lambda_array), x=lambda_array)
    yyy_den = simpson(y=yyy, x=lambda_array)
    return yyy_num / yyy_den


def avg_lmbd_v1(f_nu, trans_spline, lambda_array):
    yyy = trans_spline(lambda_array) * f_nu(lambda_array)
    yyy_num = simpson(y=yyy, x=lambda_array)
    yyy_den = simpson(y=yyy / lambda_array, x=lambda_array)
    return yyy_num / yyy_den


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml



config_data = read_config_file('outputs/4_fitted_models.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=False)


fig, ax = plt.subplots()

for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
    yyy = 10 ** ebl_class.ebl_ssp_spline(freq_ebl, 0., grid=False)

    mean_lmbd = avg_lmbd_v1(
        f_nu=UnivariateSpline(
            waves_ebl,
            yyy * waves_ebl * 1e-6 / c.value,
            s=0, k=1),
        trans_spline=spline_lorri,
        lambda_array=waves_ebl)
    mean_flux = (avg_flux(
              f_nu=UnivariateSpline(
                  waves_ebl,
                  yyy * pivot_vw * 1e-6 / c.value,
                  s=0, k=1),
              trans_spline=spline_lorri,
              lambda_array=waves_ebl)
          * c.value / mean_lmbd * 1e6)


    plt.plot(waves_ebl, yyy, c=plt.cm.CMRmap(nkey / 6.),
             label=config_data['ssp_models'][key]['name'])

    plt.errorbar(x=mean_lmbd, y=mean_flux,
                 linestyle='', color=plt.cm.CMRmap(nkey / 6.),
                 marker='d', alpha=0.8,
                 mfc='white',
                 markersize=20, markeredgewidth=2,
                 zorder=1e5
                 )
    plt.errorbar(x=mean_lmbd, y=mean_flux,
                 linestyle='', color='k',
                 marker='.',
                 mfc='k',
                 markersize=8, zorder=5e5
                 )

    print(config_data['ssp_models'][key]['name'], mean_lmbd, mean_flux)

# plt.xlim(0.25, 1.)

plt.legend()
plt.show()

