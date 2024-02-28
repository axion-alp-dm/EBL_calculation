import os
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from ebl_codes.EBL_class import EBL_model
from ebl_measurements.import_cb_measurs import import_cb_data
from scipy.interpolate import UnivariateSpline
from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM
from ebltable.ebl_from_model import EBL
from scipy.optimize import newton
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml
config_data = read_config_file(
    'scripts/input_files/input_data_paper2.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)
waves_ebl = np.logspace(np.log10(5e-6), 1, 3000)
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
    ebl_class.ebl_ssp_calculation(
        yaml_data=config_data['ssp_models']['pegase0.0001_Finke'])
    return 10**ebl_class.ebl_ssp_spline(
        np.log10(c.value * 1e6 / lambda_array), 0., grid=False)

def spline_starburst(lambda_array):
    ebl_class.ebl_ssp_calculation(
        yaml_data=config_data['ssp_models']['SB99_kneiske'])
    return 10**ebl_class.ebl_ssp_spline(
        np.log10(c.value * 1e6 / lambda_array), 0., grid=False)

list_working_models = {
    'ModelA': {'label': 'Model A', 'callable_func': spline_starburst,
               'color': 'b', 'linewidth': 3},
    'ModelB': {'label': 'Model B', 'callable_func': spline_pegase0001,
               'color': 'tab:orange', 'linewidth': 3},
    'Finke22': {'label': 'Finke22', 'callable_func': spline_finke,
                'color': 'magenta', 'linewidth': 2},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
             'color': 'k', 'linewidth': 2}
}
h=0.7
omegaM=0.3
omegaBar=0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                                    Ob0=omegaBar, Tcmb0=2.7255)

axion_mass = 4.079 * u.eV

def cosmic_decay(gay):
    return (cosmo.Odm(0.) * cosmo.critical_density0*c**4.
                / (64. * np.pi * 0.608 * u.micron * cosmo.H(0.))
                * axion_mass**2. * (10**gay*u.GeV**-1)**2.
            ).to(u.nW * u.m **-2).value

def halo_decay(gay):
    return 14.53 * axion_mass.value**3. * (10**gay*1e10)**2.

def find_gay_sum(gay, excess):
    return cosmic_decay(gay) + halo_decay(gay) - excess

def find_gay_cosmic(gay, excess):
    return cosmic_decay(gay) - excess

def find_gay_halo(gay, excess):
    return halo_decay(gay) - excess

dict_gays = {}
for ni, model in enumerate(list_working_models.keys()):
    print(model)
    model_full = list_working_models[model]

    excess = 16.37 - model_full['callable_func'](0.608)
    print(model_full['callable_func'](0.608), excess)


    print(10**newton(find_gay_cosmic, x0=-9, args=[excess]))
    print(10**newton(find_gay_halo, x0=-9, args=[excess]))
    print(10**newton(find_gay_sum, x0=-9, args=[excess]))
    print()
    dict_gays[model] = np.array([10**newton(find_gay_cosmic, x0=-9,
                                         args=[excess]),
    10**newton(find_gay_halo, x0=-9, args=[excess]),
    10**newton(find_gay_sum, x0=-9, args=[excess])])

modelB_value = 8.056225981654698

for ni, model in enumerate(dict_gays.keys()):
    print(dict_gays[model]/dict_gays['ModelB'])
    print(modelB_value + find_gay_cosmic(np.log10(dict_gays[model][0]), excess=0.),
          modelB_value + find_gay_halo(np.log10(dict_gays[model][1]), excess=0.),
          modelB_value + find_gay_sum(np.log10(dict_gays[model][2]), excess=0.))
    print()
print(16.37 - 1.47, 16.37 + 1.47)
