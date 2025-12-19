# IMPORTS --------------------------------------------#
import os
import yaml
import time
import sys
import numpy as np

print(sys.path)
sys.path.append('/home/porrassa/Desktop/EBL_ModelCode/EBL_calculation/')

from ebl_codes.sfr_models import sfr_model
from ebl_codes.metall_models import metall_model
from ebl_codes.EBL_class import EBL_model

from data.emissivity_measurs.emissivity_read_data import emissivity_data
from data.cb_measurs.import_cb_measurs import import_cb_data
from data.sfr_measurs.sfr_read import *
from data.metallicity_measurs.import_metall import import_met_data

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from scipy.interpolate import RectBivariateSpline

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


def chi2_measurs(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2.)


direct_name = ('outputs/outputs_dust_final_new/')
print(direct_name)

# Configuration file reading and data input/output ---------#
with open(direct_name + '/input_data.yml', 'r') as file:
    config_data = yaml.safe_load(file)

ebl_class = EBL_model.input_yaml_data_into_class(config_data)

# COB measurements that we are going to use
upper_lims_ebldata, igl_ebldata = import_cb_data(
    lambda_min_total=0.1, lambda_max_total=1e5,
    plot_measurs=False)

igl_ebldata = igl_ebldata[igl_ebldata['ref'] != 'ISO/ISOCAM (Clements+ ‘99)']
igl_ebldata = igl_ebldata[igl_ebldata['ref'] != 'SCUBA-2 (Hsu+ ‘16)']
igl_ebldata = igl_ebldata[igl_ebldata['ref'] != 'ALMA (Fujimoto+ ‘16)']

print(np.shape(igl_ebldata))

# Metallicity evolution data
z_data = import_met_data()
print(np.shape(z_data))

# FIGURE: sfr fit ------------------------------------------------
sfr_data = sfr_data_dict()
print(np.shape(sfr_data))
# FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS ------------------

emiss_data = emissivity_data(lambda_max=700)
# freq_emiss = c.value / (emiss_data['lambda'] * 1e-6)
print(np.shape(emiss_data))


ebl_f = EBL.readmodel('finke2022')
print(
    'CB data finke2022: ' + str(chi2_measurs(
        ebl_f.ebl_array(z=0., lmu=igl_ebldata['lambda']),
        igl_ebldata['nuInu'], igl_ebldata['1 sigma'])))

ebl_f = EBL.readmodel('saldana-lopez')
print(
    'CB saldana-lopez: ' + str(chi2_measurs(
        ebl_f.ebl_array(z=0., lmu=igl_ebldata['lambda']),
        igl_ebldata['nuInu'], igl_ebldata['1 sigma'])))


my_ebl = ['bosa.txt', 'chary.txt', '2bb.txt']

for d in my_ebl:
    ebl_finke = EBL.readascii(
            'outputs/lhaaso_new/' + d, model_name='mine')
    print(
        'CB ' + d + ' : ' + str(chi2_measurs(
            ebl_finke.ebl_array(z=0., lmu=igl_ebldata['lambda']),
            igl_ebldata['nuInu'], igl_ebldata['1 sigma'])))


list_zz_finke = os.listdir('/home/porrassa/Downloads/lumdens/')

zz_bare = []
for i in list_zz_finke:
    i = i.replace('.dat', '')
    i = i.replace('lumdens_total_z', '')
    zz_bare.append(i)

zz_bare = np.array(zz_bare, dtype=str)
zz_floats = np.array(zz_bare, dtype=float)

zz_order = np.argsort(zz_floats)
zz_floats = zz_floats[zz_order]
zz_bare = zz_bare[zz_order]

wavelengths = np.loadtxt(
    '/home/porrassa/Downloads/lumdens/lumdens_total_z0.00.dat')
wavelengths = wavelengths[:, 0]

array_lumin = np.zeros((len(wavelengths), len(zz_bare)))

for ni, ii in enumerate(zz_bare):
    data = np.loadtxt(
    '/home/porrassa/Downloads/lumdens/lumdens_total_z' + ii + '.dat')
    array_lumin[:, ni] = data[:, 1]

spline_emiss_finke = RectBivariateSpline(
    x=np.log10(wavelengths), y=zz_floats, z=array_lumin,
    kx=1, ky=1, s=0)

print('emissivities finke2022: ' + str(chi2_measurs(
    spline_emiss_finke(x=np.log10(emiss_data['lambda']),
                       y=emiss_data['z'], grid=False),
    emiss_data['eje'],
    (emiss_data['eje_n'] + emiss_data['eje_p']) / 2.))
)


#     outputs.write(
#         'sfr data: ' + str(chi2_measurs(
#             sfr(sfr_data[:, 0], aaa),
#             sfr_data[:, 3], (sfr_data[:, 4] + sfr_data[:, 5]) / 2.))
#         + '\n')
# chi2_measurs(
#             metall(z_data[:, 0], aaa),
#             z_data[:, 1], (z_data[:, 2] + z_data[:, 3]) / 2.)

