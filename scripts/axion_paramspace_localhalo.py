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
direct_name = str('individuals'
                  # + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# Choose the max and minimum wavelengthS of the data that we import
lambda_min_total = 0.  # [microns]
lambda_max_total = 5.  # [microns]

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

def chi2_upperlims(x_model, x_obs, err_obs):
    """

    :param x_model:
    :param x_obs:
    :param err_obs:
    :return:
    """
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


waves_ebl = np.geomspace(5e-6, 10, num=int(1e6))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=None, plot_measurs=False)

chi2_min =  2. * chi2_upperlims(
                x_model=spline_cuba(upper_lims_all['lambda']),
                x_obs=upper_lims_all['nuInu'],
                err_obs=upper_lims_all['1 sigma'])
print(chi2_min)

chi2_delta = chi2_min + 4.61  # for upper limits

nuInu_extra = (upper_lims_all['nuInu']
               - spline_cuba(upper_lims_all['lambda'])
               + upper_lims_all['1 sigma'] * (chi2_delta/2.)**0.5)

print(upper_lims_all['nuInu'][:15])
print((upper_lims_all['1 sigma'] * (chi2_delta/2.)**0.5)[:11])
print(spline_cuba(upper_lims_all['lambda'])[:11])
print(nuInu_extra[:11])

g_ay_array = (1e-10 *
              (nuInu_extra
               / (5.35e-2 * (2.48/upper_lims_all['lambda'])**3.)
               ) ** 0.5
              )
print(g_ay_array[:11])
print(upper_lims_all['lambda'])
aaa = (np.column_stack((
    2.48/upper_lims_all['lambda'].value,
    upper_lims_all['nuInu'].value,
    (upper_lims_all['1 sigma'] * (chi2_delta/2.)**0.5).value,
    spline_cuba(upper_lims_all['lambda']),
nuInu_extra.value, g_ay_array.value
)))
for i in range(len(aaa)):
    print(aaa[i, :])
np.save('outputs/' + direct_name + '/dips', g_ay_array.value)
np.savetxt('outputs/' + direct_name + '/dips.txt',
           np.column_stack((2.48/upper_lims_all['lambda'],
                            g_ay_array.value)))
