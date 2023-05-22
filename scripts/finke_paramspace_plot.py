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

direct_name = 'outputs/chi2_params 2023-05-17 19:42:59'

params = np.load(direct_name + '/axion_params.npy')
axion_mac2 = params[:, 0]
axion_gay = params[:, 1]

values_gay_array = np.load(direct_name + '/params_UL.npy')
values_gay_array_NH = np.load(direct_name + '/params_measur.npy')


fig_params = plt.figure(figsize=(15, 10))
axes_params = fig_params.gca()
plt.title('Finke 2022 model A')
plt.xscale('log')
plt.yscale('log')
bbb = plt.pcolor(axion_mac2, axion_gay,
                 (values_gay_array.T - np.min(values_gay_array)),
                 vmin=0., vmax=10., #rasterized=True,
                 cmap='Oranges', shading='auto'
                 )
aaa = plt.contour(axion_mac2, axion_gay,
                  (values_gay_array.T - np.min(values_gay_array)),
                  levels=[2.30, 5.99],
                  origin='lower',
                  colors=('r', 'cyan'))
plt.clabel(aaa, inline=True, fontsize=16, levels=[2.30, 5.99],
           fmt={2.30: r'69%', 5.99: r'95%'})
cbar = plt.colorbar(bbb)
cbar.set_label(r'$\Delta\chi^2_{total}$')

# NH
values = values_gay_array_NH.T - np.min(values_gay_array_NH)
alpha_grid = (43. - values) / 43.
alpha_grid = alpha_grid * 0.7
alpha_grid = alpha_grid * (values <= 43.)
bbb = plt.pcolor(axion_mac2, axion_gay,
                 values,
                 vmin=0., vmax=100., rasterized=True,
                 alpha=alpha_grid, cmap='bone', shading='auto'
                 )
aaa = plt.contour(axion_mac2, axion_gay,
                  values,
                  levels=[2.30, 5.99],
                  origin='lower',
                  colors=('r', 'cyan'))
plt.clabel(aaa, inline=True, fontsize=16, levels=[2.30, 5.99],
           fmt={2.30: r'69%', 5.99: r'95%'})
# cbar = plt.colorbar(bbb)
# cbar.set_label(r'$\Delta\chi^2_{total}$')
plt.xlabel(r'm$_a\,$c$^2$ [eV]')
plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')
plt.savefig(direct_name + '/param_space.png')
plt.savefig(direct_name + '/param_space.pdf')
plt.xlim(2, 30)
plt.ylim(2e-11, 5e-10)
plt.savefig(direct_name + '/param_space_zoom.png')
plt.savefig(direct_name + '/param_space_zoom.pdf')
# plt.close('all')
plt.show()
