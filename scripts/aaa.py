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
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
import matplotlib as mpl
# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


waves = np.loadtxt('outputs/data_new_lowres.txt')[:, 0]
data_y_new = np.loadtxt('outputs/data_new_lowres.txt')[:, 1]

method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
data_y_old = 10 ** method_y(np.log10(c.value * 1e6 / waves), 0.,
                          grid=False)

method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
data_y_SB99 = 10 ** method_y(np.log10(c.value * 1e6 / waves), 0.,
                          grid=False)

method_y = np.load('outputs/final_outputs_Zevol 2024-03-12 17:19:00/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
data_y_newnew = 10 ** method_y(np.log10(c.value * 1e6 / waves), 0.,
                          grid=False)


fig, ax = plt.subplots(figsize=(12, 8))
plt.loglog(waves, data_y_SB99, label='SB99 Z=0.02')
plt.loglog(waves, data_y_old, label='pegase Z=0.0001')
plt.loglog(waves, (data_y_new), label='pegase Z=0.0001 with griddata '
                                      ' low res implementation')
plt.loglog(waves, (data_y_newnew), label='pegase metall evol')
# plt.loglog(waves, data_y_new/data_y_old)

plt.legend()
_, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax, plot_measurs=True)

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

plt.xlim(0.08, 1e1)
plt.ylim(0.8, 130)

plt.xscale('log')
plt.yscale('log')

plt.savefig('outputs/figures_paper/metall_comparison.png',
            bbox_inches='tight')
plt.show()
