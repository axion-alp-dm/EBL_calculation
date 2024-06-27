# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebltable.ebl_from_model import EBL

from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
import matplotlib as mpl


def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index = True
            else:
                use_index = False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, N))
        return cycler("color", colors)


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

direct_name = str('xrays_zoom')
print(direct_name)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

h = 0.7
omegaM = 0.3
omegaBar = 0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                      Ob0=omegaBar, Tcmb0=2.7255)


def const_bp(xx, xmin, xmax):
    return np.ones(len(xx)) * (xx > xmin) * (xx < xmax)


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


def cosmic_axion_contr(lmbd, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = c.value / lmbd * 1e6 * u.s ** -1

    z_star = (axion_mass / (2. * h_plank.to(u.eV * u.s) * freq)
              - 1.)
    # print(axion_mass, axion_gayy, freq[0], z_star[0], cosmo.H(z_star[0]))

    ebl_axion_cube = (
            ((cosmo.Odm(0.) * cosmo.critical_density0
              * c ** 3. / (64. * np.pi * u.sr)
              * axion_gayy ** 2. * axion_mass ** 2.
              * freq
              / cosmo.H(z_star)
              ).to(u.nW * u.m ** -2 * u.sr ** -1)
             ).value
            * (z_star > 0.))
    # print(ebl_axion_cube[0])
    return ebl_axion_cube


# D_factor = 2.20656e22 * u.GeV * u.cm ** -2
D_factor = 1.11e22 * u.GeV * u.cm ** -2


def host_axion_contr(xx, mass, gay, v_dispersion=220.):
    sigma = (2. * 2.48 / mass
             * (v_dispersion * u.km * u.s ** -1 / c).to(1))
    nuInu_values = (
            14.53 * mass ** 3. * (gay * 1e10) ** 2.
            * (D_factor / (1.11e22 * u.GeV * u.cm ** -2)).to(1)
            * (v_dispersion / 220.) ** -1
            * np.exp(-0.5 * ((xx - 2.48 / mass) / sigma) ** 2.))

    return nuInu_values


waves_ebl = np.geomspace(5e-6, 2e-2, num=int(1e6))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)

axion_mass_array = np.geomspace(5e2, 1e5, num=2)
axion_gayy_array = np.geomspace(1.e-19, 3e-15, num=2)

values_gay_array_NH = np.zeros(
    (len(axion_mass_array), len(axion_gayy_array)))
values_gay_array_NH2 = np.zeros(
    (len(axion_mass_array), len(axion_gayy_array)))

# We introduce all the EBL measurements
plt.figure(figsize=(16, 10))  # figsize=(16, 10))
ax1 = plt.gca()
upper_lims_cxb, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=3.e-3,
    plot_measurs=True, ax1=ax1)

upper_lims_cub, _ = import_cb_data(
    lambda_min_total=3.e-3,
    lambda_max_total=0.1,
    ax1=ax1, plot_measurs=True)

upper_lims_cxb.sort('lambda')

upper_lims_cxb['x_min'] = np.zeros(len(upper_lims_cxb))
upper_lims_cxb['x_max'] = np.zeros(len(upper_lims_cxb))

for refi in np.unique(upper_lims_cxb['ref']):
    ind_args = upper_lims_cxb['ref'] == refi
    data_ind = upper_lims_cxb[ind_args]['lambda']

    mean_points = np.sqrt(data_ind[1:] * data_ind[:-1])

    err_pos = abs(np.append(mean_points,
                            np.sqrt(data_ind[-2] * data_ind[-1]))
                  - data_ind)
    err_neg = abs(data_ind
                  - np.insert(mean_points, 0,
                              np.sqrt(data_ind[0] * data_ind[1]))
                  )
    if refi == '$\mathrm{Chandra \ (Cappelluti \ et \ al. \ 2017)}$':
        err_neg[7] = err_pos[7]
        err_pos[6] = err_neg[6]

    upper_lims_cxb['x_min'][ind_args] = data_ind - err_neg
    upper_lims_cxb['x_max'][ind_args] = err_pos + data_ind

    plt.hlines(0.9 * upper_lims_cxb[ind_args]['nuInu'],
               xmin=data_ind - err_neg, xmax=err_pos + data_ind,
               colors='k')

P = np.zeros((len(waves_ebl), len(upper_lims_cxb)), dtype=bool)

for ni, i in enumerate(upper_lims_cxb):
    P[:, ni] = ((waves_ebl > upper_lims_cxb['x_min'][ni])
                & (waves_ebl < upper_lims_cxb['x_max'][ni]))

P_den = -(upper_lims_cxb['x_min']
          - upper_lims_cxb['x_max'])
time_init = time.process_time()

for na, aa in enumerate(axion_mass_array):

    for nb, bb in enumerate(axion_gayy_array):

        intensity_points = ((
                host_axion_contr(waves_ebl, aa, bb,
                                 v_dispersion=220)
                + cosmic_axion_contr(waves_ebl, aa, bb)
                + spline_cuba(waves_ebl)
        ))

        intensity_points2 = ((
            # host_axion_contr(waves_ebl, aa, bb,
            #                  v_dispersion=220)
                + cosmic_axion_contr(waves_ebl, aa, bb)
                + spline_cuba(waves_ebl)
        ))
        print(max(host_axion_contr(waves_ebl, aa, bb,
                                 v_dispersion=220)))

        # plt.plot(waves_ebl, intensity_points)



        for i in range(len(upper_lims_cxb)):
            minn = np.where((waves_ebl > upper_lims_cxb['x_min'][i]))[0][0]
            maxx = np.where(waves_ebl < upper_lims_cxb['x_max'][i])[0][-1]

            intt = simpson(y=intensity_points[minn:maxx],
                           x=waves_ebl[minn:maxx])

            mean_flux = intt / P_den[i]

            values_gay_array_NH[na, nb] += (
                (((upper_lims_cxb['nuInu'][i] - mean_flux)
                  / upper_lims_cxb['1 sigma'][i]) ** 2.
                 * (upper_lims_cxb['nuInu'][i] < mean_flux)
                 ))

            intt2 = simpson(y=intensity_points2[minn:maxx],
                            x=waves_ebl[minn:maxx])

            mean_flux2 = intt2 / P_den[i]

            values_gay_array_NH2[na, nb] += (
                (((upper_lims_cxb['nuInu'][i] - mean_flux2)
                  / upper_lims_cxb['1 sigma'][i]) ** 2.
                 * (upper_lims_cxb['nuInu'][i] < mean_flux2)
                 ))
        print(aa, bb)
        print(values_gay_array_NH[na, nb], values_gay_array_NH2[na, nb],
              values_gay_array_NH[na, nb] - values_gay_array_NH2[na, nb])
        print()
plt.show()