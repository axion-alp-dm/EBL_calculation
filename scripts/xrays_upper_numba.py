# IMPORTS --------------------------------------------#
import os
import time
import numpy as np
import numba.types as ty
from numba import jit, njit, prange, float64
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson

from data.cb_measurs.import_cb_measurs import import_cb_data

from astropy import units as u
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM

# from ebltable.ebl_from_model import EBL

from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm


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

cosmo_0dm = cosmo.Odm(0.)
cosmo_critden0 = cosmo.critical_density0.value




waves_ebl = np.geomspace(5e-6, 3e-3, num=int(1e6))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))


# We introduce the Finke22 and CUBA splines
# ebl = {}
# for m in EBL.get_models():
#     ebl[m] = EBL.readmodel(m)
# nuInu = {}
# for m, e in ebl.items():
#     nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
# spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)(waves_ebl)
# aaa = np.column_stack((waves_ebl, spline_cuba))
# np.savetxt('cuba.txt', aaa)
spline_cuba = np.loadtxt('data/cuba.txt')[:, 1]

axion_mass_array = np.geomspace(5e2, 1e6, num=800)
axion_gayy_array = np.geomspace(1.e-20, 3e-15, num=500)


# print(2.48/axion_mass_array)

np.save('outputs/' + direct_name + '/axion_mass', axion_mass_array)
np.save('outputs/' + direct_name + '/axion_gayy', axion_gayy_array)

# Beginning of figure specifications
plt.figure(figsize=(16, 10))  # figsize=(16, 10))
ax1 = plt.gca()

# We introduce all the EBL measurements
upper_lims_all, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=3.e-3,
    ax1=ax1, plot_measurs=True)

upper_lims_all.sort('lambda')
upper_lims_all['x_min'] = np.zeros(len(upper_lims_all))
upper_lims_all['x_max'] = np.zeros(len(upper_lims_all))

for refi in np.unique(upper_lims_all['ref']):
    ind_args = upper_lims_all['ref'] == refi
    data_ind = upper_lims_all[ind_args]['lambda']

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

    upper_lims_all['x_min'][ind_args] = data_ind - err_neg
    upper_lims_all['x_max'][ind_args] = err_pos + data_ind

    plt.errorbar(
        data_ind,
        upper_lims_all[ind_args]['nuInu'],
        yerr=upper_lims_all[ind_args]['1 sigma'],
        xerr=(err_neg, err_pos),
        marker='+', ls='')

    # for i in range(len(data_ind) - 1):
    #     plt.axvline(np.sqrt(data_ind[i] * data_ind[i + 1]), c='k', alpha=0.3)

    plt.hlines(0.9 * upper_lims_all[ind_args]['nuInu'],
               xmin=data_ind - err_neg, xmax=err_pos + data_ind,
               colors='k')

P = np.zeros((len(waves_ebl), len(upper_lims_all)))

for ni, i in enumerate(upper_lims_all):
    P[:, ni] = ((waves_ebl > upper_lims_all['x_min'][ni])
                * (waves_ebl < upper_lims_all['x_max'][ni]))

P_l = np.trapz(y=P / waves_ebl[:, np.newaxis], x=waves_ebl, axis=0)

upper_lims_all = np.array(upper_lims_all)

# plt.figure()
# plt.imshow(P, aspect='auto')

time_init = time.process_time()


#@jit
def cosmic_axion_contr(lmbd, mass, gayy):
    axion_mass = mass
    axion_gayy = gayy

    freq = 3e8 / lmbd * 1e6

    z_star = (axion_mass / (2. * 4.1356677e-15 * freq) - 1.)

    H_z = h * 100 * np.sqrt(omegaM * (1+z_star)**3. + 1. - omegaM)
    # print(axion_mass, axion_gayy, freq[0], z_star[0], H_z[0])

    ebl_axion_cube = (
            30856775814913.668 *
            (cosmo_0dm * cosmo_critden0
              * (3e8) ** 3. / (64. * np.pi)
              * axion_gayy ** 2. * axion_mass ** 2.
              * freq
              / H_z
             )
            * (z_star > 0.))
    # print(ebl_axion_cube[0])
    return ebl_axion_cube


# D_factor = 2.20656e22 * u.GeV * u.cm ** -2
D_factor = 1.11e22 # * u.GeV * u.cm ** -2


#@jit
def host_axion_contr(xx, mass, gay, v_dispersion=220.):
    sigma = (2. * 2.48 / mass * (v_dispersion / 3e5))
    nuInu_values = (
            14.53 * mass ** 3. * (gay * 1e10) ** 2.
            * 1.
            * (v_dispersion / 220.) ** -1
            * np.exp(-0.5 * ((xx - 2.48 / mass) / sigma) ** 2.))

    return nuInu_values


# #@jit(locals={'mean_flux':ty.float64})
#@jit(forceobj=False)
def calculate_chi2(aa, bb):
    # print(aa, bb)
    intensity_points = (
            (host_axion_contr(waves_ebl, aa, bb, v_dispersion=220)
             + cosmic_axion_contr(waves_ebl, aa, bb)
             + spline_cuba
             )
            * waves_ebl * 1e-6 / 3e8)
    plt.plot(waves_ebl, (
            # host_axion_contr(waves_ebl, aa, bb, v_dispersion=220)
             + cosmic_axion_contr(waves_ebl, aa, bb)
             + spline_cuba
             ))

    Pf_l = P * (intensity_points / waves_ebl)[:, np.newaxis]
    # Pf_l = Pf_l[:, 0]
    # Pf_l = simpson(y=Pf_l, x=waves_ebl, axis=0)
    Pf_l = np.trapz(y=Pf_l.T, x=waves_ebl)
    # print('Pf_l', Pf_l[0])

    Pf = P * intensity_points[:, np.newaxis]
    # Pf = Pf[:, 0]
    Pf = np.trapz(y=Pf.T, x=waves_ebl)
    # print('Pf', Pf[0])
    # Pf = simpson(y=Pf, x=waves_ebl, axis=0)

    mean_lambda = Pf / Pf_l
    mean_flux = Pf_l / P_l * 3e8 / mean_lambda * 1e6
    # print(mean_lambda[0], mean_flux[0])

    values_gay_array_NH = (
        sum(((upper_lims_all['nuInu']
              * upper_lims_all['lambda'] / mean_lambda
              - mean_flux)
             / upper_lims_all['1 sigma']
             * upper_lims_all['lambda'] / mean_lambda) ** 2.
            * (upper_lims_all['nuInu'] < mean_flux)))

    # print('upper_lims_all[nuInu', upper_lims_all['nuInu'])
    # print((((upper_lims_all['nuInu']
    #           * upper_lims_all['lambda'] / mean_lambda
    #           - mean_flux)
    #          / upper_lims_all['1 sigma']
    #          * upper_lims_all['lambda'] / mean_lambda) ** 2.
    #         * (upper_lims_all['nuInu'] < mean_flux)))
    # print(values_gay_array_NH)
    # print()

    return values_gay_array_NH


#@jit(parallel=True, forceobj=False)
def calculate_all():

    values_gay_array_NH = np.zeros(
        (len(axion_mass_array), len(axion_gayy_array)))

    for na in prange(len(axion_mass_array)):
        if na % 25 == 0:
            print('mass ', na)
            # time_init = time.process_time()

        for nb in prange(len(axion_gayy_array)):
            # if nb % 2 == 0:
            #     print('gay ', nb)
            # time_init = time.process_time()

            # print(na, nb, values_gay_array_NH[na, nb])
            values_gay_array_NH[na, nb] = calculate_chi2(
                float64(axion_mass_array[na]), axion_gayy_array[nb])
            # print(na, nb, values_gay_array_NH[na, nb])
            # print(values_gay_array_NH)

    return values_gay_array_NH

# values_gay_array_NH = np.zeros(
#         (len(axion_mass_array), len(axion_gayy_array)))
values_chi2_array = calculate_all()
np.save('outputs/' + direct_name + '/CUBA_params_Xrays',
        values_chi2_array)

print(np.max(values_chi2_array))
print(values_chi2_array)

plt.xlim(5e-6, 3e-3)
plt.ylim(5e-7, 5e-1)

plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.contour(axion_mass_array, axion_gayy_array,
            values_chi2_array.T - np.min(values_chi2_array),
            levels=[5.99], origin='lower',
            colors='cyan', zorder=1e10, linewidths=10, alpha=0.9)
nh_contours = plt.contour(axion_mass_array, axion_gayy_array,
                          values_chi2_array.T - np.min(values_chi2_array),
                          levels=[2.30], origin='lower',
                          colors='r', zorder=1e10, linewidths=4, alpha=0.9)
plt.xscale('log')
plt.yscale('log')

plt.show()
