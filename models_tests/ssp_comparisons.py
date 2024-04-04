import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from astropy import units as u

from scipy.interpolate import UnivariateSpline, RectBivariateSpline, \
    interpn, RegularGridInterpolator
from astropy.cosmology import FlatLambdaCDM, z_at_value

from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm

from scipy.interpolate import RectBivariateSpline

# import matplotlib.pylab as pl
# import matplotlib as mpl

all_size = 30
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

bi = LinearSegmentedColormap.from_list("",
                                       ["#ff0080", "#ff0080",
                                        "#a349a4", "#0000ff",
                                        "#0000ff"])
bi_r = LinearSegmentedColormap.from_list("",
                                         ["#0000ff", "#0000ff",
                                          "#a349a4", "#990099",
                                          "#ff0080",
                                          "#ff0080"])  # reversed

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


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


cividis_mine = mpl.colormaps['cividis'].resampled(10)
cividis_mine.colors[-1, 1] = 0.7
cividis_mine.colors[-1, 2] = 0.
cividis_mine.colors = cividis_mine.colors[::-1]
print(cividis_mine.colors)
N = 10
plt.rcParams["axes.prop_cycle"] = get_cycle(cividis_mine, N)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'models_tests':
    os.chdir("..")

work_with_Lnu = True

# Starburst 99 ------------------------------------------------------------
data_starburst_old = np.loadtxt(
    'data/ssp_synthetic_spectra/final_run_spectrum', skiprows=6)
t_old = np.unique(data_starburst_old[:, 0])
l_old = np.unique(data_starburst_old[:, 1])
dd_old = data_starburst_old[:, 3].reshape(t_old.shape[0], l_old.shape[0]).T

data_starburst = np.loadtxt(
    'data/ssp_synthetic_spectra/low_res_for_real.spectrum1', skiprows=6)
t = np.unique(data_starburst[:, 0])
l = np.unique(data_starburst[:, 1])
dd = data_starburst[:, 3].reshape(t.shape[0], l.shape[0]).T

st99_log_emis = np.zeros((len(l), len(t) + sum(t_old > t[-1])))
st99_t_total = np.zeros(len(t) + sum(t_old > t[-1]))
aaa = np.where((t_old - t[-1]) > 0)[0][0]

st99_t_total[:len(t)] = t
st99_t_total[len(t):] = t_old[aaa:]

st99_log_emis[:, :len(t)] = dd
st99_log_emis[:, len(t):] = dd_old[:, aaa:]

st99_log_time = np.log10(st99_t_total)  # log(time/yrs)
st99_wave = l  # amstrongs
st99_log_emis += - 6.  # - np.log10(3.828e33)

if work_with_Lnu is True:
    st99_log_emis += (np.log10(1E10 * c)
                      - 2. * np.log10(c * 1e10 / st99_wave
                                      )[:, np.newaxis])


# ----------------------------------------------------------------------
# STARBURST WITH METALLICITY EVOLUTION
def load_sb99(path_ssp, pop_filename):
    ssp_metall = np.sort(np.array(os.listdir(path_ssp),
                                  dtype=float))
    print(ssp_metall)
    d = np.loadtxt(path_ssp + '/0.004/kroupa_004.spectrum1', skiprows=6)

    # Get unique time steps and frequencies, and spectral data
    t_total = np.unique(d[:, 0])
    l_total = np.unique(d[:, 1])
    print(len(t_total), np.log10(t_total[0]), np.log10(t_total[-1]))
    print(len(l_total), np.log10(l_total[0]), np.log10(l_total[-1]))

    dd_total = np.zeros((l_total.shape[0],
                         t_total.shape[0],
                         len(ssp_metall) + 1))

    for n_met, met in enumerate(ssp_metall):
        data = np.loadtxt(
            path_ssp + str(met) + '/' + pop_filename
            + str(met).replace('0.', '')
            + '.spectrum1',
            skiprows=6)

        dd_total[:, :, n_met + 1] = data[:, 2].reshape(
            t_total.shape[0],
            l_total.shape[0]).T

    ssp_metall = np.insert(ssp_metall, 0, 1e-43)
    print(ssp_metall)
    dd_total[:, :, 0] = dd_total[:, :, 1]

    # Define the quantities we will work with
    sb99_log_time = np.log10(t_total)  # log(time/yrs)
    sb99_log_freq = np.log10(  # log(frequency/Hz)
        c / l_total[::-1] / 1E-10)
    ssp_log_emis = (dd_total[::-1]  # log(em[erg/s/Hz/M_solar])
                                    # - 6.
                                    # + np.log10(1E10 * c)
                                    # - 2. * sb99_log_freq[:, np.newaxis,
                                    #        np.newaxis]
                                    )

    ssp_log_emis[np.isnan(ssp_log_emis)] = -43.
    ssp_log_emis[
        np.invert(np.isfinite(ssp_log_emis))] = -43.

    ssp_lumin_spline = RegularGridInterpolator(
        points=(sb99_log_freq,
                sb99_log_time,
                np.log10(ssp_metall)),
        values=ssp_log_emis,
        method='linear',
        bounds_error=False, fill_value=-1.
    )
    return ssp_lumin_spline


# Pegase ------------------------------------------------------------
pegase_metall = [0.1, 0.05, 0.02, 0.008, 0.004, 0.0004, 0.0001]

data_pegase = np.loadtxt(
    'data/ssp_synthetic_spectra/pegase3/spectra_BG03/'
    'spectral_resultsZ0.02.txt')
t_pegase = np.unique(data_pegase[:, 0])
l_pegase = np.unique(data_pegase[:, 1])

dd_pegase = np.zeros((l_pegase.shape[0],
                      t_pegase.shape[0],
                      len(pegase_metall)))

for n_met, met in enumerate(pegase_metall):
    data_pegase = np.loadtxt(
        'data/ssp_synthetic_spectra/pegase3/spectra_BG03/'
        'spectral_resultsZ'
        + str(met) + '.txt')
    dd_pegase[:, :, n_met] = data_pegase[:, 2].reshape(t_pegase.shape[0],
                                                       l_pegase.shape[0]).T
    print(n_met, met)

pegase_log_time = np.log10(t_pegase * 1e6)  # log(time/yrs)
pegase_wave = l_pegase  # amstrongs
pegase_log_emis = np.log10(dd_pegase)  # - np.log10(3.828e33)

if work_with_Lnu is True:
    pegase_log_emis += (np.log10(1E10 * c)
                        - 2. * np.log10(c * 1e10 / l_pegase
                                        )[:, np.newaxis, np.newaxis])


# PopStar 09 ------------------------------------------------------------

def popstar09(path, name):
    list_files = os.listdir(path)

    numbers = []
    for listt in list_files:
        numbers.append(float(listt.replace(name, '')))

    indexes = np.argsort(numbers)
    pop09_log_time = np.sort(numbers)
    pop09_wave = np.loadtxt(path09 + list_files[0])[:, 0]
    pop09_lumin_cube = np.zeros((len(pop09_wave), len(list_files)))

    x_is_1e4 = np.argmin(np.abs(pop09_wave - 1e4))
    cut = 5e27 / 3.828e33

    for nind, ind in enumerate(indexes):
        yyy = np.loadtxt(
            path09
            + name
            + str('%.2f' % numbers[ind])
        )[:, 1]

        if np.shape(np.where(yyy[:x_is_1e4] < cut))[1] == 0:
            min_x = 0
        else:
            min_x = np.where(yyy[:x_is_1e4] < cut)[0][-1]

        # pop09_lumin_cube[min_x:, nind] = yyy[min_x:]
        pop09_lumin_cube[:, nind] = yyy

    pop09_lumin_cube = np.log10(pop09_lumin_cube) + np.log10(3.828e33)
    pop09_lumin_cube[np.isnan(pop09_lumin_cube)] = -43.
    pop09_lumin_cube[np.invert(np.isfinite(pop09_lumin_cube))] = -43.

    if work_with_Lnu is True:
        pop09_lumin_cube += (np.log10(1E10 * c)
                             - 2. * np.log10(c * 1e10 / pop09_wave
                                             )[:, np.newaxis])

    plt.figure(figsize=(10, 8))
    plt.title('ssp')

    for age in [6.0, 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.]:
        aaa = np.abs(pop09_log_time - age).argmin()
        plt.plot(pop09_wave,
                 pop09_lumin_cube[:, aaa], '-',
                 label='log(t) = %.2f'
                       % pop09_log_time[aaa])

    plt.xscale('log')
    plt.xlabel('wavelenght [A]')
    plt.ylabel(r'log$_{10}$(L$_{\lambda}$/Lsun '
               r'[erg s$^{-1}$ A$^{-1}$ Msun$^{-1}$])')
    plt.legend()

    plt.axhline(cut, linestyle='dotted')

    plt.xlim(1e2, 1e6)
    plt.ylim(20, 34)

    pop09_wave = 10 ** ((np.log10(pop09_wave[1:])
                         + np.log10(pop09_wave[:-1])) / 2.)
    pop09_lumin_cube = (pop09_lumin_cube[1:, :]
                        + pop09_lumin_cube[:-1, :]) / 2.

    for age in [6.0, 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.]:
        aaa = np.abs(pop09_log_time - age).argmin()
        plt.plot(pop09_wave,
                 pop09_lumin_cube[:, aaa], linestyle='dotted',
                 )

    return pop09_log_time, pop09_wave, pop09_lumin_cube


# PopStar 21 ------------------------------------------------------------
# path21 = 'ssp/PopStar21/Z02/KRO/'
# list_files = os.listdir(path21)
#
# numbers = []
# for listt in list_files:
#     aaa = listt.replace('.dat', '')
#     numbers.append(float(aaa.replace('SSP-KRO-stellar_Z0.02_logt', '')))
# #print(numbers)
# indexes = np.argsort(numbers)
# pop21_log_time = np.sort(numbers)
# pop21_wave = np.loadtxt(path21 + list_files[0])[:, 0]
# pop21_lumin_cube = np.zeros((len(pop21_wave), len(list_files)))
#
# for nind, ind in enumerate(indexes):
#     pop21_lumin_cube[:, nind] = np.loadtxt(
#         path21
#         + 'SSP-KRO-stellar_Z0.02_logt'
#         + str(numbers[ind])
#         + '.dat'
#     )[:, 1]
#
# pop21_lumin_cube = np.log10(pop21_lumin_cube * 3.82e33)


# Figures ------------------------------------------------------

sb99_spline = load_sb99('data/ssp_synthetic_spectra'
                        '/starburst99/kroupa_padova/',
                        'kroupa_')

lambda_array = np.logspace(2., 6., num=2000)
time_array = np.logspace(6.3, 10, num=500)
print('limits: ', lambda_array[0], lambda_array[-1],
      time_array[0], time_array[-1])
direct = 'data/ssp_synthetic_spectra/starburst99/only_stripped/'

fig = plt.figure(figsize=(10, 10))
axes = fig.gca()

all_stripped_data = np.zeros((499, 1000, 6))
metall_str_array = np.zeros(6, dtype=float)

# Stripped stars ------------------------------------------------------
for n_metall, metall in enumerate(['0002', '002', '006', '014']):
    #
    # if not os.path.exists(direct + '0.' + metall):
    #     os.makedirs(direct + '0.' + metall)

    stripp_lambda_stripped = np.log10(np.loadtxt(
        'data/ssp_synthetic_spectra/published_runs_starburst99/run_Z'
        + metall +
        '/SED_Z0.'
        + metall +
        '_starburst.txt', max_rows=1)[1:])
    stripp_times_stripped = np.log10(np.loadtxt(
        'data/ssp_synthetic_spectra/published_runs_starburst99/run_Z'
        + metall +
        '/SED_Z0.'
        + metall +
        '_starburst.txt', skiprows=8, usecols=0))
    stripp_emiss_stripped = (np.log10(np.loadtxt(
        'data/ssp_synthetic_spectra/published_runs_starburst99/run_Z'
        + metall +
        '/SED_Z0.'
        + metall +
        '_starburst.txt', skiprows=8)[:, 1:]))
    print(len(stripp_times_stripped), stripp_times_stripped[0],
          stripp_times_stripped[-1])
    print(len(stripp_lambda_stripped), stripp_lambda_stripped[0],
          stripp_lambda_stripped[-1])

    stripp_emiss_stripped[np.isnan(stripp_emiss_stripped)] = -43.
    stripp_emiss_stripped[
        np.invert(np.isfinite(stripp_emiss_stripped))] = -43.

    all_stripped_data[:, :, n_metall + 1] = stripp_emiss_stripped

    spline_stripped = RectBivariateSpline(stripp_lambda_stripped,
                                          stripp_times_stripped,
                                          stripp_emiss_stripped.T,
                                          kx=1, ky=1, s=0)
    # for time in time_array:
    #     data_emiss = lambda_array * time
    #     np.savetxt(direct + '0.' + metall + '/stripped'
    #            + metall + '.spectrum1',
    #                np.column_stack((
    #                    time * np.ones(len(lambda_array)),
    #                    lambda_array,
    #                    data_emiss
    #                )))

    for ni, age in enumerate(np.log10(  # 12.7, 50, 100, 500, 800
            [11.]) + 6.):
        # for ni, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
        color = next(axes._get_lines.prop_cycler)['color']
        print(ni, color)

        # plt.plot(st99_wave, st99_log_emis[:, aaa],
        #          linestyle='dotted', lw=2,
        #          color=color)
        metall_float = float('0.' + metall)
        print(metall_float)
        metall_str_array[n_metall + 1] = metall_float
        plt.plot(st99_wave,
                 sb99_spline((np.log10(c / st99_wave * 1e10),
                              age, np.log10(metall_float))),
                 linestyle='-.', lw=2,
                 label='%.0f Myr' % ((10 ** age) * 1e-6),
                 color=color, alpha=1.2 * n_metall / 4. + 0.1)

        # aaa = np.abs(pegase_log_time - age).argmin()
        # plt.plot(pegase_wave, pegase_log_emis[:, aaa, -1],
        #          linestyle='dotted', lw=2,
        #          label='%.0f Myr' % ((10 ** pegase_log_time[aaa]) * 1e-6),
        #          color=color, alpha=1.2 * n_metall / 4.+0.1)

        aaa = np.abs(stripp_times_stripped - age).argmin()
        plt.plot(10 ** stripp_lambda_stripped,
                 stripp_emiss_stripped[aaa, :],
                 linestyle='--', lw=1,
                 label='%.0f Myr' % ((10 ** age) * 1e-6),
                 color=color, alpha=1.2 * n_metall / 4. + 0.1)

all_stripped_data[:, :, 0] = all_stripped_data[:, :, 1]
all_stripped_data[:, :, -1] = all_stripped_data[:, :, -2]
metall_str_array[0] = 1e-10
metall_str_array[-1] = 0.05

spline_stripped_total = RegularGridInterpolator(
    points=(stripp_times_stripped,
            stripp_lambda_stripped,
            np.log10(metall_str_array),
            ),
    values=all_stripped_data,
    method='linear',
    bounds_error=False, fill_value=-1
)

for n_met, metall in enumerate([1e-10, 4.e-04, 4.e-03,
                                8.e-03, 2.e-02, 5.e-02]):
    print(metall)
    if not os.path.exists(direct + str(metall)):
        os.makedirs(direct + str(metall))

    with open(direct + str(metall) + '/stripped'
              + str(metall) + '.spectrum1', 'w') as f:
        for time in time_array:
            data_emiss = np.log10(
                # 10 ** sb99_spline(xi=(np.log10(c / lambda_array * 1e10),
                #                       np.log10(time), np.log10(metall)))
                10 ** spline_stripped_total(
                    xi=(
                        np.log10(time),
                        np.log10(lambda_array),
                        np.log10(metall)))
            )
            np.savetxt(f,
                       np.column_stack((
                           time * np.ones(len(lambda_array)),
                           lambda_array,
                           data_emiss
                       )))

print(metall_str_array)
plt.xscale('log')
plt.show()


# spline_sum = = RectBivariateSpline()
# def fig_plot(age, color):
#     # aaa = np.abs(pop21_log_time - age).argmin()
#     # plt.plot(pop21_wave, pop21_lumin_cube[:, aaa],
#     #          '-',
#     #          label='Popstar21 Z = 0.02, log(t) = %.2f' %
#     #                pop21_log_time[aaa],
#     #          color=color)
#     aaa = np.abs(pop09_log_time - age).argmin()
#     plt.plot(pop09_wave, pop09_lumin_cube[:, aaa],
#              linestyle='.',
#              label='%.1f Myr' % ((10 ** pop09_log_time[aaa]) * 1e-6),
#              color=color, alpha=0.4)
#
#     aaa = np.abs(st99_log_time - age).argmin()
#     plt.plot(st99_wave, st99_log_emis[:, aaa],
#              linestyle='dotted', lw=2,
#              color=color)
#
#     aaa = np.abs(pegase_log_time - age).argmin()
#     plt.plot(pegase_wave, pegase_log_emis[:, aaa, -1],
#              linestyle='-', lw=2,
#              label='%.1f Gyr' % ((10 ** pop09_log_time[aaa]) * 1e-6),
#              color=color)


def pop_age(path, time, wave, emiss):
    plt.figure(figsize=(10, 8))
    plt.title('ssp: %s' % path)

    for age in [6.0, 6.5, 7.5, 8., 8.5, 9., 10.]:
        aaa = np.abs(time - age).argmin()
        plt.plot(wave, emiss[:, aaa],
                 label='log(t) = %.2f'
                       % time[aaa])

    plt.xscale('log')
    plt.legend(loc=4)

    # plt.xlim(1e2, 1e6)
    # plt.ylim(10, 22)

    plt.xlabel('wavelenght [A]')


# path09 = 'data/ssp_synthetic_spectra/PopStar09/sp-kro-z0200/'
# pop09_log_time, pop09_wave, pop09_lumin_cube = popstar09(
#     path09,
#     name='spneb_kro_0.15_100_z0200_t')

# # pop_age('ssp/final_run_spectrum', st99_log_time, st99_wave, st99_log_emis)
# # pop_age(path09, pop09_log_time, pop09_wave, pop09_lumin_cube)
# # pop_age(path21, pop21_log_time, pop21_wave, pop21_lumin_cube)
#
#
fig = plt.figure(figsize=(10, 10))
axes = fig.gca()

for ni, age in enumerate(np.log10([1., 2., 3., 4, 5., 10, 20, 100, 500,
                                   900]) + 6):
    # for ni, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
    color = next(axes._get_lines.prop_cycler)['color']
    print(ni, color)

    # aaa = np.abs(st99_log_time - age).argmin()
    # plt.plot(st99_wave*1e-4, st99_log_emis[:, aaa],
    #          linestyle='dotted', lw=2,
    #          color=color)
    #
    # aaa = np.abs(pegase_log_time - age).argmin()
    # plt.plot(pegase_wave*1e-4, pegase_log_emis[:, aaa, -1],
    #          linestyle='-', lw=2,
    #          label='%.0f Myr' % ((10 ** pegase_log_time[aaa]) * 1e-6),
    #          color=color)

    plt.plot(10 ** stripp_lambda_stripped,
             spline_stripped(stripp_lambda_stripped, age, grid=False),
             linestyle='--', lw=2,
             label='%.0f Myr' % ((10 ** age) * 1e-6),
             color=color)

plt.xscale('log')

# plt.xlim(1e-2, 1e2)
# plt.ylim(24, 34)
models = ['dotted', '-']
legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
                                  color='k') for i in range(2)],
                      ['Starburst99 Z=0.2', 'Pégase 3.0  Z=0.0001'],
                      loc=8,
                      fontsize=22)

axes.add_artist(legend22)

plt.legend(fontsize=18, title='Ages', title_fontsize=22)

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{\AA}^{-1}$ M$_{\odot}^{-1}$])')

plt.savefig('outputs/figures_paper/ssp_compar.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/ssp_compar_viridis_r.jpg',
            bbox_inches='tight')

if work_with_Lnu is True:
    plt.ylabel(r'log$_{10}$(L$_{\nu}$/Lsun '
               r'[erg s$^{-1}$ Hz$^{-1}$ Msun$^{-1}$])')
    plt.ylim(10, 22)
plt.show()
# ---------------------------------------------
fig = plt.figure()
axes = fig.gca()
plt.title('ssp pegase')
for age in [1, 2, 3, 5, 10, 20, 100, 500]:
    color = next(axes._get_lines.prop_cycler)['color']

    aaa = np.abs(t_pegase - age).argmin()
    plt.plot(l_pegase, pegase_log_emis[:, aaa, 2],
             linestyle='-',
             label='%.0fMyrs' % t_pegase[aaa],
             color=color)

    aaa = np.abs((10 ** pop09_log_time) * 1e-6 - age).argmin()
    plt.plot(pop09_wave, pop09_lumin_cube[:, aaa],
             linestyle='--',
             # marker='+',
             color=color)

    aaa = np.abs(st99_t_total / 1e6 - age).argmin()
    plt.plot(st99_wave, st99_log_emis[:, aaa],
             linestyle='dotted',
             # marker='x',
             color=color)

plt.xscale('log')
plt.xlim(1e2, 1e5)
# plt.ylim(10**-7, 10)
legend11 = plt.legend(loc="lower center")
lines = ['-', '--', 'dotted']
legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=lines[i],
                                  color='k')
                       for i in range(3)],
                      ['Pegase3', 'PopStar09', 'SB99'],
                      loc='upper right',
                      title=r'SSP model')

axes.add_artist(legend11)
axes.add_artist(legend22)
plt.xlabel('wavelenght [A]')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$/Lsun '
           r'[erg s$^{-1}$ A$^{-1}$ Msun$^{-1}$])')
if work_with_Lnu is True:
    plt.ylabel(r'log$_{10}$(L$_{\nu}$/Lsun [erg s$^{-1}$ Hz$^{-1}$ Msun$^{'
               r'-1}$])')
    plt.ylim(10, 22)
#
# # ---------------------------------------------
# fig = plt.figure()
# axes = fig.gca()
# plt.title('ssp pegase')
# for n_met, met in enumerate(pegase_metall):
#     age = 10
#     color = next(axes._get_lines.prop_cycler)['color']
#
#     aaa = np.abs(t_pegase - age).argmin()
#     plt.plot(l_pegase, pegase_log_emis[:, aaa, n_met],
#              linestyle='-',
#              label=met,
#              color=color)
#
# plt.xscale('log')
# plt.xlim(1e2, 1e5)
# plt.ylim(-7, 1)
# legend11 = plt.legend()
#
# axes.add_artist(legend11)
#
# plt.xlabel('wavelenght [A]')
# plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '
#            r'[erg s$^{-1}$ A$^{-1}$ Msun$^{-1}$])')
# if work_with_Lnu is True:
#     plt.ylabel(r'log$_{10}$(L$_{\nu}$/Lsun [erg s$^{-1}$ Hz$^{-1}$ Msun$^{'
#                r'-1}$])')
#     plt.ylim(10, 22)

# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['axes.labelsize'] = 24
# plt.rcParams['lines.markersize'] = 10
# plt.rc('font', size=24)
# plt.rc('axes', titlesize=30)
# plt.rc('axes', labelsize=30)
# plt.rc('xtick', labelsize=30)
# plt.rc('ytick', labelsize=30)
# plt.rc('legend', fontsize=22)
# plt.rc('figure', titlesize=20)
# plt.rc('xtick', top=True, direction='in')
# plt.rc('ytick', right=True, direction='in')
# plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
# plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
# plt.rc('xtick.minor', size=7, width=1.5)
# plt.rc('ytick.minor', size=7, width=1.5)

# ---------------------------------------------
# fig = plt.figure()
# axes = fig.gca()
# # plt.title('ssp pegase')
#
# print(pegase_metall[2])

# ages = [1, 5, 10, 50, 100, 500, 1000]
# cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
#
# for ploti in range(len(ages) + 1):
#
#     plt.figure(figsize=(10, 10))
#     for n_age, age in enumerate(ages[:ploti]):
#         aaa = np.abs(t_pegase - age).argmin()
#         plt.plot(l_pegase, pegase_log_emis[:, aaa, 2],
#                  linestyle='-',
#                  label='%s Myrs' % age)
#
#     plt.xscale('log')
#     plt.xlim(1e2, 1e5)
#     plt.ylim(26, 33.5)
#     legend11 = plt.legend(ncol=1, loc=1)
#
#     plt.xlabel(r'Wavelength [$Å$]')
#     plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '
#                r'[erg s$^{-1}$ $Å^{-1}$ M$_{\odot}^{-1}$])')
#     plt.savefig('outputs/ssp_all_' + str(ploti) + '.png')
#
#     fig, ax = plt.subplots(12, figsize=(25, 10))
#
#     ax0 = plt.subplot(121)
#
#     for n_age, age in enumerate(ages[:ploti]):
#         aaa = np.abs(t_pegase - age).argmin()
#         ax0.plot(l_pegase, pegase_log_emis[:, aaa, 2],
#                  linestyle='-',
#                  label='%s Myrs' % age)
#
#     ax0.set_xscale('log')
#     ax0.set_xlim(1e2, 1e5)
#     ax0.set_ylim(26, 33.5)
#     legend11 = plt.legend(ncol=1, loc=1)
#
#     ax0.set_xlabel(r'Wavelength [$Å$]')
#     ax0.set_ylabel(r'log$_{10}$(L$_{\lambda}$ '
#                    r'[erg s$^{-1}$ $Å^{-1}$ M$_{\odot}^{-1}$])')
#     if work_with_Lnu is True:
#         ax0.ylabel(r'log$_{10}$(L$_{\nu}$/L_{\odot}'
#                    r' [erg s$^{-1}$ Hz$^{-1}$ M_{\odot}$^{-1}$])')
#         ax0.set_ylim(10, 22)
#
#     # plt.subplots_adjust(wspace=5)
#
#     ax1 = plt.subplot(122)
#     # ax1 = ax[0]
#     x_sfr = np.linspace(0, 10)
#     m1 = [0.015, 2.7, 2.9, 5.6]
#     sfr = (lambda mi, x: eval(
#         'lambda ci, x : ci[0] * (1 + x)**ci[1]'
#         ' / (1 + ((1+x)/ci[2])**ci[3])')(mi, x))
#     plt.plot(x_sfr, sfr(m1, x_sfr), color='green',
#              label='Madau&Dickinson 14')
#
#     plt.yscale('log')
#     plt.xlabel('redshift z')
#     plt.ylabel(r'SFR [M$_{\odot}$ yr$^{-1}$Mpc$^{-3}$]')
#
#     z_ticks = []
#     ax_ll = ax1.twiny()
#
#     # lookback time z_ticks = []
#     lookback_labels = np.arange(1, 13.2, 1)[::-1]
#     for ll in lookback_labels * u.Gyr:
#         z_ticks.append(
#             z_at_value(cosmo.lookback_time, ll, zmin=1e-8, zmax=40.))
#
#     ax_ll.set_xticks(z_ticks)
#     # ax_ll.set_xlim(v)
#     ax_ll.set_xticklabels(
#         ["{0:.0f}".format(f) if (i % 2 or f >= 10) else "" for i, f in
#          enumerate(lookback_labels)])
#     ax_ll.set_xlabel("Lookback time [Gyr]")
#
#     v = ax1.set_xlim(0, 10)
#     ax_ll.set_xlim(v)
#
#     ax1.legend()
#
#     plt.savefig('outputs/ssp_timeline_' + str(ploti) + '.png')

plt.show()
