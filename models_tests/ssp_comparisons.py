import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'models_tests':
    os.chdir("..")

work_with_Lnu = False

# Starburst 99 ------------------------------------------------------------
data_starburst_old = np.loadtxt('ssp/final_run_spectrum', skiprows=6)
t_old = np.unique(data_starburst_old[:, 0])
l_old = np.unique(data_starburst_old[:, 1])
dd_old = data_starburst_old[:, 3].reshape(t_old.shape[0], l_old.shape[0]).T

data_starburst = np.loadtxt('ssp/low_res_for_real.spectrum1', skiprows=6)
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

# Pegase ------------------------------------------------------------
pegase_metall = [0.1, 0.05, 0.02, 0.008, 0.004, 0.0004, 0.0001]

data_pegase = np.loadtxt(
    'ssp/pegase3/spectral_resultsZ0.02.txt')
t_pegase = np.unique(data_pegase[:, 0])
l_pegase = np.unique(data_pegase[:, 1])

dd_pegase = np.zeros((l_pegase.shape[0],
                      t_pegase.shape[0],
                      len(pegase_metall)))

for n_met, met in enumerate(pegase_metall):
    data_pegase = np.loadtxt(
        'ssp/pegase3/spectral_resultsZ' + str(met) + '.txt')
    dd_pegase[:, :, n_met] = data_pegase[:, 2].reshape(t_pegase.shape[0],
                                                       l_pegase.shape[0]).T

pegase_log_time = np.log10(t_pegase)  # log(time/yrs)
pegase_wave = l  # amstrongs
pegase_log_emis = np.log10(dd_pegase)  # - np.log10(3.828e33)

if work_with_Lnu is True:
    pegase_log_emis += (np.log10(1E10 * c)
                        - 2. * np.log10(c * 1e10 / l_pegase
                                        )[:, np.newaxis])


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

def fig_plot(age, color):
    # aaa = np.abs(pop21_log_time - age).argmin()
    # plt.plot(pop21_wave, pop21_lumin_cube[:, aaa],
    #          '-',
    #          label='Popstar21 Z = 0.02, log(t) = %.2f' %
    #                pop21_log_time[aaa],
    #          color=color)
    aaa = np.abs(pop09_log_time - age).argmin()
    plt.plot(pop09_wave, pop09_lumin_cube[:, aaa],
             linestyle='solid',
             label='%.1f Myr' % ((10 ** pop09_log_time[aaa]) * 1e-6),
             color=color)
    aaa = np.abs(st99_log_time - age).argmin()
    plt.plot(st99_wave, st99_log_emis[:, aaa],
             linestyle='dotted',
             # label='SB99, log(t) = %.2f' % st99_log_time[aaa],
             color=color)


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


path09 = 'ssp/PopStar09/sp-kro-z0200/'
pop09_log_time, pop09_wave, pop09_lumin_cube = popstar09(
    path09,
    name='spneb_kro_0.15_100_z0200_t')

# pop_age('ssp/final_run_spectrum', st99_log_time, st99_wave, st99_log_emis)
# pop_age(path09, pop09_log_time, pop09_wave, pop09_lumin_cube)
# pop_age(path21, pop21_log_time, pop21_wave, pop21_lumin_cube)


fig = plt.figure()
axes = fig.gca()
color = ['b', 'orange', 'k', 'r', 'green', 'grey', 'limegreen', 'purple',
         'brown', 'gray']
plt.title('ssp comparisons at Z=0.02')

for ni, age in enumerate(np.log10([1., 2., 3., 4, 5., 10, 20, 100, 500,
                                   900]) + 6):
    # for ni, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
    fig_plot(age=age, color=color[ni])

plt.xscale('log')

plt.xlim(1e2, 1e6)
# plt.ylim(-15, 1)
models = ['solid', 'dotted']
legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
                                  color='k') for i in range(2)],
                      ['Popstar09', 'Starburst99'], loc=8,
                      title=r'Components')

axes.add_artist(legend22)

plt.legend()

plt.xlabel('wavelenght [A]')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$/Lsun '
           r'[erg s$^{-1}$ A$^{-1}$ Msun$^{-1}$])')

if work_with_Lnu is True:
    plt.ylabel(r'log$_{10}$(L$_{\nu}$/Lsun [erg s$^{-1}$ Hz$^{-1}$ Msun$^{'
               r'-1}$])')
    plt.ylim(10, 22)

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

# ---------------------------------------------
fig = plt.figure()
axes = fig.gca()
plt.title('ssp pegase')
for n_met, met in enumerate(pegase_metall):
    age = 10
    color = next(axes._get_lines.prop_cycler)['color']

    aaa = np.abs(t_pegase - age).argmin()
    plt.plot(l_pegase, pegase_log_emis[:, aaa, n_met],
             linestyle='-',
             label=met,
             color=color)


plt.xscale('log')
plt.xlim(1e2, 1e5)
plt.ylim(-7, 1)
legend11 = plt.legend()


axes.add_artist(legend11)

plt.xlabel('wavelenght [A]')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '
           r'[erg s$^{-1}$ A$^{-1}$ Msun$^{-1}$])')
if work_with_Lnu is True:
    plt.ylabel(r'log$_{10}$(L$_{\nu}$/Lsun [erg s$^{-1}$ Hz$^{-1}$ Msun$^{'
               r'-1}$])')
    plt.ylim(10, 22)

plt.show()
