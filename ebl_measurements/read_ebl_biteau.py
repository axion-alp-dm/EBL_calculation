import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)


def read_singular_file(parent_dir, no_label=True):
    list_files = os.listdir(parent_dir)
    list_files.sort()

    markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

    for ni, name in enumerate(list_files):

        data = Table.read(parent_dir + '/' + name,
                          format='ascii.ecsv')

        x_data = data.colnames[0]
        data[x_data] = data[x_data].to(u.um, equivalencies=u.spectral())

        if data['nuInu'].unit.is_equivalent(u.Jy / u.sr):
            data['nuInu'] = (data['nuInu'].to(u.W / u.m ** 2 / u.Hz / u.sr)
                             * data[x_data].to(
                        u.Hz, equivalencies=u.spectral()))
            data['nuInu_errn'] = (
                    data['nuInu_errn'].to(u.W / u.m ** 2 / u.Hz / u.sr)
                    * data[x_data].to(
                u.Hz, equivalencies=u.spectral()))
            data['nuInu_errp'] = (
                    data['nuInu_errp'].to(u.W / u.m ** 2 / u.Hz / u.sr)
                    * data[x_data].to(
                u.Hz, equivalencies=u.spectral()))

        if no_label:
            label = ''
        else:
            label = data.meta['label']

        plt.errorbar(x=data[x_data],
                     y=data['nuInu'].to(u.nW / u.m ** 2 / u.sr),
                     yerr=[data['nuInu_errn'].to(u.nW / u.m ** 2 / u.sr),
                           data['nuInu_errp'].to(u.nW / u.m ** 2 / u.sr)],
                     label=label, linestyle='',
                     marker=markers[ni % len(markers)]
                     )


def read_specific_obs_type(parent_dir, obs_type, no_label=False):
    list_dirs = os.listdir(parent_dir)
    list_dirs.sort()

    for directory in list_dirs:

        list_files = os.listdir(parent_dir + '/' + directory)
        list_files.sort()

        markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

        for ni, name in enumerate(list_files):

            data = Table.read(parent_dir + '/' + directory + '/' + name,
                              format='ascii.ecsv')

            if data.meta['observable_type'] == obs_type:
                x_data = data.colnames[0]
                data[x_data] = data[x_data].to(u.um,
                                               equivalencies=u.spectral())

                if data['nuInu'].unit.is_equivalent(u.Jy / u.sr):
                    data['nuInu'] = (
                            data['nuInu'].to(
                                u.W / u.m ** 2 / u.Hz / u.sr)
                            * data[x_data].to(
                        u.Hz, equivalencies=u.spectral()))
                    data['nuInu_errn'] = (data['nuInu_errn'].to(
                        u.W / u.m ** 2 / u.Hz / u.sr)
                                          * data[x_data].to(
                                u.Hz, equivalencies=u.spectral()))
                    data['nuInu_errp'] = (data['nuInu_errp'].to(
                        u.W / u.m ** 2 / u.Hz / u.sr)
                                          * data[x_data].to(
                                u.Hz, equivalencies=u.spectral()))

                if no_label:
                    label = ''
                else:
                    label = data.meta['label']

                plt.errorbar(x=data[x_data],
                             y=data['nuInu'].to(u.nW / u.m ** 2 / u.sr),
                             yerr=[data['nuInu_errn'].to(
                                 u.nW / u.m ** 2 / u.sr),
                                 data['nuInu_errp'].to(
                                     u.nW / u.m ** 2 / u.sr)],
                             label=label, linestyle='',
                             marker=markers[ni % len(markers)]
                             )


def dictionary_datatype(parent_dir, obs_type,
                        lambda_min=0., lambda_max=5,
                        plot_measurs=False):
    print(os.getcwd())
    list_dirs = os.listdir(parent_dir)
    list_dirs.sort()

    lambdas = np.array([])
    nuInu = np.array([])
    nuInu_errn = np.array([])
    nuInu_errp = np.array([])


    markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

    for directory in list_dirs:

        list_files = os.listdir(parent_dir + '/' + directory)
        list_files.sort()

        for ni, name in enumerate(list_files):

            data = Table.read(parent_dir + '/' + directory + '/' + name,
                              format='ascii.ecsv')

            if data.meta['observable_type'] == obs_type:

                x_data = data.colnames[0]

                if data['nuInu'].unit.is_equivalent(u.Jy / u.sr):
                    data['nuInu'] = (
                            data['nuInu'].to(
                                u.W / u.m ** 2 / u.Hz / u.sr)
                            * data[x_data].to(
                        u.Hz, equivalencies=u.spectral()))
                    data['nuInu_errn'] = (data['nuInu_errn'].to(
                        u.W / u.m ** 2 / u.Hz / u.sr)
                                          * data[x_data].to(
                                u.Hz, equivalencies=u.spectral()))
                    data['nuInu_errp'] = (data['nuInu_errp'].to(
                        u.W / u.m ** 2 / u.Hz / u.sr)
                                          * data[x_data].to(
                                u.Hz, equivalencies=u.spectral()))

                # Change of units to our standard
                data[x_data] = data[x_data].to(u.um,
                                               equivalencies=u.spectral())
                data['nuInu'] = data['nuInu'].to(u.nW / u.m ** 2 / u.sr)
                data['nuInu_errn'] = data['nuInu_errn'].to(
                    u.nW / u.m ** 2 / u.sr)
                data['nuInu_errp'] = data['nuInu_errp'].to(
                    u.nW / u.m ** 2 / u.sr)
                lambdas = np.append(lambdas, data[x_data])

                nuInu = np.append(nuInu,
                                  data['nuInu'])
                nuInu_errn = np.append(nuInu_errn,
                                       data['nuInu_errn'])
                nuInu_errp = np.append(nuInu_errp,
                                       data['nuInu_errp'])

                if plot_measurs is True:
                    plt.errorbar(x=data[x_data], y=data['nuInu'],
                                 yerr=[data['nuInu_errn'],
                                       data['nuInu_errp']],
                                 # label=data.meta['label'],
                                 linestyle='',
                                 marker=markers[ni % len(markers)]
                                 )

    t = Table(np.column_stack((lambdas, nuInu, nuInu_errn, nuInu_errp)),
              names=('lambda', 'nuInu', 'nuInu_errn', 'nuInu_errp'),
              units=(u.um, u.nW / u.m ** 2 / u.sr,
                     u.nW / u.m ** 2 / u.sr, u.nW / u.m ** 2 / u.sr))

    t = t[(t['lambda'] >= lambda_min) * (t['lambda'] <= lambda_max)]

    if plot_measurs is True:
        plt.scatter(x=t['lambda'], y=t['nuInu'],
                      s=80, facecolors='none', edgecolors='r',
                      label='chosen'
                      )

    return t


# dictionary_datatype('optical_data_2023', 'IGL', plot_measurs=True)


# fig = plt.figure(figsize=(15, 8))
# plt.title('CIB')
# axes = fig.gca()
# read_singular_file('optical_data_2023/CIB', no_label=False)
# # read_singular_file('optical_data_2023/COB', no_label=False)
# # read_singular_file('optical_data_2023/ZL', no_label=False)
#
# legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
#                      loc="upper left", title=r'Measurements')
#
# plt.yscale('log')
# plt.xscale('log')
# # plt.ylim(1, 120)
# # plt.xlim(0.3, 5.5)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
#
# axes.add_artist(legend1)
#
# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
#
# fig = plt.figure(figsize=(15, 8))
# plt.title('M')
# axes = fig.gca()
# read_specific_obs_type('optical_data_2023', 'M')
#
# model_finke = np.loadtxt('../ebl_codes/EBL_intensity_total_z0.00.dat')
# plt.plot(model_finke[:, 0] / 1e4, model_finke[:, 1], '-k')
#
# legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
#                      loc="upper left", title=r'Measurements')
#
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim(1, 120)
# # plt.xlim(0.3, 5.5)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
#
# axes.add_artist(legend1)
#
# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
#
# fig = plt.figure(figsize=(15, 8))
# plt.title('IGL')
# axes = fig.gca()
# read_specific_obs_type('optical_data_2023', 'IGL')
#
# model_finke = np.loadtxt('../ebl_codes/EBL_intensity_total_z0.00.dat')
# plt.plot(model_finke[:, 0] / 1e4, model_finke[:, 1], '-k')
#
# legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
#                      loc="upper left", title=r'Measurements')
#
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim(1, 120)
# # plt.xlim(0.3, 5.5)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
#
# axes.add_artist(legend1)
#
# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
#
# fig = plt.figure(figsize=(15, 8))
# plt.title('UL')
# axes = fig.gca()
# read_specific_obs_type('optical_data_2023', 'UL')
#
# model_finke = np.loadtxt('../ebl_codes/EBL_intensity_total_z0.00.dat')
# plt.plot(model_finke[:, 0] / 1e4, model_finke[:, 1], '-k')
#
# legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
#                      loc="upper left", title=r'Measurements')
#
# plt.yscale('log')
# plt.xscale('log')
# plt.ylim(1, 120)
# # plt.xlim(0.3, 5.5)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
#
# axes.add_artist(legend1)
#
# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
# fig = plt.figure(figsize=(15, 8))
# plt.title('COB')
# axes = fig.gca()
# # read_singular_file('optical_data_2023/CIB', no_label=False)
# read_singular_file('optical_data_2023/COB', no_label=False)
# # read_singular_file('optical_data_2023/ZL', no_label=False)
#
# legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
#                      loc="upper left", title=r'Measurements')
#
# plt.yscale('log')
# plt.xscale('log')
# # plt.ylim(1, 120)
# # plt.xlim(0.3, 5.5)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
#
# axes.add_artist(legend1)
#
# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
#
# fig = plt.figure(figsize=(15, 8))
#
# plt.title('ZL')
# axes = fig.gca()
# # read_singular_file('optical_data_2023/CIB', no_label=False)
# # read_singular_file('optical_data_2023/COB', no_label=False)
# read_singular_file('optical_data_2023/ZL', no_label=False)
#
# legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
#                      loc="upper left", title=r'Measurements')
#
# plt.yscale('log')
# plt.xscale('log')
# # plt.ylim(1, 120)
# # plt.xlim(0.3, 5.5)
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
#
# axes.add_artist(legend1)
#
# plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)

# plt.show()
