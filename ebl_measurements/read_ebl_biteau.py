import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import c
from astropy.table import Table

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


def read_singular_file(parent_dir, no_label=True):
    list_files = os.listdir(parent_dir)
    list_files.sort()

    if parent_dir == 'optical_data_2023/CIB':
        names = ['altieri1999.ecsv', 'bethermin2012.ecsv',
                 'clements1999.ecsv', 'driver2016.ecsv',
                 'finkbeiner2000.ecsv',  # 'fujimoto2016.ecsv',
                 'hopwood2010.ecsv',  # 'hsu2016.ecsv',
                 # 'juvela2009.ecsv',
                 'marsden2009.ecsv', 'matsuura2011.ecsv',
                 'odegard2007.ecsv',  # 'odegard2019.ecsv',
                 # 'penin2012.ecsv']\
                 ]
    elif parent_dir == 'optical_data_2023/COB':
        names = list_files
    else:
        names = list_files

    markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

    for ni, name in enumerate(names):

        data = Table.read(parent_dir + '/' + name,
                          format='ascii.ecsv')
        x_data = data.colnames[0]

        if data[x_data].unit is u.Hz:
            data[x_data] = (c / data['nu']).to(u.um)

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


fig = plt.figure(figsize=(15, 8))
axes = fig.gca()
read_singular_file('optical_data_2023/CIB', no_label=False)
read_singular_file('optical_data_2023/COB', no_label=True)
read_singular_file('optical_data_2023/ZL', no_label=False)

legend1 = plt.legend(bbox_to_anchor=(1.04, 1),
                     loc="upper left", title=r'Measurements')

plt.yscale('log')
plt.xscale('log')
plt.ylim(1, 120)
plt.xlim(0.3, 5.5)
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

axes.add_artist(legend1)

plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)

plt.show()

# def plot_ebl_measurement_collection(file, color='.75', cm=plt.cm.rainbow, yearmin=None, yearmax=None, nolabel=False): # Dark2
#
#     with open(file, 'r') as stream:
#         try:
#             ebl_m = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     ebl_m = sorted(ebl_m.items(), key=lambda t: t[1]['year'])
#
#     if yearmin or yearmax:
#         ebl_m_n = []
#         for m in ebl_m:
#             if yearmin and m['year'] >= yearmin:
#                 ebl_m_n.append(m)
#             if yearmax and m['year'] < yearmax:
#                 ebl_m_n.append(m)
#         ebl_m = ebl_m_n
#
#     markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']
#
#     for mi, m in enumerate(ebl_m):
#         yerr = [m[1]['data']['ebl_err_low'], m[1]['data']['ebl_err_high']]
#
#         if m[1]['is_upper_limit'] or m[1]['is_lower_limit']:
#             yerr = [10. ** (np.log10(x) + 0.14) - x for x in m[1]['data']['ebl']]
#             if m[1]['is_upper_limit'] :
#                 yerr = [yerr, list(map(lambda x: 0., range(len(yerr))))]
#             else:
#                 yerr = [list(map(lambda x: 0., range(len(yerr)))), yerr]
#         if cm:
#             color = cm(float(mi) / (len(ebl_m) - 1.))
#         label = m[1]['full_name']
#         if nolabel:
#             label = ""
#         plt.errorbar(x=m[1]['data']['lambda'], y=m[1]['data']['ebl'],
#                      yerr=yerr, label=label,
#                      marker=markers[mi % len(markers)], color=color, mec=color,
#                      linestyle='None', lolims=m[1]['is_lower_limit'], uplims=m[1]['is_upper_limit'])
