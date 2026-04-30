# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as trans
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data, dictionary_datatype

from astropy import units as u
import astropy.constants as c

from ebltable.ebl_from_model import EBL
from ebltable.tau_from_model import OptDepth

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
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=10)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=10)
plt.rc('xtick.minor', size=5, width=1.)
plt.rc('ytick.minor', size=5, width=1.)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


my_ebl = ['bosa.txt', 'chary.txt', '2bb.txt']
# my_ebl = ['chary.txt']


waves_ebl = np.geomspace(0.05, 1e3, num=int(1e4))

my_directory = 'outputs/outputs_systematics_14perct/'


ebl_finke = EBL.readmodel('finke2022')
ebl_SL = EBL.readmodel('saldana-lopez')

direct_franceschini_data = '/home/porrassa/Downloads/franceschini2017/'

table1 = np.loadtxt(
    direct_franceschini_data + 'table1.txt', dtype=float)
table2 = np.loadtxt(
    direct_franceschini_data + 'table2.txt', dtype=float)
table3 = np.loadtxt(
    direct_franceschini_data + 'table3.txt', dtype=float)

# []  ''  _  {}  p

zz_array1 = np.unique(table1[0, :])
table1 = table1[1:, :]
zz_array2 = np.unique(table2[0, :])
table2 = table2[1:, :]
zz_array3 = np.unique(table3[0, :])
table3 = table3[1:, :]

zz_array = np.concatenate((zz_array1, zz_array2, zz_array3))
table = np.concatenate((table1, table2, table3), axis=1)

energyarray = np.linspace(-3, 1.5, num=500)
dataarray = np.zeros((len(zz_array), len(energyarray)))

for ni, ii in enumerate(zz_array):
    xx = np.concatenate(
        ([table[0, 2*ni]*1.001], table[:, 2*ni], [table[-1, 2*ni]*1.001]))
    yy = np.concatenate(([-43], table[:, 2*ni+1], [-43]))
    aa = UnivariateSpline(xx, yy, k=1, s=0, ext=3)
    dataarray[ni, :] = aa(energyarray)

plt.figure(figsize=(10, 10))
for ni, ii in enumerate(zz_array):
    plt.plot(energyarray, dataarray[ni, :], label=ii)

dataarray = c.c / 4. / np.pi * 10**energyarray * u.eV * 10**dataarray / u.cm**3
dataarray = dataarray.to(u.nW*u.m**-2) / (1 + zz_array[:, np.newaxis])**3
wavelength_array = (c.h*c.c/(10**(energyarray)*u.eV)).to(u.micron)

plt.figure(figsize=(10, 10))
for ni, ii in enumerate(zz_array):
    plt.plot(wavelength_array, dataarray[ni, :], label=ii)

sort_array = np.argsort(wavelength_array)
dataarray = (dataarray.value)[:, sort_array]
wavelength_array = (wavelength_array.value)[sort_array]

aabigsline = RectBivariateSpline(
    x=zz_array, y=wavelength_array, z=dataarray, kx=1, ky=1, s=0)

# for ni, ii in enumerate(zz_array):
#     plt.scatter(wavelength_array, aabigsline(ii, wavelength_array),
#                 s=12)

ebl_franccc = EBL(z=zz_array, lmu=wavelength_array,
                nuInu=dataarray.T, model='franc')


plt.xscale('log')
plt.yscale('log')
plt.legend(ncol=4, loc=2)
plt.ylim(bottom=0.03)
# plt.show()


fig, ax = plt.subplots(3, 2, figsize=(10, 8))
plt.subplots_adjust(wspace=0, hspace=0)
z_array = [0, 0.2, 0.4, 0.8, 1, 2]

colors = {
    'bosa.txt': 'b',
    'chary.txt': 'orange',
    '2bb.txt':'g',
}
labels = {
    'bosa.txt': 'BOSA',
    'chary.txt': 'Chary',
    '2bb.txt':'2BB',
}

waves_fran = np.geomspace(0.1, 180)

for ni, zi in enumerate(z_array):
    plt.subplot(3, 2, ni+1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.1, 1e3)

    plt.text(s=r'$z=$' + str(zi),
             x=0.5, y=0.85, color='k',
             transform=(ax.flatten())[ni].transAxes,
             fontsize=18, horizontalalignment='center',
             bbox=dict(facecolor='w', alpha=1, edgecolor='grey'))

    if ni == 0 or ni == 1:
        plt.ylim(1, 25)
        waves_fran = np.geomspace(0.1, 250)

    if ni == 2 or ni == 3:
        plt.ylim(0.3, 25)
        waves_fran = np.geomspace(0.1, 180)

    if ni == 4 or ni == 5:
        plt.ylim(0.06, 30)
        waves_fran = np.geomspace(0.1, 100)

    if ni == 0:

        for d in my_ebl:
            bbb = EBL.readascii(
                'outputs/outputs_dust_final_new/' + d,
                model_name='mine')
            plt.plot(waves_ebl,
                     bbb.ebl_array(z=zi, lmu=waves_ebl),
                     linestyle='--', lw=1,
                     # label='10% ' + labels[d],
                     label='No syst',
                     c='k')
            bbb = EBL.readascii(
                my_directory + d,
                model_name='mine')
            plt.plot(waves_ebl,
                     bbb.ebl_array(z=zi, lmu=waves_ebl),
                     linestyle='-', lw=1, label='14% ' + labels[d],
                     c=colors[d])
        # plt.plot(waves_ebl,
        #          ebl_finke.ebl_array(z=zi, lmu=waves_ebl),
        #          linestyle='dotted', lw=1, label='Finke+22', c='fuchsia')
        # plt.plot(waves_ebl,
        #          ebl_SL.ebl_array(z=zi, lmu=waves_ebl),
        #          linestyle='-.', lw=1, label='Saldana-Lopez+21', c='r')
        # plt.plot(waves_fran,
        #          ebl_franccc.ebl_array(z=zi, lmu=waves_fran),
        #          label='Franceschini+17', linestyle='--', lw=1, c='k')
    else:
        for d in my_ebl:
            bbb = EBL.readascii(
                'outputs/outputs_dust_final_new/' + d,
                model_name='mine')
            plt.plot(waves_ebl,
                     bbb.ebl_array(z=zi, lmu=waves_ebl),
                     linestyle='--', lw=1, c='k')
            bbb = EBL.readascii(
                my_directory + d,
                model_name='mine')
            plt.plot(waves_ebl,
                     bbb.ebl_array(z=zi, lmu=waves_ebl),
                     linestyle='-', lw=1, c=colors[d])
        # plt.plot(waves_ebl,
        #          ebl_finke.ebl_array(z=zi, lmu=waves_ebl),
        #          linestyle='dotted', lw=1, c='fuchsia')
        # plt.plot(waves_ebl,
        #          ebl_SL.ebl_array(z=zi, lmu=waves_ebl),
        #          linestyle='-.', lw=1, c='r')
        # plt.plot(waves_fran,
        #          ebl_franccc.ebl_array(z=zi, lmu=waves_fran),
        #          linestyle='--', lw=1, c='k')


plt.subplot(3, 2, 3)
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')


aa = plt.subplot(3, 2, 5)
aa.set_xticks([0.1, 1, 10, 100, 1000],
              labels=('0.1', '1', '10', '100', ''))
plt.xlabel(r'Wavelength (µm)')


aa = plt.subplot(3, 2, 6)
aa.set_xticks([0.1, 1, 10, 100, 1000],
              labels=('0.1', '1', '10', '100', '1000'))
plt.xlabel(r'Wavelength (µm)')

for i in range(1, 5):
    plt.subplot(3, 2, i)
    plt.tick_params('x', labelbottom=False)

for i in [2, 4, 6]:
    plt.subplot(3, 2, i)
    plt.tick_params('y', labelleft=False)

dx = 10 / 72.
offset = trans.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
for nn, label in enumerate(ax[2][1].xaxis.get_majorticklabels()):
    if nn == 4:
        label.set_transform(label.get_transform() - offset)

plt.subplot(3, 2, 1)
plt.legend(loc=8, bbox_to_anchor=(1., 1.01),
           ncol=3,
           fontsize=16)
handles, legends = ax[0][0].get_legend_handles_labels()
sort_legend = [0, 5, 1, 4, 2, 3]
# plt.legend([handles[i] for i in sort_legend],
#            [legends[i] for i in sort_legend],
#            loc=8, bbox_to_anchor=(1., 1.01),
#            ncol=3, fontsize=16)

# lines = ['-', 'dotted']
# legend1 = plt.legend(ncol=3, loc=3,
#                       fontsize=18,
#                       title_fontsize=20, title='Redshift')
# legend2 = plt.legend([plt.Line2D([], [], linestyle=lines[i],
#                                  color='k')
#                       for i in range(2)],
#                      ['Our model', 'Finke22'],
#                      loc=2, fontsize=16, framealpha=0.4)

# ax.add_artist(legend1)
# ax.add_artist(legend2)

plt.savefig('outputs/outputs_dust_final_new/cb_redshifs.pdf',
            bbox_inches='tight')
plt.savefig('outputs/outputs_dust_final_new/cb_redshifs.png',
            bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 10))
energy_array = np.geomspace(0.1, 1e2)

# webplot = np.loadtxt(direct_franceschini_data + 'z1_webdigitizer.csv',
#                      skiprows=1, delimiter=','
#                      )
# print(webplot)
zz_array = [0.1, 0.2, 0.3, 0.4, 0.5]

for ni, zi in enumerate(zz_array):
    print(zi)

    bbb = EBL.readascii(
        'outputs/outputs_systematics_10perct/chary.txt', model_name='mine')
    plt.plot(energy_array,
             bbb.optical_depth(z0=zi, ETeV=energy_array),
             linestyle='-', lw=2,
             c='k', alpha=1 - zi, label=zi)
    for d in my_ebl:
        bbb = EBL.readascii(
            my_directory + d, model_name='mine')
        plt.plot(energy_array,
                 bbb.optical_depth(z0=zi, ETeV=energy_array),
                 linestyle='-', lw=1, c=colors[d],
                 alpha=1 - zi)

    # franc_opt = OptDepth.readmodel(model='franceschini2017')

    # plt.plot(energy_array,
    #         franc_opt.opt_depth(z=zi, ETeV=energy_array),
    #              linestyle='-', lw=2, label='ebltable z='+str(zi))


    # opacityy = ebl_franccc.optical_depth(z0=zi, ETeV=energy_array)
    # plt.scatter(energy_array, opacityy, label='Franceschini17',
    #             s=20)

# plt.scatter(webplot[:, 0], webplot[:, 1], marker='+', s=25,
#             label='Webplot z=1', c='k')

plt.xscale('log')
plt.yscale('log')
plt.xlim(0.1, 1e2)
plt.legend(fontsize=10, ncol=3, )

plt.xlabel('Energy (TeV)')
plt.ylabel(r'Optical depth $\tau$')

plt.savefig('outputs/outputs_dust_final_new/optdepth_redshifs.pdf',
            bbox_inches='tight')
plt.savefig('outputs/outputs_dust_final_new/optdepth_redshifs.png',
            bbox_inches='tight')

plt.show()
