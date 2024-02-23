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


config_data = read_config_file(
    'scripts/input_files/input_data_paper.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)

waves_ebl = np.geomspace(5e-6, 10, num=int(1e6))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


def spline_pegase0001(lambda_array):
    method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                          grid=False)


def spline_starburst(lambda_array):
    method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                          grid=False)


list_working_models = {
    'ModelA': {'label': 'Model A', 'callable_func': spline_starburst,
               'color': 'b', 'linewidth': 3},
    'ModelB': {'label': 'Model B', 'callable_func': spline_pegase0001,
               'color': 'tab:orange', 'linewidth': 3},
    'Finke22': {'label': 'Finke22', 'callable_func': spline_finke,
                'color': 'magenta', 'linewidth': 2},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
             'color': 'k', 'linewidth': 2}
}

# Beginning of figure specifications
plt.figure(figsize=(16, 10))  # figsize=(16, 10))
ax1 = plt.gca()

handlers, labels = [], []
for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]

    plt.loglog(waves_ebl, model['callable_func'](waves_ebl),
               c=model['color'], lw=model['linewidth'], zorder=2
               )

    handlers.append(plt.Line2D([], [],
                               linewidth=model['linewidth'],
                               linestyle='-',
                               color=model['color']))
    labels.append(model['label'])

ebl_class.change_axion_contribution(1e2, 1e-13)
plt.loglog(waves_ebl,
           (10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0.,
                                             grid=False)
            + spline_cuba(waves_ebl)), c='green', zorder=1)

handlers.append(plt.Line2D([], [],
                           linewidth=2,
                           linestyle='-',
                           color='green'))
labels.append(r'CUBA + cosmic axion''\n '
              r'decay (example)''\n'
              r'    m$_a = 10^2$ eV''\n'
              r'    g$_{a\gamma} = 10^{-13}$ GeV$^{-1}$')

# We introduce all the EBL measurements
upper_lims_all, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax1, plot_measurs=True)

plt.xlim(5e-6, 1e1)
plt.ylim(5e-3, 120)
legend22 = plt.legend(handlers, labels,
                      loc=7, bbox_to_anchor=(1., 0.3),
                      title=r'Models', fontsize=16)


handles, labels = ax1.get_legend_handles_labels()

for i in range(len(labels)):
    if labels[i].__contains__('LORRI'):
        handles[i] = (plt.Line2D([], [], linestyle='',
                                 color='g', markerfacecolor='w',
                                 marker='*', markersize=16),
                      plt.Line2D([], [], linestyle='',
                                 color='k', markerfacecolor='k',
                                 marker='.', markersize=8)
                      )
legend11 = plt.legend(handles, labels,
                      handler_map={tuple: HandlerTuple(ndivide=1)},
                      title='Measurements', ncol=2, loc=2,
                      fontsize=11.5,
                      title_fontsize=20)  # , bbox_to_anchor=(1.001, 0.99))

ax1.add_artist(legend11)
ax1.add_artist(legend22)

plt.annotate(text='', xy=(3e-3, 7e-3), xytext=(5e-6, 7e-3),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='', xy=(0.1, 7e-3), xytext=(3e-3, 7e-3),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='', xy=(10, 7e-3), xytext=(0.1, 7e-3),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='CXB', xy=(1e-4, 7.5e-3), alpha=0.7, color='grey')
plt.annotate(text='CUB', xy=(0.035, 7.5e-3), alpha=0.7, color='grey')
plt.annotate(text='COB', xy=(1, 7.5e-3), alpha=0.7, color='grey')

plt.xlabel(r'Wavelength ($\mu$m)')

plt.savefig('outputs/figures_paper/cb.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb.png', bbox_inches='tight')



fig, ax = plt.subplots(figsize=(8.75, 7))

z_array = [0., 0.2, 0.5, 1., 2., 5.]
lambda_array = np.geomspace(0.05, 10.)

N = len(z_array)
cividis_mine = mpl.colormaps['cividis']._resample(N)
cividis_mine.colors[-1, 1] = 0.7
cividis_mine.colors[-1, 2] = 0.
cividis_mine.colors = cividis_mine.colors[::-1]
ax.set_prop_cycle(get_cycle(cividis_mine, N))

def spline_pegase0001(lambda_array, z_value):
    method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), z_value,
                          grid=False)


def spline_starburst(lambda_array, z_value):
    method_y = np.load('outputs/final_outputs_check_NOlims 2024-02-22 15:26:04/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), z_value,
                          grid=False)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array(z_array), lambda_array)

for ni, i in enumerate(z_array):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(lambda_array, spline_starburst(lambda_array, i),
             color=color, linestyle='-', label=i, lw=2)
    plt.plot(lambda_array, spline_pegase0001(lambda_array, i),
             color=color, linestyle='--', lw=2)
    plt.plot(lambda_array, nuInu['finke2022'][ni],
             color=color, linestyle='dotted', lw=2)


plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

plt.xlim(0.1, 1e1)
plt.ylim(5e-3, 20)

plt.xscale('log')
plt.yscale('log')

lines = ['-', '--', 'dotted']
legend1 = plt.legend(ncol=3, loc=3,
                      fontsize=18,
                      title_fontsize=20, title='Redshift')
legend2 = plt.legend([plt.Line2D([], [], linestyle=lines[i],
                                 color='k')
                      for i in range(3)],
                     ['Model A', 'Model B', 'Finke22'],
                     loc=2, fontsize=16, framealpha=0.4)

ax.add_artist(legend1)
ax.add_artist(legend2)

plt.savefig('outputs/figures_paper/cb_redshifs.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_redshifs.png',
            bbox_inches='tight')

# plt.show()
