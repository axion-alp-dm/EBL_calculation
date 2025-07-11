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
from data.cb_measurs.import_cb_measurs import import_cb_data, dictionary_datatype

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
plt.rc('xtick', top=False, direction='in')
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


waves_ebl = np.geomspace(5e-6, 1e1, num=int(1e6))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


list_working_models = {
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
             'color': 'k', 'linewidth': 2.5, 'ls': 'dotted'}
}

# Beginning of figure specifications
fig, ax1 = plt.subplots(figsize=(16, 10))  # figsize=(16, 10))


# We introduce all the EBL measurements
upper_lims_all, _ = import_cb_data(
    lambda_min_total=0,
    lambda_max_total=10,
    ax1=ax1, plot_measurs=True)



ax1.set_xlim(5e-6, 1e1)
ax1.set_ylim(5e-3, 120)


# legend22 = plt.legend(handlers, labels,
#                       loc=1,# bbox_to_anchor=(1., 0.3),
#                       title=r'Models', fontsize=16)


handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
for i in range(len(labels)):
    if labels[i].__contains__('LORRI'):
        print(handles[i].get_c())
        handles[i] = (plt.Line2D([], [], linestyle='',
                                 color=handles[i].get_c(), markerfacecolor='w',
                                 marker='*', markersize=16),
                      plt.Line2D([], [], linestyle='',
                                 color='k', markerfacecolor='k',
                                 marker='.', markersize=6)
                      )
legend11 = plt.legend(handles, labels,
                      handler_map={tuple: HandlerTuple(ndivide=1)},
                      title='Measurements', ncol=2, loc=2,
                      fontsize=11.5,
                      title_fontsize=20)  # , bbox_to_anchor=(1.001, 0.99))

ax1.add_artist(legend11)
# ax1.add_artist(legend22)

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

ax1.set_xlabel(r'Wavelength ($\mu$m)')

ax1.set_xscale('log')
ax1.set_yscale('log')
def tick_function(X):
    # return 17.5/X
    return (h_plank * c / X / u.micron).to(u.eV).value
def tick_function_2(X):
    # return 2.48/X
    return (h_plank * c / X / u.eV).to(u.micron).value
aaa = tick_function(2.48)
print(aaa)
print(tick_function_2(aaa))
ax3 = ax1.secondary_xaxis('top',
                         functions=(tick_function, tick_function_2))
ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel('Photon energy (eV)', labelpad=12)

plt.savefig('/home/porrassa/Desktop/ICRC2025/alps_ebl/cb1.pdf',
            bbox_inches='tight')
plt.savefig('/home/porrassa/Desktop/ICRC2025/alps_ebl/cb1.png',
            bbox_inches='tight')


handlers, labels = [], []
for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]

    plt.loglog(waves_ebl, model['callable_func'](waves_ebl),
               ls=model['ls'],
               c=model['color'], lw=model['linewidth'], zorder=2/(ni+1)
               )

    handlers.append(plt.Line2D([], [],
                               linewidth=model['linewidth'],
                               linestyle=model['ls'],
                               color=model['color']))
    labels.append(model['label'])


legend22 = plt.legend(handlers, labels,
                      loc=7, bbox_to_anchor=(1., 0.3),
                      title=r'Model', fontsize=16)
ax1.add_artist(legend11)
ax1.add_artist(legend22)

plt.savefig('/home/porrassa/Desktop/ICRC2025/alps_ebl/cb2.pdf',
            bbox_inches='tight')
plt.savefig('/home/porrassa/Desktop/ICRC2025/alps_ebl/cb2.png',
            bbox_inches='tight')


input_file_dir = ('outputs/outputs_dust_final_new/')
config_data = read_config_file(input_file_dir + 'input_data.yml')
# config_data = read_config_file(input_file_dir + 'input_example2.yml')
# config_data = read_config_file(input_file_dir + 'input_data_cob_fitted.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=True)

wv_alp, int_alp = ebl_class.ebl_axion_calculation(
    wavelength=waves_ebl, zz_array=0.,
    axion_mass=1e2,
    axion_gayy=1e-13
    )
plt.loglog(wv_alp, int_alp + spline_cuba(wv_alp),
         linestyle='-', color='green')


handlers.append(plt.Line2D([], [],
                           linewidth=2,
                           linestyle='-',
                           color='green'))
labels.append(r'CUBA + cosmic axion''\n '
              r'decay (example)''\n'
              r'    m$_a = 10^2$ eV''\n'
              r'    g$_{a\gamma} = 10^{-13}$ GeV$^{-1}$')

legend22 = plt.legend(handlers, labels,
                      loc=7, bbox_to_anchor=(1., 0.3),
                      title=r'Models', fontsize=16)
ax1.add_artist(legend11)
ax1.add_artist(legend22)

plt.savefig('/home/porrassa/Desktop/ICRC2025/alps_ebl/cb3.pdf',
            bbox_inches='tight')
plt.savefig('/home/porrassa/Desktop/ICRC2025/alps_ebl/cb3.png',
            bbox_inches='tight')



plt.show()
