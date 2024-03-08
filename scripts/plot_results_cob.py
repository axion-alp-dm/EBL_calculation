import os
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerTuple

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.import_cb_measurs import import_cb_data

from scipy.interpolate import UnivariateSpline

from astropy.constants import c

from ebltable.ebl_from_model import EBL

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
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
direct_name = str('outputs/final_outputs 2023-10-04 16:15:16'
                  )
print(direct_name)

# Choose the max and minimum wavelengthS of the data that we import
lambda_min_total = 0.  # [microns]
lambda_max_total = 5.  # [microns]

xlab = r'$m_a c^2$ (eV)'
ylab = r'$g_{a\gamma}$ (GeV$^{-1}$)'
g_min = 1.0e-20
g_max = 1.0e-7
m_min = 1.0e-2
m_max = 1.0e7
lw = 2.5
lfs = 30
tfs = 26
tickdir = 'in'
Grid = False
Shape = 'Custom'
figsize = (16.5, 14)
mathpazo = False
TopAndRightTicks = False
majorticklength = 13
minorticklength = 10
xtick_rotation = 20.0
tick_pad = 8
x_labelpad = 10
y_labelpad = 10
FrequencyAxis = False
N_Hz = 1
upper_xlabel = r"$\nu_a$ [Hz]"
plt.rcParams['axes.linewidth'] = lw
plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=tfs)


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


config_data = read_config_file(
    'scripts/input_files/input_data_paper2.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)

waves_ebl = np.logspace(np.log10(5e-6), 1, 3000)
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
    method_y = np.load(
        'outputs/pegase0.0001_Finkespline.npy',
        allow_pickle=True).item()
    return 10 ** method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                          grid=False)


def spline_starburst(lambda_array):
    method_y = np.load(
        'outputs/SB99_kneiskespline.npy',
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

ihl_spline = np.loadtxt('outputs/all_paramspace/ihl_spline.txt')

fig, ax = plt.subplots(figsize=(9, 7))

upper_lims_all, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax, plot_measurs=True)

plt.plot(ihl_spline[:, 0], ihl_spline[:, 1], 'k', linestyle='dotted', lw=2)
plt.plot(waves_ebl, spline_pegase0001(waves_ebl), 'k', linestyle='--', lw=2)
plt.plot(waves_ebl, spline_starburst(waves_ebl), 'r', ls='-', c='r')

print(6.383888174498693058e-01/(6.383888174498693058e-01+spline_pegase0001(
    5.))*100)

plt.plot(ihl_spline[:, 0], ihl_spline[:, 1]
         + spline_pegase0001(ihl_spline[:, 0]),
         'k', ls='-', lw=2)


linestyles = ['-','-', '--', 'dotted']
colors = ['r', 'k', 'k', 'k']
plt.legend([plt.Line2D([], [],
                       linewidth=2,
                       linestyle=linestyles[i],
                       color=colors[i])
            for i in range(4)],
           ['Model A', 'Model B + IHL', 'Model B',
            'IHL'],
           loc=4,  # bbox_to_anchor=(0.85, 0.01),
           fontsize=20)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 10)
ax.set_ylim(0.01, 120)

ax.set_ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')  # , labelpad=25)
ax.set_xlabel(r'Wavelength ($\mu$m)')  # , labelpad=10)

plt.savefig('outputs/figures_paper/ihl.png', bbox_inches='tight')
plt.savefig('outputs/figures_paper/ihl.pdf', bbox_inches='tight')

plt.show()

'''
direct_name_axionparams = 'outputs/new_axionparams 2024-02-23 10:46:37'
# params = np.load(direct_name_axionparams + '/axion_params.npy')
axion_mac2 = np.load(direct_name_axionparams + '/axion_mass.npy')
axion_gay = np.load(direct_name_axionparams + '/axion_gayy.npy')
list_models = np.load(direct_name_axionparams + '/list_models.npy')
'''
# Beginning of figure specifications
'''
# ---------------------------------------------------------------------
fig_params, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 12))
plt.subplots_adjust(wspace=0, hspace=0)
ebl_class.change_axion_contribution(mass=2.48 / 0.17, gayy=1.7e-11)

for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]
    ax.loglog(waves_ebl, model['callable_func'](waves_ebl),
              c=model['color'], lw=model['linewidth'],
              label=model['label'])
    ax.loglog(waves_ebl, (model['callable_func'](waves_ebl)
                          + 10 ** ebl_class.ebl_axion_spline(
                freq_array_ebl, 0., grid=False)),
              c=model['color'], lw=model['linewidth'],
              label=model['label'])

upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=ax, plot_measurs=True)


zorders = [1, 1, 3, 4, 2, 1, 1]
linewidths = [8, 8, 4, 4, 2, 1]
linestyles = ['-', '-', '-.', 'dotted']

for ni, model in enumerate(list_working_models.keys()):
    print(model)
    model_full = list_working_models[model]
    values_gay_array = np.load(direct_name_axionparams + '/' +
                               str(model) + '_params_UL.npy')

    cob_contours = ax2.contour(
        axion_mac2, axion_gay,
        (values_gay_array.T - np.min(values_gay_array)),
        levels=[4.61], origin='lower',
        linestyles=linestyles[ni],
        colors=model_full['color'], alpha=0.8,
        linewidths=5,  # linewidths[ni],
        zorder=1,
        label=model_full['label'],
        extent=(axion_mac2[0], axion_mac2[-1], None, None))

colors = ['b', 'r', 'orange', 'fuchsia', 'green']
linewidths = [3, 3, 2, 2, 2]
legend22 = plt.legend([plt.Line2D([], [],
                                  linewidth=linewidths[i],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(4)],
                      ['Model A', 'Model B',
                       'Finke22', 'CUBA'],
                      loc=4,  # bbox_to_anchor=(0.85, 0.01),
                      title=r'Models', fontsize=16)

ax.set_xlim(0.1, 6)
ax.set_ylim(0.6, 120)

ax.set_ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)', labelpad=25)

ax.set_xticks([1], [r'10$^0$'])
ax.tick_params(axis='x', direction='in', pad=-35)
ax.tick_params(axis='y', direction='in', pad=22)

ax.set_facecolor("none")
ax2.set_facecolor("none")

ax2.set_ylabel(ylab, fontsize=32, labelpad=y_labelpad)
ax2.set_xlabel(xlab, fontsize=26)

ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.add_artist(legend22)

ax2.set_ylim(6e-12, 2e-8)
ax2.set_xlim(2.48 / 0.1, 2.48 / 6)


def tick_function(X):
    return X


ax3 = ax.secondary_xaxis('top',
                         functions=(tick_function, tick_function))

ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel(r'Wavelength ($\mu$m)', labelpad=10)

ax4 = ax2.secondary_xaxis('top',
                          functions=(tick_function, tick_function))

ax4.tick_params(axis='x', direction='in', pad=-40)

plt.savefig('outputs/figures_paper/cb_zoom2.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_zoom2.png', bbox_inches='tight')
'''
# ---------------------------------------------------------------------
fig_params, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 15))
plt.subplots_adjust(hspace=0.27)

plt.subplot(212)


def tick_function_2(X):
    return 2.48 / X


ax.set_ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)', labelpad=25)

ax2.set_xlabel(xlab, fontsize=lfs, labelpad=x_labelpad)
ax2.set_ylabel(ylab, fontsize=lfs, labelpad=y_labelpad)

ax2.tick_params(which='major', direction=tickdir, width=2.5,
                length=majorticklength, right=TopAndRightTicks,
                top=TopAndRightTicks, pad=12)
ax2.tick_params(which='minor', direction=tickdir, width=1,
                length=minorticklength, right=TopAndRightTicks,
                top=TopAndRightTicks)

ax.set_xlim(0.1, 6)
ax.set_ylim(0.6, 120)

ax.tick_params(axis='y', direction='in', pad=22)

ax.set_yscale('log')
ax.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xscale('log')

ax2.set_ylim(6e-12, 2e-8)
ax2.set_xlim(2.48 / 0.1, 2.48 / 6)

ax.set_xlabel(r'Wavelength ($\mu$m)', labelpad=5)

ax3 = ax.secondary_xaxis('top',
                         functions=(tick_function_2, tick_function_2))

ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel(xlab, labelpad=10)

ax4 = ax2.secondary_xaxis('top',
                          functions=(tick_function_2, tick_function_2))

ax4.tick_params(axis='x', direction='in', pad=0)

locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=50)
locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                               numticks=100)
ax2.xaxis.set_major_locator(locmaj)
ax2.xaxis.set_minor_locator(locmin)
ax2.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=100)
locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                               numticks=100)

plt.yticks(10 ** (np.arange(-12, np.log10(g_max), 2)))
ax2.yaxis.set_major_locator(locmaj)
ax2.yaxis.set_minor_locator(locmin)
ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

zorders = [4, 3, 2, 1]
linestyles = ['-', '-', '--', 'dotted']
linewidths = [5, 5, 4, 3]

print(list_working_models)
# for ni, model in enumerate(list_working_models.keys()):
#     print(model)
#     model_full = list_working_models[model]
#     values_gay_array = np.load(direct_name_axionparams + '/' +
#                                str(model) + '_params_UL.npy')
#
#     cob_contours = ax2.contour(
#         axion_mac2, axion_gay,
#         (values_gay_array.T - np.min(values_gay_array)),
#         levels=[4.61], origin='lower',
#         linestyles=linestyles[ni],
#         colors=model_full['color'], alpha=0.9,
#         linewidths=linewidths[ni],
#         zorder=1,
#         label=model_full['label'],
#         extent=(axion_mac2[0], axion_mac2[-1], None, None))

# The COB models plotting
for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]
    ax.plot(waves_ebl, model['callable_func'](waves_ebl),
            c=model['color'], lw=model['linewidth'],
            label=model['label'])

upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=ax, plot_measurs=True)

ax2.legend([plt.Line2D([], [],
                       linestyle='-',
                       linewidth=list_working_models[model]['linewidth'],
                       color=list_working_models[model]['color'])
            for i, model in enumerate(list_working_models.keys())],
           list_working_models.keys())

handles, labels = ax2.get_legend_handles_labels()
linewidths = [3, 3, 2, 2]
for ni, model in enumerate(list_working_models.keys()):
    labels.append(list_working_models[model]['label'])
    handles.append(plt.Line2D([], [],
                              linestyle='-',
                              color=list_working_models[model]['color'],
                              linewidth=linewidths[ni]))
    if labels[ni].__contains__('Finke'):
        handles[ni] = (plt.Line2D([], [], linestyle='-',
                                  color='magenta', linewidth=linewidths[ni]),
                       plt.Line2D([], [], linestyle='--',
                                  color='magenta', linewidth=linewidths[ni])
                       )
    if labels[ni].__contains__('CUBA'):
        handles[ni] = (plt.Line2D([], [], linestyle='-',
                                  color='k', linewidth=linewidths[ni]),
                       plt.Line2D([], [], linestyle='dotted',
                                  color='k', linewidth=linewidths[ni])
                       )
print(handles, labels)
legend11 = ax2.legend(handles, labels,
                      handler_map={tuple: HandlerTuple(ndivide=2)},
                      loc=4, title=r'Models', fontsize=20)
ax2.add_artist(legend11)

plt.savefig('outputs/figures_paper/cb_zoom.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_zoom.png', bbox_inches='tight')
plt.show()

plt.show()
