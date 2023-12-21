import os
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.import_cb_measurs import import_cb_data

from scipy.interpolate import UnivariateSpline

from astropy.constants import c

from ebltable.ebl_from_model import EBL

# from axion_paramspace_calc import test_aa

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
direct_name = str('outputs/test/'
                  # + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
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

# ---------------------------------------------------------------------
fig_params, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 12))
# ax2 = ax.twinx()  # .twiny()
plt.subplots_adjust(wspace=0, hspace=0)


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


config_data = read_config_file('outputs/final_outputs 2023-10-04 16:15:16/'
                               'input_data.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)


waves_ebl = np.logspace(np.log10(5e-6), 1, 3000)
# waves_ebl = np.linspace(0.1, 10, num=5000)
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

def test_1(xx):
    return np.ones(np.shape(xx))
def test_10(xx):
    return np.ones(np.shape(xx))*10.
def test_nh(xx):
    return np.ones(np.shape(xx))*16.8

def spline_pegase0001(lambda_array):
    method_y = np.load('outputs/final_outputs 2023-10-04 16:15:16/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
    return 10**method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                    grid=False)


def spline_starburst(lambda_array):
    method_y = np.load('outputs/final_outputs 2023-10-04 16:15:16/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
    return 10**method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                    grid=False)


# list_working_models = np.load(
#     'outputs/test/dict_params.npy',
#                               allow_pickle=True).item()

list_working_models = {
    'ModelA': {'label': 'Model A', 'callable_func': spline_pegase0001,
               'color': 'b', 'linewidth': 3},
    'ModelB': {'label': 'Model B', 'callable_func': spline_starburst,
               'color': 'r', 'linewidth': 3},
    'Finke22': {'label': 'Finke+22', 'callable_func': spline_finke,
               'color': 'orange', 'linewidth': 2},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
               'color': 'fuchsia', 'linewidth': 2},
    # 'test1': {'label': 'test1', 'callable_func': test_1,
    #            'color': 'k', 'linewidth': 2},
    # 'test10': {'label': 'test10', 'callable_func': test_10,
    #            'color': 'green', 'linewidth': 2},
    # 'testnh': {'label': 'testnh', 'callable_func': test_nh,
    #            'color': 'violet', 'linewidth': 2}
}

# Beginning of figure specifications

ebl_class.change_axion_contribution(mass=2.48/1.45, gayy=5.76e-10)
# ax2.plot(1.45, 5.76e-10, marker='+', color='r')

for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]
    ax.loglog(waves_ebl, model['callable_func'](waves_ebl),
               c=model['color'], lw=model['linewidth'],
               label=model['label'])
    ax.loglog(waves_ebl, (model['callable_func'](waves_ebl)
                          + 10**ebl_class.ebl_axion_spline(
                freq_array_ebl, 0., grid=False)),
               c=model['color'], lw=model['linewidth'],
               label=model['label'])

upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=ax, plot_measurs=True)


params = np.load(direct_name + '/axion_params.npy')
axion_mac2 = params[:, 0]
axion_gay = params[:, 1]
list_models = np.load(direct_name + '/list_models.npy')

zorders = [1, 1, 4, 3, 2, 1, 1]
linewidths = [8, 8, 4, 4, 2, 1]

for ni, model in enumerate(list_working_models.keys()):
    print(model)
    model_full = list_working_models[model]
    values_gay_array = np.load(direct_name + '/' +
                               str(model) + '_params_UL.npy')

    cob_contours = ax2.contour(
        axion_mac2, axion_gay,
        (values_gay_array.T - np.min(values_gay_array)),
        levels=[4.61], origin='lower', linestyles='-',
        colors=model_full['color'], alpha=0.8,
        linewidths=linewidths[ni],
        zorder=1,
        label=model_full['label'],
        extent=(axion_mac2[0], axion_mac2[-1], None, None))


values_gay_array = np.load(direct_name + '/test10' + '_params_UL.npy')
cob10_contours = plt.contour(
        2.48 / axion_mac2, axion_gay,
        np.log10(values_gay_array.T - np.min(values_gay_array)),
        levels=[np.log10(4.61)], origin='lower', alpha=0)
cob10_contours = cob10_contours.allsegs[0][0][::-1]
spline10 = UnivariateSpline(cob10_contours[:, 0],
                            cob10_contours[:, 1], k=1, s=0)
values_gay_array = np.load(direct_name + '/testnh' + '_params_UL.npy')
cobnh_contours = plt.contour(
        2.48 / axion_mac2, axion_gay,
        np.log10(values_gay_array.T - np.min(values_gay_array)),
        levels=[np.log10(4.61)], origin='lower', alpha=0)

cobnh_contours = cobnh_contours.allsegs[0][0][::-1]
splinenh = UnivariateSpline(cobnh_contours[:, 0],
                            cobnh_contours[:, 1], k=1, s=0)



colors = ['b', 'r', 'orange', 'fuchsia', 'green']
linewidths = [3, 3, 2, 2, 2]
legend22 = plt.legend([plt.Line2D([], [],
                                  linewidth=linewidths[i],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(4)],
                      ['Model A', 'Model B',
                       'Finke22', 'CUBA'],
                      loc=4, #bbox_to_anchor=(0.85, 0.01),
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

# ax2.set_ylim([1e-12, g_max])
ax2.set_ylim(6e-12, 2e-8)
ax2.set_xlim(2.48/0.1, 2.48/6)


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
plt.show()
'''
# ---------------------------------------------------------------------
fig_params, (ax, ax2) = plt.subplots(1, 2, figsize=(26, 9))

plt.subplot(122)
ax2.set_xlabel(xlab, fontsize=lfs, labelpad=x_labelpad)
ax2.set_ylabel(ylab, fontsize=lfs, labelpad=y_labelpad)
ax2.tick_params(which='major', direction=tickdir, width=2.5,
                length=majorticklength, right=TopAndRightTicks,
                top=TopAndRightTicks, pad=tick_pad)
ax2.tick_params(which='minor', direction=tickdir, width=1,
                length=minorticklength, right=TopAndRightTicks,
                top=TopAndRightTicks)

ax2.set_yscale('log')
ax2.set_xscale('log')

ax2.set_xlim(2.48 / 6, 2.48 / 0.1)
ax2.set_ylim([1e-12, g_max])

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
# plt.xticks(rotation=xtick_rotation)
direct_name = '/home/porrassa/Desktop/EBL_ModelCode/EBL_calculation/' \
              'outputs/test'
# direct_name = '/home/porrassa/Desktop/EBL_ModelCode/EBL_calculation/' \
#         'outputs/final_outputs_clumpy_vdispersion'
params = np.load(direct_name + '/axion_params.npy')
axion_mac2 = params[:, 0]
axion_gay = params[:, 1]
list_models = np.load(direct_name + '/list_models.npy')
print(list_models)
list_colors = ['b', 'r', 'orange', 'fuchsia', 'green']
zorders = [4, 3, 2, 1]

for ni, model in enumerate(list_models[:, 0]):
    values_gay_array = np.load(direct_name + '/' +
                               str(model) + '_params_UL.npy')
    values_gay_array_NH = np.load(direct_name + '/' +
                                  str(model) + '_params_measur.npy')

    ax2.contour(axion_mac2, axion_gay,
                (values_gay_array.T - np.min(values_gay_array)),
                levels=[4.61], origin='lower', linestyle='-',
                colors=list_colors[ni], alpha=0.9, linewidths=5,
                zorder=zorders[ni],
                label=model)

plt.legend([plt.Line2D([], [], linestyle='-', linewidth=3,
                       color=list_colors[i])
            for i in range(4)],
           list_models[:, 1])


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

# Parameter space for axion characteristics and rest of necessary arrays
axion_mac2 = np.logspace(0, 7, num=250)
axion_gay = np.logspace(-12, -9, num=250)

waves_ebl = np.logspace(np.log10(5e-6), 1, 3000)
# waves_ebl = np.linspace(0.1, 10, num=5000)
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
    method_y = np.load('outputs/final_outputs 2023-10-04 16:15:16/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
    return method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                    grid=False)


def spline_starburst(lambda_array):
    method_y = np.load('outputs/final_outputs 2023-10-04 16:15:16/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
    return method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                    grid=False)


list_working_models = {
    'ModelA': {'label': 'Model A', 'callable_func': spline_pegase0001},
    'ModelB': {'label': 'Model B', 'callable_func': spline_starburst},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba},
    'Finke22': {'label': 'Finke+22', 'callable_func': spline_finke}
}

# Beginning of figure specifications
plt.subplot(121)

plt.loglog(waves_ebl, spline_cuba(waves_ebl), c='fuchsia')
plt.loglog(waves_ebl, spline_finke(waves_ebl), c='orange')
plt.loglog(waves_ebl, 10 ** spline_pegase0001(waves_ebl), c='r', lw=3)
plt.loglog(waves_ebl, 10 ** spline_starburst(waves_ebl), c='b', lw=3)

upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=ax, plot_measurs=True)

plt.xlim(0.1, 6)
plt.ylim(0.6, 30)

colors = ['b', 'r', 'orange', 'fuchsia', 'green']
linewidths = [3, 3, 2, 2, 2]
legend22 = plt.legend([plt.Line2D([], [],
                                  linewidth=linewidths[i],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(4)],
                      ['Model A', 'Model B',
                       'Finke22', 'CUBA'],
                      loc=8, bbox_to_anchor=(0.65, 0.01),
                      title=r'Models', fontsize=16)

plt.savefig('outputs/figures_paper/cb_zoom.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_zoom.png', bbox_inches='tight')
plt.show()
'''
