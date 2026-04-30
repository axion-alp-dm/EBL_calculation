# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data, dictionary_datatype

from astropy import units as u
import astropy.constants as c

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


# We initialize the class with the input file
# config_data = read_config_file('outputs/final_outputs_Zevol_fixezZsolar '
#                        '2024-04-11 13:41:34/' + 'input_data.yml')
# ebl_class = EBL_model.input_yaml_data_into_class(config_data,
#                                                  log_prints=True)
# ebl_class.ebl_ssp_calculation(
#     config_data['ssp_models']['SB99_dustFinke'])


waves_ebl = np.geomspace(5e-6, 1e4, num=int(1e6))
# waves_ebl = np.geomspace(0.08, 1e4, num=int(1e6))
freq_array_ebl = np.log10(c.c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)




# def spline_starburst(lambda_array):
#     return 10 ** ebl_class.ebl_ssp_spline(
#         np.log10(c.c.value * 1e6 / lambda_array), 0.,
#                           grid=False)


list_working_models = {
    # 'ModelA': {'label': 'Our model', 'callable_func': spline_starburst,
    #            'color': 'b', 'linewidth': 3, 'ls': '-'},
    # 'Finke22': {'label': 'Finke22', 'callable_func': spline_finke,
    #             'color': 'fuchsia', 'linewidth': 2, 'ls': '--'},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba,
             'color': 'k', 'linewidth': 2.5, 'ls': 'dotted'}
}

plt.figure(figsize=(12, 3))

plt.text(x=0.05, y=0.5,
         s=r'$\nu I_{\nu}(\lambda,z) = $'
           r'$\frac{c^2}{4\pi\lambda}\int_{z}^{z_\mathrm{max}}$'
           r'$ \varepsilon_{\nu\prime }$'
           r'$\left(\lambda\frac{1+z}{1+z\prime }, z\prime \right)$'
           r'$\,\frac{1}{(1+z\prime ) H(z\prime )}\, $'
             r'$\mathrm{d}z\prime $',
         fontsize=35
         )

plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.savefig('outputs/figures_paper/nuInu_formula.png',
            bbox_inches='tight', dpi=500)
# plt.show()

plt.figure(figsize=(12, 3))

plt.text(x=0.05, y=0.5,
         s=r'$\varepsilon_{\nu_\mathrm{stellar}}(\lambda, z) $'
           r'$= \int^{z_\mathrm{max}}_{z}L_{\nu}^\mathrm{SSP}$'
           r'$\left(\lambda, \tau_\star, Z\right)$'
           r'$\,\rho_{\star}(z\prime )$'
           r'$\,\frac{1}{(1+z\prime ) H(z\prime )}\, \mathrm{d}z\prime $',
         fontsize=30
         )

plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.savefig('outputs/figures_paper/emissivity_formula.png',
            bbox_inches='tight', dpi=500)

plt.figure(figsize=(10, 10))
xx = np.linspace(0.15, 0.5, num=200)
yy = np.sin((xx-0.5) * 70) * 0.02 + 0.5
plt.plot(xx, yy, c='k', lw=3)
yy = np.cos((xx) * 70) * 0.02 + 0.52
plt.plot(yy, xx, c='k', lw=3)
plt.annotate(text='', xy=(0.5, 0.5), xytext=(0.8, 0.6),
             arrowprops=dict(arrowstyle='<-', color='k', lw=3),
             alpha=1, zorder=-10)
plt.annotate(text='', xy=(0.5, 0.5), xytext=(0.8, 0.4),
             arrowprops=dict(arrowstyle='<-', color='k', lw=3),
             alpha=1, zorder=-10)

plt.annotate(text=r'$\gamma$ (VHE)', xy=(0.26, 0.55), color='k',
             fontsize=28)
plt.annotate(text=r'$\gamma$ (EBL)', xy=(0.57, 0.21), color='k',
             fontsize=28)
plt.annotate(text=r'e$^{+}$', xy=(0.82, 0.6), color='k',
             fontsize=28)
plt.annotate(text=r'e$^{-}$', xy=(0.82, 0.38), color='k',
             fontsize=28)
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.savefig('outputs/figures_paper/pair_production.png',
            bbox_inches='tight', dpi=500)
# plt.show()

plt.figure(figsize=(8, 8))
plt.text(0.05, 0.5, s=r'$\tau (E_0, z_0) = \int_0 ^{z_0} \mathrm{d}z\,$'
                      r'$ \frac{\mathrm{d} L}{\mathrm{d} z}(z)$'
                     r'$ \int_0 ^{\infty} \mathrm{d}\epsilon$'
                      r' $\frac{\mathrm{d} n}{\mathrm{d} \epsilon}$'
                     r'$ (\epsilon, z)$')
plt.text(0.3, 0.38, s=r'$\int_{-1} ^{1} \mathrm{d}\mu \,\frac{1 - \mu}{2} $'
                      r'$\,\sigma_{\gamma \gamma} $'
                     r'$\left[\beta\left(E_0, z, \epsilon, \mu\right)\right]$')

plt.savefig('outputs/figures_paper/tau_from_ebl.png',
            bbox_inches='tight', dpi=500)

plt.figure(figsize=(8, 8))
plt.text(0.05, 0.5, s=r' $\frac{\mathrm{d} n}{\mathrm{d} \epsilon}$'
                     r'$ = \frac{c}{4 \pi} \epsilon^2 \, \nu I_{\nu}$',
         fontsize=60)

plt.savefig('outputs/figures_paper/dnde_nuInu.png',
            bbox_inches='tight', dpi=500)

# plt.show()

# Beginning of figure specifications
fig, ax1 = plt.subplots(figsize=(16, 10))  # figsize=(16, 10))


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

# ebl_class.change_axion_contribution(1e2, 1e-13)
# plt.loglog(waves_ebl,
#            (10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0.,
#                                              grid=False)
#             + spline_cuba(waves_ebl)), c='green', zorder=0.5, ls='-')

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
    lambda_min_total=0,
    lambda_max_total=1e4,
    ax1=ax1, plot_measurs=True)



ax1.set_xlim(5e-6, 1e1)
ax1.set_ylim(5e-3, 120)
# ax1.set_xlim(0.08, 1235)
# ax1.set_ylim(0.13, 120)
legend22 = plt.legend(handlers, labels,
                      loc=7, bbox_to_anchor=(1., 0.3),
                      title=r'Models', fontsize=16)
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
# legend11 = plt.legend(handles, labels,
#                       handler_map={tuple: HandlerTuple(ndivide=1)},
#                       title='Measurements', ncol=1, loc=2,
#                       fontsize=10,
#                       title_fontsize=15, bbox_to_anchor=(1.01, 1.05))

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
    return (c.h * c.c / X / u.micron).to(u.eV).value
def tick_function_2(X):
    # return 2.48/X
    return (c.h * c.c / X / u.eV).to(u.micron).value
aaa = tick_function(2.48)
print(aaa)
print(tick_function_2(aaa))
ax3 = ax1.secondary_xaxis('top',
                         functions=(tick_function, tick_function_2))
ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel('Photon energy (eV)', labelpad=12)

plt.savefig('outputs/figures_paper/cb.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb.png', bbox_inches='tight')


# Beginning of figure specifications
fig, ax1 = plt.subplots(figsize=(12, 8))

# We introduce all the EBL measurements
upper_lims_all, igl_ebldata = import_cb_data(
    lambda_min_total=0.08,
    lambda_max_total=1e4,
    ax1=ax1, plot_measurs=True,
obs_not_taken=['ALMA (Fujimoto+ ‘16)',
               'SCUBA-2 (Hsu+ ‘16)',
               'ISO/ISOCAM (Clements+ ‘99)',
               'NH/LORRI (Symons+ ‘23)'])

upper_lims_all.sort(['lambda'])
aaa = np.where(upper_lims_all['lambda'][1:] == upper_lims_all['lambda'][:-1])[0]
upper_lims_all['nuInu'][aaa] = np.min(
    (upper_lims_all['nuInu'][aaa], upper_lims_all['nuInu'][aaa+1]), axis=0)
upper_lims_all.remove_rows([aaa+1])
upper_lims_all.remove_row(
    np.where(upper_lims_all['ref']=='NH/LORRI (Postman+ ‘24)')[0][0])
aaa = np.where(upper_lims_all['nuInu'] < 1e-6)[0]
upper_lims_all['nuInu'][aaa] = upper_lims_all['1 sigma'][aaa]

spline_upper = UnivariateSpline(
    np.log10(upper_lims_all['lambda']),
    np.log10(upper_lims_all['nuInu']), #+upper_lims_all['1 sigma']),
    k=1, s=0, ext=3)


igl_ebldata.sort(['lambda'])
aaa = np.where(igl_ebldata['lambda'][1:] == igl_ebldata['lambda'][:-1])[0]
igl_ebldata['nuInu'][aaa] = np.min(
    (igl_ebldata['nuInu'][aaa], igl_ebldata['nuInu'][aaa+1]), axis=0)
igl_ebldata.remove_rows([aaa+1])

spline_lower = UnivariateSpline(
    np.log10(igl_ebldata['lambda']),
    np.log10(igl_ebldata['nuInu']), #+igl_ebldata['1 sigma']),
    k=1, s=0, ext=3)

handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]

# legend11 = plt.legend(handles, labels,
#                       ncol=1, loc=6,
#                       fontsize=14,
#                       bbox_to_anchor=(1.03, 0.5))

waves_fine = np.geomspace(0.09, 1e3, num=500)

# plt.fill_between(
#     x=waves_fine,
#     y1=10**spline_lower(np.log10(waves_fine)),
#     y2=10**spline_upper(np.log10(waves_fine)),
# color='gray', zorder=0, alpha=0.2, lw=0)

plt.annotate(text='', xy=(0.09, 0.9), xytext=(5, 0.9),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='', xy=(5, 0.9), xytext=(1e3, 0.9),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='Optical', xy=(1, 1), alpha=0.7, color='grey')
plt.annotate(text='Infrared', xy=(60, 1), alpha=0.7, color='grey')

ax1.set_xlim(0.09, 1e3)
ax1.set_ylim(0.8, 120)

# ax1.add_artist(legend11)

ax1.set_xlabel(r'Wavelength (µm)')

ax1.set_xscale('log')
# ax1.set_yscale('log')
def tick_function(X):
    return (c.h * c.c / X / u.micron).to(u.eV).value
def tick_function_2(X):
    return (c.h * c.c / X / u.eV).to(u.micron).value

aaa = tick_function(2.48)
ax3 = ax1.secondary_xaxis('top',
                         functions=(tick_function, tick_function_2))
ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel('Photon energy (eV)', labelpad=12)


plt.savefig('outputs/figures_paper/cb_measursIGL.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_measursIGL.png',
            bbox_inches='tight', dpi=1000)

# Beginning of figure specifications
fig, ax1 = plt.subplots(figsize=(12, 8))

# We introduce all the EBL measurements
upper_lims_all, igl_ebldata = import_cb_data(
    lambda_min_total=0.08,
    lambda_max_total=1e4,
    ax1=ax1, plot_measurs=True,
obs_not_taken=['ALMA (Fujimoto+ ‘16)',
               'SCUBA-2 (Hsu+ ‘16)',
               'ISO/ISOCAM (Clements+ ‘99)',
               'NH/LORRI (Symons+ ‘23)'])

upper_lims_all.sort(['lambda'])
aaa = np.where(upper_lims_all['lambda'][1:] == upper_lims_all['lambda'][:-1])[0]
upper_lims_all['nuInu'][aaa] = np.min(
    (upper_lims_all['nuInu'][aaa], upper_lims_all['nuInu'][aaa+1]), axis=0)
upper_lims_all.remove_rows([aaa+1])
upper_lims_all.remove_row(
    np.where(upper_lims_all['ref']=='NH/LORRI (Postman+ ‘24)')[0][0])
aaa = np.where(upper_lims_all['nuInu'] < 1e-6)[0]
upper_lims_all['nuInu'][aaa] = upper_lims_all['1 sigma'][aaa]

spline_upper = UnivariateSpline(
    np.log10(upper_lims_all['lambda']),
    np.log10(upper_lims_all['nuInu']), #+upper_lims_all['1 sigma']),
    k=1, s=0, ext=3)


igl_ebldata.sort(['lambda'])
aaa = np.where(igl_ebldata['lambda'][1:] == igl_ebldata['lambda'][:-1])[0]
igl_ebldata['nuInu'][aaa] = np.min(
    (igl_ebldata['nuInu'][aaa], igl_ebldata['nuInu'][aaa+1]), axis=0)
igl_ebldata.remove_rows([aaa+1])

spline_lower = UnivariateSpline(
    np.log10(igl_ebldata['lambda']),
    np.log10(igl_ebldata['nuInu']), #+igl_ebldata['1 sigma']),
    k=1, s=0, ext=3)

handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]

# legend11 = plt.legend(handles, labels,
#                       ncol=1, loc=6,
#                       fontsize=14,
#                       bbox_to_anchor=(1.03, 0.5))

waves_fine = np.geomspace(0.09, 1e3, num=500)

plt.fill_between(
    x=waves_fine,
    y1=10**spline_lower(np.log10(waves_fine)),
    y2=10**spline_upper(np.log10(waves_fine)),
color='gray', zorder=0, alpha=0.2, lw=0)

plt.annotate(text='', xy=(0.09, 0.9), xytext=(5, 0.9),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='', xy=(5, 0.9), xytext=(1e3, 0.9),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='Optical', xy=(1, 1), alpha=0.7, color='grey')
plt.annotate(text='Infrared', xy=(60, 1), alpha=0.7, color='grey')

ax1.set_xlim(0.09, 1e3)
ax1.set_ylim(0.8, 120)

# ax1.add_artist(legend11)

ax1.set_xlabel(r'Wavelength (µm)')

ax1.set_xscale('log')
# ax1.set_yscale('log')
def tick_function(X):
    return (c.h * c.c / X / u.micron).to(u.eV).value
def tick_function_2(X):
    return (c.h * c.c / X / u.eV).to(u.micron).value

aaa = tick_function(2.48)
ax3 = ax1.secondary_xaxis('top',
                         functions=(tick_function, tick_function_2))
ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel('Photon energy (eV)', labelpad=12)


plt.savefig('outputs/figures_paper/cb_measurs.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_measurs.png',
            bbox_inches='tight', dpi=1000)

my_ebl = ['bosa.txt', 'chary.txt', '2bb.txt']


waves_ebl = np.geomspace(0.05, 1e3, num=int(1e4))


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

dataarray = c.c / 4. / np.pi * 10**energyarray * u.eV * 10**dataarray / u.cm**3
dataarray = dataarray.to(u.nW*u.m**-2) / (1 + zz_array[:, np.newaxis])**3
wavelength_array = (c.h*c.c/(10**(energyarray)*u.eV)).to(u.micron)


sort_array = np.argsort(wavelength_array)
dataarray = (dataarray.value)[:, sort_array]
wavelength_array = (wavelength_array.value)[sort_array]

aabigsline = RectBivariateSpline(
    x=zz_array, y=wavelength_array, z=dataarray, kx=1, ky=1, s=0)

ebl_franccc = EBL(z=zz_array, lmu=wavelength_array,
                nuInu=dataarray.T, model='franc')

plt.plot(waves_fine,
         ebl_finke.ebl_array(z=0, lmu=waves_fine),
                 linestyle='dotted', lw=3, label='Finke+22', c='fuchsia')
plt.plot(waves_fine,
                 ebl_SL.ebl_array(z=0, lmu=waves_fine),
                 linestyle='-.', lw=3, label='Saldana-Lopez+21', c='r')
waves_fran = np.geomspace(0.1, 250)
plt.plot(waves_fran,
                 ebl_franccc.ebl_array(z=0, lmu=waves_fran),
                 label='Franceschini+17', linestyle='--', lw=3, c='k')

legend33 = ax1.legend([
    plt.Line2D([], [], linewidth=3, linestyle='dotted', color='fuchsia'),
    plt.Line2D([], [], linewidth=3, linestyle='-.', color='r'),
    plt.Line2D([], [], linewidth=3, linestyle='--', color='k')],
    ['Finke+22', 'Saldana-Lopez+21', 'Franceschini+17'],
    loc=1, fontsize=18
    )
ax1.add_artist(legend33)

plt.savefig('outputs/figures_paper/cb_measurs_models.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_measurs_models.png',
            bbox_inches='tight', dpi=1000)
# plt.show()
# ----------------------------------------------------------------------

# Beginning of figure specifications
fig, ax1 = plt.subplots(figsize=(10, 6.5))

# We introduce all the EBL measurements
upper_lims_all, igl_ebldata = import_cb_data(
    lambda_min_total=0,
    lambda_max_total=1e4,
    ax1=ax1, plot_measurs=True,
obs_not_taken=['ALMA (Fujimoto+ ‘16)',
               'SCUBA-2 (Hsu+ ‘16)',
               'ISO/ISOCAM (Clements+ ‘99)'])


handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]

legend11 = plt.legend(handles, labels,
                      ncol=1, loc=6,
                      fontsize=14,
                      bbox_to_anchor=(1.03, 0.5))

handles_lines = []
labels_lines = []

list_working_models = {
    'ModelBosa': {'label': 'BOSA',
               'callable_func': 'bosa',
               'color': 'b'},
    'ModelChary': {'label': 'Chary',
               'callable_func': 'chary',
               'color': 'orange'},
    'Model3body': {'label': '2BB',
               'callable_func': '2bb',
               'color': 'g'},
}

f_color = {
    'ModelBosa': 'dodgerblue',
    'ModelChary': 'C1',
    'Model3body':'green',
}

## 10% systematics
# data = {
#   'ModelChary': {
#     'x': [1.00000000e-01, 1.03128316e-01, 1.06354496e-01, 1.09681601e-01,
#           1.13112788e-01, 1.16651313e-01, 1.20300535e-01, 1.24063916e-01,  1.27945027e-01, 1.31947552e-01, 1.36075289e-01, 1.40332154e-01,  1.44722187e-01, 1.49249555e-01, 1.53918552e-01, 1.58733611e-01,  1.63699300e-01, 1.68820332e-01, 1.74101565e-01, 1.79548012e-01,  1.85164842e-01, 1.90957383e-01, 1.96931134e-01, 2.03091762e-01,  2.09445114e-01, 2.15997219e-01, 2.22754295e-01, 2.29722754e-01,  2.36909207e-01, 2.44320476e-01, 2.51963593e-01, 2.59845810e-01,  2.67974609e-01, 2.76357701e-01, 2.85003044e-01, 2.93918840e-01,  3.03113550e-01, 3.12595900e-01, 3.22374888e-01, 3.32459793e-01,  3.42860186e-01, 3.53585937e-01, 3.64647222e-01, 3.76054540e-01,  3.87818715e-01, 3.99950910e-01, 4.12462638e-01, 4.25365773e-01,  4.38672559e-01, 4.52395623e-01, 4.66547988e-01, 4.81143084e-01,  4.96194760e-01, 5.11717301e-01, 5.27725435e-01, 5.44234355e-01,  5.61259726e-01, 5.78817704e-01, 5.96924951e-01, 6.15598650e-01,  6.34856522e-01, 6.54716840e-01, 6.75198452e-01, 6.96320794e-01,  7.18103909e-01, 7.40568469e-01, 7.63735792e-01, 7.87627861e-01,  8.12267350e-01, 8.37677640e-01, 8.63882844e-01, 8.90907830e-01,  9.18778243e-01, 9.47520530e-01, 9.77161967e-01, 1.00773068e+00,  1.03925568e+00, 1.07176689e+00, 1.10529514e+00, 1.13987227e+00,  1.17553107e+00, 1.21230540e+00, 1.25023015e+00, 1.28934130e+00,  1.32967597e+00, 1.37127243e+00, 1.41417017e+00, 1.45840988e+00,  1.50403355e+00, 1.55108448e+00, 1.59960730e+00, 1.64964807e+00,  1.70125428e+00, 1.75447489e+00, 1.80936041e+00, 1.86596292e+00,  1.92433614e+00, 1.98453546e+00, 2.04661800e+00, 2.11064268e+00,  2.17667025e+00, 2.24476338e+00, 2.31498667e+00, 2.38740677e+00,  2.46209240e+00, 2.53911443e+00, 2.61854596e+00, 2.70046235e+00,  2.78494135e+00, 2.87206312e+00, 2.96191033e+00, 3.05456825e+00,  3.15012480e+00, 3.24867066e+00, 3.35029934e+00, 3.45510729e+00,  3.56319397e+00, 3.67466194e+00, 3.78961698e+00, 3.90816818e+00,  4.03042803e+00, 4.15651256e+00, 4.28654141e+00, 4.42063797e+00,  4.55892950e+00, 4.70154722e+00, 4.84862648e+00, 5.00030684e+00,  5.15673224e+00, 5.31805113e+00, 5.48441658e+00, 5.65598646e+00,  5.83292359e+00, 6.01539588e+00, 6.20357648e+00, 6.39764396e+00,  6.59778248e+00, 6.80418197e+00, 7.01703829e+00, 7.23655342e+00,  7.46293569e+00, 7.69639990e+00, 7.93716762e+00, 8.18546731e+00,  8.44153460e+00, 8.70561248e+00, 8.97795155e+00, 9.25881025e+00,  9.54845510e+00, 9.84716096e+00, 1.01552113e+01, 1.04728984e+01,  1.08005237e+01, 1.11383983e+01, 1.14868426e+01, 1.18461873e+01,  1.22167735e+01, 1.25989528e+01, 1.29930878e+01, 1.33995527e+01,  1.38187331e+01, 1.42510267e+01, 1.46968439e+01, 1.51566076e+01,  1.56307542e+01, 1.61197336e+01, 1.66240098e+01, 1.71440614e+01,  1.76803818e+01, 1.82334800e+01, 1.88038809e+01, 1.93921257e+01,  1.99987727e+01, 2.06243975e+01, 2.12695939e+01, 2.19349740e+01,  2.26211693e+01, 2.33288310e+01, 2.40586305e+01, 2.48112606e+01,  2.55874352e+01, 2.63878910e+01, 2.72133877e+01, 2.80647085e+01,  2.89426612e+01, 2.98480792e+01, 3.07818214e+01, 3.17447741e+01,  3.27378510e+01, 3.37619944e+01, 3.48181763e+01, 3.59073989e+01,  3.70306958e+01, 3.81891330e+01, 3.93838098e+01, 4.06158599e+01,  4.18864524e+01, 4.31967930e+01, 4.45481252e+01, 4.59417313e+01,  4.73789339e+01, 4.88610967e+01, 5.03896263e+01, 5.19659730e+01,  5.35916329e+01, 5.52681486e+01, 5.69971109e+01, 5.87801607e+01,  6.06189899e+01, 6.25153435e+01, 6.44710211e+01, 6.64878784e+01,  6.85678294e+01, 7.07128478e+01, 7.29249692e+01, 7.52062927e+01,  7.75589832e+01, 7.99852734e+01, 8.24874655e+01, 8.50679342e+01,  8.77291280e+01, 9.04735724e+01, 9.33038717e+01, 9.62227117e+01,  9.92328623e+01, 1.02337180e+02, 1.05538610e+02, 1.08840192e+02,  1.12245057e+02, 1.15756437e+02, 1.19377664e+02, 1.23112175e+02,  1.26963513e+02, 1.30935333e+02, 1.35031404e+02, 1.39255613e+02,  1.43611969e+02, 1.48104605e+02, 1.52737785e+02, 1.57515906e+02,  1.62443501e+02, 1.67525247e+02, 1.72765966e+02, 1.78170632e+02,  1.83744372e+02, 1.89492477e+02, 1.95420401e+02, 2.01533769e+02,  2.07838382e+02, 2.14340223e+02, 2.21045463e+02, 2.27960464e+02,  2.35091788e+02, 2.42446202e+02, 2.50030685e+02, 2.57852435e+02,  2.65918874e+02, 2.74237657e+02, 2.82816678e+02, 2.91664078e+02,  3.00788252e+02, 3.10197859e+02, 3.19901828e+02, 3.29909369e+02,  3.40229977e+02, 3.50873446e+02, 3.61849876e+02, 3.73169684e+02,  3.84843611e+02, 3.96882735e+02, 4.09298482e+02, 4.22102632e+02,  4.35307336e+02, 4.48925126e+02, 4.62968923e+02, 4.77452054e+02,  4.92388263e+02, 5.07791724e+02, 5.23677054e+02, 5.40059328e+02,  5.56954091e+02, 5.74377375e+02, 5.92345715e+02, 6.10876161e+02,  6.29986298e+02, 6.49694260e+02, 6.70018750e+02, 6.90979055e+02,  7.12595063e+02, 7.34887289e+02, 7.57876886e+02, 7.81585671e+02,  8.06036141e+02, 8.31251499e+02, 8.57255673e+02, 8.84073340e+02,  9.11729948e+02, 9.40251743e+02, 9.69665789e+02, 1.00000000e+03],
#     'y': [0.05958889, 0.05746732, 0.11198388, 0.16656535, 0.17583196,
#           0.24021026,  0.30136606, 0.3286781, 0.38278416, 0.47429655, 0.55173881, 0.61228492,  0.70636761, 0.77210427, 0.8766975, 0.94148174, 1.00917984, 1.12372907,  1.20303757, 1.30059529, 1.40418208, 1.49506855, 1.59345756, 1.68690047,  1.7795219, 1.89371696, 1.97273388, 2.0771484, 2.16360011, 2.24805721,  2.32537815, 2.40784171, 2.49208871, 2.57305675, 2.65330235, 2.72852733,  2.81055952, 2.89722006, 2.97713576, 3.06825404, 3.14813209, 3.22821815,  3.32098624, 3.3894626, 3.5083516, 3.64299109, 3.8186818, 3.97673057,  4.16771692, 4.35999739, 4.5823839, 4.85076448, 5.0794805, 5.31805514,  5.59524397, 5.80084066, 6.12317979, 6.39999129, 6.66598483, 6.98345515,  7.28566643, 7.5732971, 7.84973998, 8.16808926, 8.39596095, 8.68371777,  8.96088065, 9.24212565, 9.48589511, 9.73925226, 10.03662876, 10.24073737,  10.51080717, 10.75494504, 10.95780404, 11.17527751, 11.34677275, 11.4767182,  11.62366398, 11.71753346, 11.78280817, 11.84843727, 11.85066004, 11.87804107,  11.86440835, 11.8372401, 11.79096513, 11.73953394, 11.69630301, 11.63873175,  11.59661141, 11.50511437, 11.43472146, 11.32195628, 11.1786384, 11.01329112,  10.81387394, 10.58731407, 10.3701595, 10.1058198, 9.8220435, 9.54340014,  9.23091889, 8.9191516, 8.557809, 8.20015215, 7.87705356, 7.53146278,  7.19473226, 6.8777321, 6.55094076, 6.24354504, 5.93804823, 5.6302007,  5.34734672, 5.05502028, 4.78423554, 4.51723626, 4.25683231, 4.00946422,  3.77097253, 3.54693749, 3.32575617, 3.11752166, 2.90935105, 2.715693,  2.53488766, 2.3600808, 2.19694546, 2.04754352, 1.9256971, 1.8601603,  1.77389016, 1.71327179, 1.69944759, 1.64999965, 1.62796841, 1.64291291,  1.60512604, 1.61010558, 1.63794641, 1.6782055, 1.8113143, 1.84478457,  1.95107578, 1.98992934, 2.09371757, 2.07473526, 2.14707428, 2.15262546,  2.18663694, 2.20912279, 2.24709702, 2.33635387, 2.38990256, 2.55324243,  2.57673821, 2.63142967, 2.79948634, 2.82533385, 2.82042297, 2.93340518,  2.90014926, 2.92777245, 2.91805066, 2.91528966, 2.94505659, 2.95667735,  2.98307459, 2.93203773, 2.96808093, 2.90436858, 2.91978218, 2.87818447,  2.87334357, 2.86773176, 2.85627335, 2.86745813, 2.83440154, 2.84876701,  2.86063992, 2.859805, 2.88263872, 2.90525425, 2.95746878, 2.98718346,  3.03013832, 3.09488537, 3.15564051, 3.24055386, 3.3276828, 3.42113281,  3.52765272, 3.64432569, 3.76978848, 3.90706483, 4.05704755, 4.2167128,  4.38770015, 4.56965555, 4.76091959, 4.96277507, 5.17360761, 5.39317829,  5.62107223, 5.85726655, 6.10137688, 6.35183219, 6.60898203, 6.87142343,  7.13864825, 7.41019785, 7.68471677, 7.9616833, 8.24003101, 8.51980435,  8.79879605, 9.07758609, 9.35444134, 9.6284324, 9.89876634, 10.16522076,  10.42520274, 10.67741683, 10.9213193, 11.15555875, 11.37947453, 11.59236528,  11.79347576, 11.98146647, 12.15517805, 12.3123704, 12.44965733, 12.56969947,  12.66826191, 12.74664651, 12.80407592, 12.83952573, 12.85206571, 12.84123246,  12.80056292, 12.73771357, 12.64885248, 12.533746, 12.39638207, 12.23473002,  12.04962732, 11.84256069, 11.61172842, 11.36182984, 11.09225441, 10.80460659,  10.50114838, 10.18405983, 9.85416699, 9.51342859, 9.16289558, 8.80636352,  8.4436194, 8.07804502, 7.71048112, 7.3429119, 6.97739577, 6.61537283,  6.25726892, 5.90552094, 5.56113119, 5.22499194, 4.89848432, 4.5825624,  4.27717164, 3.98401825, 3.70304535, 3.43463936, 3.17923676, 2.93692796,  2.70795094, 2.49214618, 2.28936087, 2.09934873, 1.92189018, 1.75660002,  1.60294742, 1.46057567, 1.32885458, 1.20737691, 1.0955468, 0.99276788,  0.89851709, 0.81228309, 0.73344034, 0.66153377, 0.59604276, 0.53647026,  0.48235774, 0.43329356, 0.38886854, 0.34866558, 0.31235036, 0.27958048],
#     'dy': [0.00246406, 0.00270263, 0.00413003, 0.00572154, 0.0059946, 0.0079666,  0.00913646, 0.00971798, 0.01074838, 0.01250115, 0.014006, 0.01495796,  0.01624641, 0.01745541, 0.01862373, 0.01949308, 0.02052398, 0.02143018,  0.0224144, 0.02320733, 0.02419585, 0.02492161, 0.02578879, 0.0264857,  0.02722198, 0.02812557, 0.02875525, 0.02963697, 0.03024651, 0.03090972,  0.03147143, 0.03219803, 0.03262175, 0.03320248, 0.03379227, 0.03421585,  0.0347359, 0.03548602, 0.03605452, 0.03655385, 0.03719627, 0.03787654,  0.03889441, 0.03960363, 0.04075606, 0.04214388, 0.04405667, 0.04577418,  0.0482349, 0.05047636, 0.05362199, 0.0572971, 0.06068162, 0.06434801,  0.06875146, 0.07249846, 0.07678514, 0.08097239, 0.08471875, 0.08889026,  0.09266491, 0.09757794, 0.10129108, 0.10556094, 0.11000677, 0.11377304,  0.11688405, 0.11929446, 0.12269454, 0.12532753, 0.12541823, 0.12882392,  0.12848543, 0.12842151, 0.12929262, 0.12784338, 0.1286225, 0.12866583,  0.12894852, 0.12949341, 0.13012769, 0.13093848, 0.13152007, 0.13256655,  0.13342775, 0.13429436, 0.1350262, 0.13570193, 0.13665422, 0.13749667,  0.13886215, 0.13923934, 0.14064368, 0.14181052, 0.14226931, 0.14262262,  0.14250706, 0.14194158, 0.14167298, 0.14016174, 0.1386656, 0.13706695,  0.13431278, 0.13185029, 0.1275977, 0.12371076, 0.11963035, 0.1154726,  0.1110514, 0.10723038, 0.10297737, 0.0991703, 0.09500723, 0.09110149,  0.08748767, 0.08337993, 0.07977181, 0.07599941, 0.07229186, 0.06866837,  0.06513693, 0.06177749, 0.05832189, 0.05512061, 0.05185016, 0.04880332,  0.0458677, 0.04299382, 0.04032518, 0.04844285, 0.03932456, 0.04885114,  0.06680359, 0.06887605, 0.04878468, 0.0729098, 0.05330487, 0.05338136,  0.07547006, 0.04749581, 0.08045209, 0.05825818, 0.05519948, 0.05389998,  0.05327752, 0.04977859, 0.05395545, 0.05362017, 0.05340981, 0.05412502,  0.04777667, 0.05009514, 0.04466632, 0.04998635, 0.04854997, 0.06671962,  0.04731303, 0.04645289, 0.04878293, 0.05978424, 0.04625649, 0.04561353,  0.05279606, 0.04550392, 0.0448346, 0.0436489, 0.0429613, 0.04247841,  0.04244488, 0.04126011, 0.04134056, 0.03986217, 0.03962496, 0.03861266,  0.03813283, 0.0378394, 0.03739746, 0.03739657, 0.03668138, 0.03674841,  0.03679671, 0.03664714, 0.03653628, 0.03656986, 0.03700487, 0.0371314,  0.03767365, 0.03846781, 0.03918093, 0.04054217, 0.04285502, 0.04467147,  0.04669396, 0.04888823, 0.05117757, 0.05357126, 0.0560717, 0.05863092,  0.0612525, 0.06390257, 0.06658597, 0.06926759, 0.07192853, 0.07462682,  0.07731234, 0.08004792, 0.08279201, 0.08541655, 0.08816947, 0.09096032,  0.09376686, 0.09659658, 0.09945045, 0.10232124, 0.10521654, 0.10938306,  0.11230309, 0.11527473, 0.11827578, 0.12128973, 0.12431396, 0.1273586,  0.13105502, 0.13413197, 0.13721516, 0.14030091, 0.14338307, 0.14646075,  0.14951413, 0.15251991, 0.15545619, 0.15828846, 0.16096002, 0.16350645,  0.16585459, 0.16801858, 0.17076993, 0.17254151, 0.17387931, 0.17370735,  0.17453862, 0.17500345, 0.17533136, 0.17522462, 0.17481468, 0.17406324,  0.17297436, 0.17156249, 0.16979842, 0.16879245, 0.16627348, 0.16344207,  0.16032523, 0.15695253, 0.15332626, 0.14946628, 0.14474542, 0.14054646,  0.13616656, 0.1316595, 0.12703245, 0.1223099, 0.11751701, 0.11267192,  0.10777957, 0.10241715, 0.09757542, 0.09275655, 0.0879802, 0.08326713,  0.07861771, 0.074084, 0.06962308, 0.06527186, 0.06104853, 0.05696227,  0.0529084, 0.04915031, 0.04555736, 0.04213401, 0.03888607, 0.03581526,  0.03292132, 0.0302052, 0.02766351, 0.02529471, 0.02309242, 0.02104856,  0.01915746, 0.0174119, 0.01580275, 0.01432374, 0.01296677, 0.01172416,  0.01058782, 0.00955287, 0.00860804, 0.00774771, 0.00696626, 0.00625706]
#   },
#   'Model3body': {
#     'x': [1.00000000e-01, 1.03128316e-01, 1.06354496e-01, 1.09681601e-01,
#           1.13112788e-01, 1.16651313e-01, 1.20300535e-01, 1.24063916e-01,  1.27945027e-01, 1.31947552e-01, 1.36075289e-01, 1.40332154e-01,  1.44722187e-01, 1.49249555e-01, 1.53918552e-01, 1.58733611e-01,  1.63699300e-01, 1.68820332e-01, 1.74101565e-01, 1.79548012e-01,  1.85164842e-01, 1.90957383e-01, 1.96931134e-01, 2.03091762e-01,  2.09445114e-01, 2.15997219e-01, 2.22754295e-01, 2.29722754e-01,  2.36909207e-01, 2.44320476e-01, 2.51963593e-01, 2.59845810e-01,  2.67974609e-01, 2.76357701e-01, 2.85003044e-01, 2.93918840e-01,  3.03113550e-01, 3.12595900e-01, 3.22374888e-01, 3.32459793e-01,  3.42860186e-01, 3.53585937e-01, 3.64647222e-01, 3.76054540e-01,  3.87818715e-01, 3.99950910e-01, 4.12462638e-01, 4.25365773e-01,  4.38672559e-01, 4.52395623e-01, 4.66547988e-01, 4.81143084e-01,  4.96194760e-01, 5.11717301e-01, 5.27725435e-01, 5.44234355e-01,  5.61259726e-01, 5.78817704e-01, 5.96924951e-01, 6.15598650e-01,  6.34856522e-01, 6.54716840e-01, 6.75198452e-01, 6.96320794e-01,  7.18103909e-01, 7.40568469e-01, 7.63735792e-01, 7.87627861e-01,  8.12267350e-01, 8.37677640e-01, 8.63882844e-01, 8.90907830e-01,  9.18778243e-01, 9.47520530e-01, 9.77161967e-01, 1.00773068e+00,  1.03925568e+00, 1.07176689e+00, 1.10529514e+00, 1.13987227e+00,  1.17553107e+00, 1.21230540e+00, 1.25023015e+00, 1.28934130e+00,  1.32967597e+00, 1.37127243e+00, 1.41417017e+00, 1.45840988e+00,  1.50403355e+00, 1.55108448e+00, 1.59960730e+00, 1.64964807e+00,  1.70125428e+00, 1.75447489e+00, 1.80936041e+00, 1.86596292e+00,  1.92433614e+00, 1.98453546e+00, 2.04661800e+00, 2.11064268e+00,  2.17667025e+00, 2.24476338e+00, 2.31498667e+00, 2.38740677e+00,  2.46209240e+00, 2.53911443e+00, 2.61854596e+00, 2.70046235e+00,  2.78494135e+00, 2.87206312e+00, 2.96191033e+00, 3.05456825e+00,  3.15012480e+00, 3.24867066e+00, 3.35029934e+00, 3.45510729e+00,  3.56319397e+00, 3.67466194e+00, 3.78961698e+00, 3.90816818e+00,  4.03042803e+00, 4.15651256e+00, 4.28654141e+00, 4.42063797e+00,  4.55892950e+00, 4.70154722e+00, 4.84862648e+00, 5.00030684e+00,  5.15673224e+00, 5.31805113e+00, 5.48441658e+00, 5.65598646e+00,  5.83292359e+00, 6.01539588e+00, 6.20357648e+00, 6.39764396e+00,  6.59778248e+00, 6.80418197e+00, 7.01703829e+00, 7.23655342e+00,  7.46293569e+00, 7.69639990e+00, 7.93716762e+00, 8.18546731e+00,  8.44153460e+00, 8.70561248e+00, 8.97795155e+00, 9.25881025e+00,  9.54845510e+00, 9.84716096e+00, 1.01552113e+01, 1.04728984e+01,  1.08005237e+01, 1.11383983e+01, 1.14868426e+01, 1.18461873e+01,  1.22167735e+01, 1.25989528e+01, 1.29930878e+01, 1.33995527e+01,  1.38187331e+01, 1.42510267e+01, 1.46968439e+01, 1.51566076e+01,  1.56307542e+01, 1.61197336e+01, 1.66240098e+01, 1.71440614e+01,  1.76803818e+01, 1.82334800e+01, 1.88038809e+01, 1.93921257e+01,  1.99987727e+01, 2.06243975e+01, 2.12695939e+01, 2.19349740e+01,  2.26211693e+01, 2.33288310e+01, 2.40586305e+01, 2.48112606e+01,  2.55874352e+01, 2.63878910e+01, 2.72133877e+01, 2.80647085e+01,  2.89426612e+01, 2.98480792e+01, 3.07818214e+01, 3.17447741e+01,  3.27378510e+01, 3.37619944e+01, 3.48181763e+01, 3.59073989e+01,  3.70306958e+01, 3.81891330e+01, 3.93838098e+01, 4.06158599e+01,  4.18864524e+01, 4.31967930e+01, 4.45481252e+01, 4.59417313e+01,  4.73789339e+01, 4.88610967e+01, 5.03896263e+01, 5.19659730e+01,  5.35916329e+01, 5.52681486e+01, 5.69971109e+01, 5.87801607e+01,  6.06189899e+01, 6.25153435e+01, 6.44710211e+01, 6.64878784e+01,  6.85678294e+01, 7.07128478e+01, 7.29249692e+01, 7.52062927e+01,  7.75589832e+01, 7.99852734e+01, 8.24874655e+01, 8.50679342e+01,  8.77291280e+01, 9.04735724e+01, 9.33038717e+01, 9.62227117e+01,  9.92328623e+01, 1.02337180e+02, 1.05538610e+02, 1.08840192e+02,  1.12245057e+02, 1.15756437e+02, 1.19377664e+02, 1.23112175e+02,  1.26963513e+02, 1.30935333e+02, 1.35031404e+02, 1.39255613e+02,  1.43611969e+02, 1.48104605e+02, 1.52737785e+02, 1.57515906e+02,  1.62443501e+02, 1.67525247e+02, 1.72765966e+02, 1.78170632e+02,  1.83744372e+02, 1.89492477e+02, 1.95420401e+02, 2.01533769e+02,  2.07838382e+02, 2.14340223e+02, 2.21045463e+02, 2.27960464e+02,  2.35091788e+02, 2.42446202e+02, 2.50030685e+02, 2.57852435e+02,  2.65918874e+02, 2.74237657e+02, 2.82816678e+02, 2.91664078e+02,  3.00788252e+02, 3.10197859e+02, 3.19901828e+02, 3.29909369e+02,  3.40229977e+02, 3.50873446e+02, 3.61849876e+02, 3.73169684e+02,  3.84843611e+02, 3.96882735e+02, 4.09298482e+02, 4.22102632e+02,  4.35307336e+02, 4.48925126e+02, 4.62968923e+02, 4.77452054e+02,  4.92388263e+02, 5.07791724e+02, 5.23677054e+02, 5.40059328e+02,  5.56954091e+02, 5.74377375e+02, 5.92345715e+02, 6.10876161e+02,  6.29986298e+02, 6.49694260e+02, 6.70018750e+02, 6.90979055e+02,  7.12595063e+02, 7.34887289e+02, 7.57876886e+02, 7.81585671e+02,  8.06036141e+02, 8.31251499e+02, 8.57255673e+02, 8.84073340e+02,  9.11729948e+02, 9.40251743e+02, 9.69665789e+02, 1.00000000e+03],
#     'y': [0.07116281, 0.07181733, 0.12898948, 0.18801264, 0.20073073,
#           0.27032487,  0.33350701, 0.36496816, 0.41913461, 0.51230031, 0.59464946, 0.65555845,  0.75084413, 0.81914254, 0.92223871, 0.99136415, 1.0618379,  1.1705502,  1.25220175, 1.3457719,  1.44776903, 1.53629959, 1.63235943, 1.72286346,  1.81127601, 1.92325695, 2.0001148,  2.10059747, 2.18535728, 2.26786071,  2.34087305, 2.42181186, 2.50298012, 2.5818459,  2.66128478, 2.734545,  2.81556715, 2.90092692, 2.98061144, 3.07098386, 3.15143714, 3.23215821,  3.326097, 3.39688496, 3.51954043, 3.66065623, 3.84355286, 4.00614612,  4.20548028, 4.40258593, 4.63150646, 4.90715825, 5.14108979, 5.38323068,  5.66802352, 5.87896301, 6.20197343, 6.48121247, 6.74763898, 7.0646081,  7.36431198, 7.65458828, 7.9282111, 8.24326443, 8.47413512, 8.7567631,  9.02729262, 9.29791592, 9.53726462, 9.78166492, 10.05740351, 10.25777431,  10.50439741, 10.72861794, 10.91691086, 11.10539696, 11.26200925, 11.37289916,  11.49936311, 11.57747584, 11.63048375, 11.6797618, 11.6687384, 11.68369102,  11.65912668, 11.62242354, 11.56805387, 11.50940948, 11.4581607, 11.39328661,  11.34125443, 11.24515586, 11.1635676, 11.04074821, 10.89270784, 10.72217983,  10.52024332, 10.29339228, 10.07440468, 9.81358403, 9.53357067, 9.25877366,  8.95442115, 8.6499901,  8.30222883, 7.95728746, 7.64625846, 7.31375587,  6.99050523, 6.68515216, 6.37202169, 6.0773494,  5.7863768,  5.49355333,  5.22604657, 4.9521596,  4.69995632, 4.45383312, 4.21651005, 3.99415282,  3.78287719, 3.588578,  3.40017594, 3.22717267, 3.05824263, 2.90624831,  2.77035073, 2.64427847, 2.53286067, 2.43582534, 2.35021424, 2.27865715,  2.21719048, 2.16840559, 2.12868406, 2.0992771,  2.08017642, 2.06951514,  2.06797199, 2.07312806, 2.08574664, 2.10371587, 2.12837148, 2.15739907,  2.19096685, 2.22771777, 2.26752653, 2.30912661, 2.35232929, 2.39621499,  2.44008789, 2.48351251, 2.52573064, 2.56634315, 2.60450081, 2.64017145,  2.67261103, 2.70149496, 2.72669215, 2.74767316, 2.76420378, 2.77630089,  2.78354818, 2.78600823, 2.78384041, 2.77691134, 2.76547916, 2.74987486,  2.7302759,  2.70716339, 2.68106895, 2.65255978, 2.62233409, 2.59115489,  2.56003418, 2.52985481, 2.50165725, 2.47682827, 2.45638823, 2.44166988,  2.43426124, 2.4352674,  2.44622182, 2.46868519, 2.50377422, 2.55302869,  2.61774165, 2.69893651, 2.7979204,  2.91548918, 3.05246159, 3.20967439,  3.38725966, 3.58573197, 3.80526148, 4.04525039, 4.30583776, 4.58642702,  4.8856909,  5.20337648, 5.53816997, 5.88799496, 6.25230229, 6.62919795,  7.01586203, 7.411621,  7.81414948, 8.21998274, 8.62850952, 9.03711793,  9.44188818, 9.84242029, 10.23598545, 10.61846018, 10.9897526, 11.34720259,  11.68676149, 12.00871619, 12.31066903, 12.58888931, 12.8441193, 13.07438697,  13.27649718, 13.45169536, 13.59855764, 13.71459167, 13.80155693, 13.85864932,  13.88417171, 13.88036238, 13.84704895, 13.78333649, 13.691866,  13.57303661,  13.42669685, 13.25575638, 13.06108023, 12.84314175, 12.60495123, 12.34770451,  12.07232664, 11.78176898, 11.47740281, 11.16042134, 10.83356995, 10.4982477,  10.15573439, 9.80846171, 9.45773081, 9.10476067, 8.7515988,  8.39935541,  8.0490836,  7.70241665, 7.36021907, 7.02332122, 6.69294022, 6.36968248,  6.05413203, 5.7471228,  5.44901202, 5.16014872, 4.88103327, 4.61180571,  4.35260667, 4.10366552, 3.86494629, 3.6364206,  3.4181092,  3.20984521,  3.01147286, 2.82286409, 2.64376313, 2.47392708, 2.31313089, 2.16106569,  2.01743747, 1.88196595, 1.75431839, 1.63417872, 1.52124302, 1.41517724,  1.31566491, 1.22240135, 1.135067,  1.05336001, 0.97699039, 0.90566322,  0.83910067, 0.77703679, 0.71920697, 0.66536246, 0.61526617, 0.56868587,  0.5254039,  0.48521352, 0.44791459, 0.41332018, 0.38125318, 0.35154388],
#     'dy': [0.00245663, 0.00302269, 0.00414706, 0.00562527, 0.00648497, 0.00801188,  0.00932548, 0.01035696, 0.01094347, 0.01280145, 0.01438696, 0.01537701,  0.01689914, 0.01795805, 0.01926041, 0.020468,  0.02136828, 0.02226938,  0.02340949, 0.02403637, 0.02513656, 0.02591615, 0.02678357, 0.02753039,  0.02832397, 0.02926547, 0.02997634, 0.03080147, 0.03141775, 0.03212703,  0.03249663, 0.0331939, 0.03368224, 0.03431299, 0.03507389, 0.03561613,  0.03628068, 0.03697467, 0.03764143, 0.03834917, 0.03912131, 0.03991804,  0.04091357, 0.04196073, 0.04359507, 0.04579251, 0.04856223, 0.05097622,  0.0543996, 0.05707327, 0.06077568, 0.06501644, 0.06860128, 0.07212738,  0.0768486, 0.08075142, 0.08461896, 0.08857782, 0.09214071, 0.09595705,  0.09928179, 0.1039542, 0.10708795, 0.11089019, 0.11514077, 0.11852104,  0.12104802, 0.12320074, 0.1260599, 0.12804049, 0.12793858, 0.13014834,  0.12975772, 0.1295086, 0.13018794, 0.12919945, 0.12996915, 0.130119,  0.13068886, 0.13138758, 0.13217882, 0.13319826, 0.13393464, 0.13513914,  0.13608972, 0.13703534, 0.13780859, 0.13853427, 0.13955085, 0.14042326,  0.14181526, 0.14232596, 0.14399475, 0.14530359, 0.14564089, 0.14596378,  0.14568293, 0.14488957, 0.14458769, 0.14274513, 0.14094827, 0.139094,  0.13614797, 0.1335418, 0.12914355, 0.12506988, 0.12097172, 0.11668185,  0.11214598, 0.10826231, 0.10387553, 0.09997611, 0.09582335, 0.09172408,  0.08798178, 0.08370503, 0.07990843, 0.07594343, 0.07199699, 0.06817061,  0.06446596, 0.06105347, 0.05761658, 0.05455409, 0.05151823, 0.04892073,  0.04669713, 0.04486623, 0.04358415, 0.04290518, 0.04284342, 0.04345896,  0.0447242, 0.04660972, 0.04905417, 0.05200275, 0.05540676, 0.05918354,  0.0632807, 0.06763863, 0.07220859, 0.07693873, 0.08179202, 0.08672834,  0.09170753, 0.09669433, 0.10166192, 0.10656899, 0.11138856, 0.11610062,  0.12066456, 0.12504889, 0.12925143, 0.13322802, 0.13695839, 0.14043529,  0.14362381, 0.14650757, 0.1490859, 0.15132659, 0.15322156, 0.15477441,  0.15595792, 0.15677036, 0.15721976, 0.15728266, 0.15696362, 0.15627374,  0.1551934, 0.15373372, 0.15190942, 0.14970913, 0.14715574, 0.14427337,  0.14107659, 0.13761479, 0.13394346, 0.13014011, 0.12631433, 0.1226044,  0.11920088, 0.11632239, 0.11423459, 0.11323398, 0.11359807, 0.11558159,  0.11934902, 0.12496775, 0.13240367, 0.14150589, 0.15210272, 0.16396227,  0.17682108, 0.19047736, 0.20468189, 0.21919597, 0.23385895, 0.24845431,  0.26277399, 0.2767059, 0.29007929, 0.30270126, 0.31451395, 0.32539261,  0.33516631, 0.34383101, 0.35130542, 0.35745927, 0.362334,  0.36589589,  0.36806985, 0.36893789, 0.3685169, 0.36679393, 0.36389029, 0.35987785,  0.35480808, 0.34884231, 0.34210703, 0.33471531, 0.326867,  0.31873329,  0.31047556, 0.30231628, 0.29444816, 0.28703977, 0.28029605, 0.2743714,  0.26936928, 0.26540674, 0.26252503, 0.26069054, 0.25988794, 0.26001568,  0.26091188, 0.26246596, 0.26449909, 0.26680431, 0.26926014, 0.27169496,  0.27393304, 0.27589622, 0.27746336, 0.27851971, 0.27904155, 0.27896416,  0.27823178, 0.27686537, 0.27484614, 0.27216387, 0.26886893, 0.26497424,  0.26049979, 0.25551129, 0.25004,  0.24412281, 0.23782929, 0.23119857,  0.22427399, 0.21712084, 0.20977856, 0.2022895, 0.19469858, 0.1870632,  0.17940884, 0.17178027, 0.1642052, 0.15671248, 0.1493351, 0.14209285,  0.13500613, 0.12809706, 0.12137792, 0.11486118, 0.10855979, 0.10247951,  0.09662613, 0.09100536, 0.08561778, 0.080464,  0.07554434, 0.07085559,  0.06640029, 0.06216347, 0.05814536, 0.05434045, 0.05073848, 0.04734144,  0.04413783, 0.04112074, 0.03828235, 0.0356152, 0.0331119, 0.03076463,  0.02856593, 0.02650852, 0.02458495, 0.02278816, 0.02111131, 0.01954758]
# },
#   'ModelBosa': {
#     'x': [1.00000000e-01, 1.03128316e-01, 1.06354496e-01, 1.09681601e-01,
#           1.13112788e-01, 1.16651313e-01, 1.20300535e-01, 1.24063916e-01,  1.27945027e-01, 1.31947552e-01, 1.36075289e-01, 1.40332154e-01,  1.44722187e-01, 1.49249555e-01, 1.53918552e-01, 1.58733611e-01,  1.63699300e-01, 1.68820332e-01, 1.74101565e-01, 1.79548012e-01,  1.85164842e-01, 1.90957383e-01, 1.96931134e-01, 2.03091762e-01,  2.09445114e-01, 2.15997219e-01, 2.22754295e-01, 2.29722754e-01,  2.36909207e-01, 2.44320476e-01, 2.51963593e-01, 2.59845810e-01,  2.67974609e-01, 2.76357701e-01, 2.85003044e-01, 2.93918840e-01,  3.03113550e-01, 3.12595900e-01, 3.22374888e-01, 3.32459793e-01,  3.42860186e-01, 3.53585937e-01, 3.64647222e-01, 3.76054540e-01,  3.87818715e-01, 3.99950910e-01, 4.12462638e-01, 4.25365773e-01,  4.38672559e-01, 4.52395623e-01, 4.66547988e-01, 4.81143084e-01,  4.96194760e-01, 5.11717301e-01, 5.27725435e-01, 5.44234355e-01,  5.61259726e-01, 5.78817704e-01, 5.96924951e-01, 6.15598650e-01,  6.34856522e-01, 6.54716840e-01, 6.75198452e-01, 6.96320794e-01,  7.18103909e-01, 7.40568469e-01, 7.63735792e-01, 7.87627861e-01,  8.12267350e-01, 8.37677640e-01, 8.63882844e-01, 8.90907830e-01,  9.18778243e-01, 9.47520530e-01, 9.77161967e-01, 1.00773068e+00,  1.03925568e+00, 1.07176689e+00, 1.10529514e+00, 1.13987227e+00,  1.17553107e+00, 1.21230540e+00, 1.25023015e+00, 1.28934130e+00,  1.32967597e+00, 1.37127243e+00, 1.41417017e+00, 1.45840988e+00,  1.50403355e+00, 1.55108448e+00, 1.59960730e+00, 1.64964807e+00,  1.70125428e+00, 1.75447489e+00, 1.80936041e+00, 1.86596292e+00,  1.92433614e+00, 1.98453546e+00, 2.04661800e+00, 2.11064268e+00,  2.17667025e+00, 2.24476338e+00, 2.31498667e+00, 2.38740677e+00,  2.46209240e+00, 2.53911443e+00, 2.61854596e+00, 2.70046235e+00,  2.78494135e+00, 2.87206312e+00, 2.96191033e+00, 3.05456825e+00,  3.15012480e+00, 3.24867066e+00, 3.35029934e+00, 3.45510729e+00,  3.56319397e+00, 3.67466194e+00, 3.78961698e+00, 3.90816818e+00,  4.03042803e+00, 4.15651256e+00, 4.28654141e+00, 4.42063797e+00,  4.55892950e+00, 4.70154722e+00, 4.84862648e+00, 5.00030684e+00,  5.15673224e+00, 5.31805113e+00, 5.48441658e+00, 5.65598646e+00,  5.83292359e+00, 6.01539588e+00, 6.20357648e+00, 6.39764396e+00,  6.59778248e+00, 6.80418197e+00, 7.01703829e+00, 7.23655342e+00,  7.46293569e+00, 7.69639990e+00, 7.93716762e+00, 8.18546731e+00,  8.44153460e+00, 8.70561248e+00, 8.97795155e+00, 9.25881025e+00,  9.54845510e+00, 9.84716096e+00, 1.01552113e+01, 1.04728984e+01,  1.08005237e+01, 1.11383983e+01, 1.14868426e+01, 1.18461873e+01,  1.22167735e+01, 1.25989528e+01, 1.29930878e+01, 1.33995527e+01,  1.38187331e+01, 1.42510267e+01, 1.46968439e+01, 1.51566076e+01,  1.56307542e+01, 1.61197336e+01, 1.66240098e+01, 1.71440614e+01,  1.76803818e+01, 1.82334800e+01, 1.88038809e+01, 1.93921257e+01,  1.99987727e+01, 2.06243975e+01, 2.12695939e+01, 2.19349740e+01,  2.26211693e+01, 2.33288310e+01, 2.40586305e+01, 2.48112606e+01,  2.55874352e+01, 2.63878910e+01, 2.72133877e+01, 2.80647085e+01,  2.89426612e+01, 2.98480792e+01, 3.07818214e+01, 3.17447741e+01,  3.27378510e+01, 3.37619944e+01, 3.48181763e+01, 3.59073989e+01,  3.70306958e+01, 3.81891330e+01, 3.93838098e+01, 4.06158599e+01,  4.18864524e+01, 4.31967930e+01, 4.45481252e+01, 4.59417313e+01,  4.73789339e+01, 4.88610967e+01, 5.03896263e+01, 5.19659730e+01,  5.35916329e+01, 5.52681486e+01, 5.69971109e+01, 5.87801607e+01,  6.06189899e+01, 6.25153435e+01, 6.44710211e+01, 6.64878784e+01,  6.85678294e+01, 7.07128478e+01, 7.29249692e+01, 7.52062927e+01,  7.75589832e+01, 7.99852734e+01, 8.24874655e+01, 8.50679342e+01,  8.77291280e+01, 9.04735724e+01, 9.33038717e+01, 9.62227117e+01,  9.92328623e+01, 1.02337180e+02, 1.05538610e+02, 1.08840192e+02,  1.12245057e+02, 1.15756437e+02, 1.19377664e+02, 1.23112175e+02,  1.26963513e+02, 1.30935333e+02, 1.35031404e+02, 1.39255613e+02,  1.43611969e+02, 1.48104605e+02, 1.52737785e+02, 1.57515906e+02,  1.62443501e+02, 1.67525247e+02, 1.72765966e+02, 1.78170632e+02,  1.83744372e+02, 1.89492477e+02, 1.95420401e+02, 2.01533769e+02,  2.07838382e+02, 2.14340223e+02, 2.21045463e+02, 2.27960464e+02,  2.35091788e+02, 2.42446202e+02, 2.50030685e+02, 2.57852435e+02,  2.65918874e+02, 2.74237657e+02, 2.82816678e+02, 2.91664078e+02,  3.00788252e+02, 3.10197859e+02, 3.19901828e+02, 3.29909369e+02,  3.40229977e+02, 3.50873446e+02, 3.61849876e+02, 3.73169684e+02,  3.84843611e+02, 3.96882735e+02, 4.09298482e+02, 4.22102632e+02,  4.35307336e+02, 4.48925126e+02, 4.62968923e+02, 4.77452054e+02,  4.92388263e+02, 5.07791724e+02, 5.23677054e+02, 5.40059328e+02,  5.56954091e+02, 5.74377375e+02, 5.92345715e+02, 6.10876161e+02,  6.29986298e+02, 6.49694260e+02, 6.70018750e+02, 6.90979055e+02,  7.12595063e+02, 7.34887289e+02, 7.57876886e+02, 7.81585671e+02,  8.06036141e+02, 8.31251499e+02, 8.57255673e+02, 8.84073340e+02,  9.11729948e+02, 9.40251743e+02, 9.69665789e+02, 1.00000000e+03],
#     'y': [0.06300638, 0.06024614, 0.11792336, 0.17536302, 0.18479468,
#           0.25162201,  0.31529283, 0.34327826, 0.39971883, 0.49540809, 0.57533586, 0.63829436,  0.73595776, 0.8030857, 0.91144181, 0.9768608, 1.04463837, 1.16347698,  1.24349086, 1.34327503, 1.44828952, 1.54074094, 1.63961845, 1.73350254,  1.82726305, 1.94135538, 2.02051664, 2.1251308, 2.20980777, 2.29322731,  2.36951898, 2.45001576, 2.53253824, 2.61178739, 2.69003816, 2.7634442,  2.84358063, 2.92861048, 3.00667968, 3.09624888, 3.17413205, 3.25242441,  3.34339216, 3.41013088, 3.52889882, 3.66461078, 3.84212248, 4.0004176,  4.19325175, 4.38671725, 4.60930224, 4.87944141, 5.10750113, 5.34453755,  5.62128174, 5.82422609, 6.14578284, 6.42089835, 6.68467421, 7.00051361,  7.30082339, 7.58441874, 7.85799715, 8.1734837, 8.39577575, 8.68001207,  8.95434539, 9.23405585, 9.47383219, 9.72431265, 10.02391166, 10.22340815,  10.49673765, 10.74377612, 10.9466218, 11.17149786, 11.34509373, 11.47913546,  11.63274662, 11.73004472, 11.79830411, 11.86950443, 11.87575365, 11.9077979,  11.89851774, 11.87487262, 11.83214092, 11.78376481, 11.74390435, 11.68970778,  11.65188143, 11.56275997, 11.49655588, 11.38776376, 11.24650437, 11.08344863,  10.88549233, 10.65991929, 10.44420055, 10.17981225, 9.89629618, 9.61798447,  9.30446207, 8.99216541, 8.6282207, 8.26847221, 7.94376504, 7.59667241,  7.2586423, 6.94145732, 6.61429156, 6.30744674, 6.00301342, 5.70494225,  5.42819581, 5.16556805, 4.88424212, 4.61742422, 4.37442296, 4.11817263,  3.88653407, 3.67195426, 3.43649727, 3.2496348, 3.0383738, 2.85199438,  2.67780417, 2.50101803, 2.34882065, 2.19672718, 2.06930905, 1.94164862,  1.83317148, 1.74149642, 1.67158585, 1.60705062, 1.58842688, 1.49682332,  1.45077551, 1.4398288, 1.42714032, 1.47842276, 1.51814988, 1.56610248,  1.60329267, 1.61004734, 1.69114027, 1.59801983, 1.63504142, 1.63615036,  1.61820347, 1.62097608, 1.65351983, 1.66522813, 1.73844073, 1.83155375,  1.78893108, 1.855305, 1.95971007, 1.965064, 1.93738198, 2.02314398,  2.00847463, 2.02350135, 1.98854927, 2.01802692, 2.05162622, 2.10695341,  2.12669922, 2.11153179, 2.1458686, 2.12685353, 2.14106417, 2.14786108,  2.13935536, 2.16983875, 2.14456084, 2.18906885, 2.17106135, 2.19306523,  2.22159417, 2.22787155, 2.25711407, 2.27683449, 2.32917317, 2.36015924,  2.39483919, 2.45427229, 2.50288243, 2.57454844, 2.64229786, 2.71279133,  2.80562982, 2.89786725, 3.00023359, 3.11820828, 3.2470644, 3.38189309,  3.53396819, 3.69971356, 3.87845519, 4.07255968, 4.28170549, 4.50686737,  4.74776263, 5.00433744, 5.27743923, 5.56620849, 5.87058719, 6.19012324,  6.52330466, 6.86964558, 7.22864383, 7.59727483, 7.97565221, 8.36169194,  8.75247404, 9.14738619, 9.54404407, 9.93904295, 10.33170296, 10.71932692,  11.09838077, 11.46820851, 11.82600117, 12.16805481, 12.49392558, 12.80116892,  13.08623804, 13.3496498, 13.58890429, 13.80178529, 13.98895141, 14.14925042,  14.28067028, 14.38352768, 14.45492798, 14.48891264, 14.48796612, 14.45250559,  14.37955789, 14.27398303, 14.13547915, 13.96451033, 13.76378566, 13.53432877,  13.27677221, 12.99504601, 12.69013018, 12.36309855, 12.01791674, 11.65587487,  11.27907939, 10.89031067, 10.49214628, 10.08537451, 9.67352492, 9.2571892,  8.83950334, 8.42246257, 8.00826228, 7.5965859, 7.1911572, 6.79271642,  6.40208825, 6.02126788, 5.65070245, 5.29136116, 4.94439847, 4.6102242,  4.28936625, 3.98254265, 3.68986732, 3.41158639, 3.14800337, 2.89891461,  2.66436614, 2.44418293, 2.23793631, 2.04536352, 1.8660716, 1.69954671,  1.54526201, 1.40271631, 1.27126183, 1.15033058, 1.03932677, 0.93764753,  0.84469223, 0.759892, 0.68267118, 0.61248459, 0.5488154, 0.49115641,  0.43903816, 0.39200507, 0.34963122, 0.31151587, 0.27728493, 0.24658749],
#     'dy': [0.00718235, 0.00767221, 0.01041338, 0.01378561, 0.01397281, 0.01851651,  0.01985689, 0.0210587, 0.0224318, 0.02375285, 0.0268019, 0.02680454,  0.02760929, 0.02987479, 0.02953225, 0.03048214, 0.03202934, 0.03095365,  0.03100663, 0.03075133, 0.03008201, 0.02975961, 0.02927458, 0.02897469,  0.02817494, 0.02846293, 0.02831003, 0.02842937, 0.02877597, 0.02900622,  0.02919265, 0.02966707, 0.03006102, 0.03058554, 0.03115356, 0.0315959,  0.03218361, 0.0328599, 0.03348757, 0.03422947, 0.03494273, 0.03569268,  0.03656046, 0.03731203, 0.03846049, 0.03980444, 0.0415943, 0.0432543,  0.0453825, 0.04732789, 0.04991482, 0.05297438, 0.05571773, 0.05859876,  0.06210066, 0.06502059, 0.06876549, 0.07223985, 0.07555079, 0.07927084,  0.08269685, 0.0866421, 0.08987231, 0.09368142, 0.0970976, 0.10055683,  0.10366155, 0.10647783, 0.10956026, 0.11247237, 0.11441949, 0.11724087,  0.11936405, 0.12093689, 0.12287477, 0.1239997, 0.12572266, 0.12649753,  0.12755491, 0.12831413, 0.12871572, 0.12934571, 0.12940927, 0.12980725,  0.12994275, 0.13000597, 0.12985969, 0.12976837, 0.12972603, 0.12970493,  0.12999299, 0.12945436, 0.12954018, 0.1293644, 0.12907178, 0.12844811,  0.12737083, 0.12614452, 0.12491142, 0.12249374, 0.12058534, 0.11858136,  0.11597537, 0.11334732, 0.10956698, 0.10595759, 0.10239631, 0.0987181,  0.09495735, 0.09159129, 0.0879834, 0.08471012, 0.08122386, 0.07798587,  0.07488434, 0.07193569, 0.06856919, 0.06534404, 0.06248458, 0.05924563,  0.056367, 0.05368142, 0.05055844, 0.04825174, 0.04547761, 0.04305779,  0.04068521, 0.03818461, 0.03605233, 0.03380821, 0.03195377, 0.03000887,  0.02837166, 0.02690219, 0.02591126, 0.02486169, 0.02495861, 0.02323194,  0.02247551, 0.02262293, 0.0228737, 0.02482371, 0.02673285, 0.02820063,  0.02883348, 0.02894295, 0.03072997, 0.02742692, 0.0277537, 0.02737705,  0.02684944, 0.02710417, 0.02806947, 0.02851597, 0.0300103, 0.03176005,  0.03048931, 0.03123276, 0.03259097, 0.03239637, 0.03129227, 0.03223406,  0.03160703, 0.03174109, 0.03093485, 0.0312577, 0.03141936, 0.03179026,  0.0319993, 0.03119824, 0.03127682, 0.03045344, 0.0302274, 0.02976429,  0.02926552, 0.02951795, 0.02878098, 0.02910097, 0.02838686, 0.02851667,  0.02891212, 0.02888009, 0.02938385, 0.02996674, 0.03110152, 0.03214031,  0.03338895, 0.03508225, 0.03685768, 0.03896012, 0.04120919, 0.04369542,  0.04653255, 0.04959887, 0.05295299, 0.05663841, 0.06061375, 0.06486631,  0.06941706, 0.07422969, 0.07928931, 0.08452216, 0.08992221, 0.09546674,  0.10105741, 0.10670226, 0.11234632, 0.11790625, 0.12337029, 0.12870981,  0.13383565, 0.1387578, 0.1434498, 0.14784702, 0.15198864, 0.15586691,  0.15945565, 0.16280135, 0.16590933, 0.1687643, 0.17141578, 0.17387047,  0.17610808, 0.17817202, 0.18005242, 0.18172111, 0.18320275, 0.18446553,  0.18549906, 0.18629153, 0.18681937, 0.18704843, 0.18699207, 0.18663466,  0.18594452, 0.1849569, 0.18365686, 0.18202755, 0.18010847, 0.17790256,  0.17541028, 0.17268087, 0.16973233, 0.16657495, 0.16325784, 0.15979672,  0.15620841, 0.15253194, 0.14877741, 0.144947, 0.14108074, 0.13717223,  0.13323013, 0.12926753, 0.12529025, 0.12129286, 0.11728839, 0.11326645,  0.10923907, 0.10521122, 0.10118955, 0.0971563, 0.09314079, 0.08913916,  0.08515206, 0.08119627, 0.07727107, 0.07338629, 0.06955366, 0.06577622,  0.06206142, 0.05841923, 0.05485644, 0.05168983, 0.04977561, 0.04461088,  0.04154381, 0.03856358, 0.03571751, 0.03301089, 0.03044652, 0.02802532,  0.02574627, 0.02360827, 0.02160771, 0.01973942, 0.01799941, 0.01641006,  0.01490596, 0.01351851, 0.01224115, 0.01106956, 0.00999667, 0.00901452,  0.00811751, 0.00729992, 0.0065563, 0.00588132, 0.0052698, 0.0047168, ]
# }}


## 14% systematics
data = {
  'ModelChary': {
    'x': [1.00000000e-01, 1.03128316e-01, 1.06354496e-01, 1.09681601e-01,
          1.13112788e-01, 1.16651313e-01, 1.20300535e-01, 1.24063916e-01,  1.27945027e-01, 1.31947552e-01, 1.36075289e-01, 1.40332154e-01,  1.44722187e-01, 1.49249555e-01, 1.53918552e-01, 1.58733611e-01,  1.63699300e-01, 1.68820332e-01, 1.74101565e-01, 1.79548012e-01,  1.85164842e-01, 1.90957383e-01, 1.96931134e-01, 2.03091762e-01,  2.09445114e-01, 2.15997219e-01, 2.22754295e-01, 2.29722754e-01,  2.36909207e-01, 2.44320476e-01, 2.51963593e-01, 2.59845810e-01,  2.67974609e-01, 2.76357701e-01, 2.85003044e-01, 2.93918840e-01,  3.03113550e-01, 3.12595900e-01, 3.22374888e-01, 3.32459793e-01,  3.42860186e-01, 3.53585937e-01, 3.64647222e-01, 3.76054540e-01,  3.87818715e-01, 3.99950910e-01, 4.12462638e-01, 4.25365773e-01,  4.38672559e-01, 4.52395623e-01, 4.66547988e-01, 4.81143084e-01,  4.96194760e-01, 5.11717301e-01, 5.27725435e-01, 5.44234355e-01,  5.61259726e-01, 5.78817704e-01, 5.96924951e-01, 6.15598650e-01,  6.34856522e-01, 6.54716840e-01, 6.75198452e-01, 6.96320794e-01,  7.18103909e-01, 7.40568469e-01, 7.63735792e-01, 7.87627861e-01,  8.12267350e-01, 8.37677640e-01, 8.63882844e-01, 8.90907830e-01,  9.18778243e-01, 9.47520530e-01, 9.77161967e-01, 1.00773068e+00,  1.03925568e+00, 1.07176689e+00, 1.10529514e+00, 1.13987227e+00,  1.17553107e+00, 1.21230540e+00, 1.25023015e+00, 1.28934130e+00,  1.32967597e+00, 1.37127243e+00, 1.41417017e+00, 1.45840988e+00,  1.50403355e+00, 1.55108448e+00, 1.59960730e+00, 1.64964807e+00,  1.70125428e+00, 1.75447489e+00, 1.80936041e+00, 1.86596292e+00,  1.92433614e+00, 1.98453546e+00, 2.04661800e+00, 2.11064268e+00,  2.17667025e+00, 2.24476338e+00, 2.31498667e+00, 2.38740677e+00,  2.46209240e+00, 2.53911443e+00, 2.61854596e+00, 2.70046235e+00,  2.78494135e+00, 2.87206312e+00, 2.96191033e+00, 3.05456825e+00,  3.15012480e+00, 3.24867066e+00, 3.35029934e+00, 3.45510729e+00,  3.56319397e+00, 3.67466194e+00, 3.78961698e+00, 3.90816818e+00,  4.03042803e+00, 4.15651256e+00, 4.28654141e+00, 4.42063797e+00,  4.55892950e+00, 4.70154722e+00, 4.84862648e+00, 5.00030684e+00,  5.15673224e+00, 5.31805113e+00, 5.48441658e+00, 5.65598646e+00,  5.83292359e+00, 6.01539588e+00, 6.20357648e+00, 6.39764396e+00,  6.59778248e+00, 6.80418197e+00, 7.01703829e+00, 7.23655342e+00,  7.46293569e+00, 7.69639990e+00, 7.93716762e+00, 8.18546731e+00,  8.44153460e+00, 8.70561248e+00, 8.97795155e+00, 9.25881025e+00,  9.54845510e+00, 9.84716096e+00, 1.01552113e+01, 1.04728984e+01,  1.08005237e+01, 1.11383983e+01, 1.14868426e+01, 1.18461873e+01,  1.22167735e+01, 1.25989528e+01, 1.29930878e+01, 1.33995527e+01,  1.38187331e+01, 1.42510267e+01, 1.46968439e+01, 1.51566076e+01,  1.56307542e+01, 1.61197336e+01, 1.66240098e+01, 1.71440614e+01,  1.76803818e+01, 1.82334800e+01, 1.88038809e+01, 1.93921257e+01,  1.99987727e+01, 2.06243975e+01, 2.12695939e+01, 2.19349740e+01,  2.26211693e+01, 2.33288310e+01, 2.40586305e+01, 2.48112606e+01,  2.55874352e+01, 2.63878910e+01, 2.72133877e+01, 2.80647085e+01,  2.89426612e+01, 2.98480792e+01, 3.07818214e+01, 3.17447741e+01,  3.27378510e+01, 3.37619944e+01, 3.48181763e+01, 3.59073989e+01,  3.70306958e+01, 3.81891330e+01, 3.93838098e+01, 4.06158599e+01,  4.18864524e+01, 4.31967930e+01, 4.45481252e+01, 4.59417313e+01,  4.73789339e+01, 4.88610967e+01, 5.03896263e+01, 5.19659730e+01,  5.35916329e+01, 5.52681486e+01, 5.69971109e+01, 5.87801607e+01,  6.06189899e+01, 6.25153435e+01, 6.44710211e+01, 6.64878784e+01,  6.85678294e+01, 7.07128478e+01, 7.29249692e+01, 7.52062927e+01,  7.75589832e+01, 7.99852734e+01, 8.24874655e+01, 8.50679342e+01,  8.77291280e+01, 9.04735724e+01, 9.33038717e+01, 9.62227117e+01,  9.92328623e+01, 1.02337180e+02, 1.05538610e+02, 1.08840192e+02,  1.12245057e+02, 1.15756437e+02, 1.19377664e+02, 1.23112175e+02,  1.26963513e+02, 1.30935333e+02, 1.35031404e+02, 1.39255613e+02,  1.43611969e+02, 1.48104605e+02, 1.52737785e+02, 1.57515906e+02,  1.62443501e+02, 1.67525247e+02, 1.72765966e+02, 1.78170632e+02,  1.83744372e+02, 1.89492477e+02, 1.95420401e+02, 2.01533769e+02,  2.07838382e+02, 2.14340223e+02, 2.21045463e+02, 2.27960464e+02,  2.35091788e+02, 2.42446202e+02, 2.50030685e+02, 2.57852435e+02,  2.65918874e+02, 2.74237657e+02, 2.82816678e+02, 2.91664078e+02,  3.00788252e+02, 3.10197859e+02, 3.19901828e+02, 3.29909369e+02,  3.40229977e+02, 3.50873446e+02, 3.61849876e+02, 3.73169684e+02,  3.84843611e+02, 3.96882735e+02, 4.09298482e+02, 4.22102632e+02,  4.35307336e+02, 4.48925126e+02, 4.62968923e+02, 4.77452054e+02,  4.92388263e+02, 5.07791724e+02, 5.23677054e+02, 5.40059328e+02,  5.56954091e+02, 5.74377375e+02, 5.92345715e+02, 6.10876161e+02,  6.29986298e+02, 6.49694260e+02, 6.70018750e+02, 6.90979055e+02,  7.12595063e+02, 7.34887289e+02, 7.57876886e+02, 7.81585671e+02,  8.06036141e+02, 8.31251499e+02, 8.57255673e+02, 8.84073340e+02,  9.11729948e+02, 9.40251743e+02, 9.69665789e+02, 1.00000000e+03],
    'y': [
0.05803514,  0.05597487,  0.10924021,  0.16264132,  0.17175118,  0.23487453
,  0.29503384,  0.32198954,  0.37530462,  0.4653316,  0.54180126,  0.60159797
,  0.69467999,  0.75980009,  0.8635243,  0.92805247,  0.99558858,  1.10939441
,  1.18853009,  1.28590403,  1.38936419,  1.48024726,  1.57884581,  1.672604
,  1.76533617,  1.88008752,  1.95937525,  2.06414046,  2.15145734,  2.23642829
,  2.31442407,  2.39745762,  2.48232541,  2.56371496,  2.64433167,  2.71997749
,  2.80221979,  2.88901732,  2.96896705,  3.06003465,  3.13985302,  3.21973653
,  3.31224358,  3.38030569,  3.49829314,  3.63145561,  3.80546871,  3.96240221
,  4.15150592,  4.34238406,  4.56345001,  4.83001684,  5.05765066,  5.29532714
,  5.57114975,  5.77614981,  6.09770381,  6.37391579,  6.63965333,  6.95677729
,  7.2588721,  7.54657675,  7.82326054,  8.14188036,  8.37026749,  8.65868423
,  8.93637694,  9.21829746,  9.46272508,  9.71671745, 10.01494718, 10.21941181
, 10.49009924, 10.73463316, 10.937714,  11.15578355, 11.32730885, 11.45741685
, 11.60457901, 11.69848186, 11.76374654, 11.82938455, 11.83170719, 11.85902289
, 11.84539268, 11.81824472, 11.77198702, 11.72054588, 11.67735165, 11.6198562
, 11.57784886, 11.48649157, 11.41628722, 11.30384335, 11.16093346, 10.99607313
, 10.79728037, 10.5713757,  10.35488752, 10.09122546,  9.80817701,  9.53018799
,  9.2183292,  8.90719422,  8.54641689,  8.18933277,  7.86660366,  7.52142041
,  7.18500778,  6.86826757,  6.54171467,  6.23453046,  5.92920972,  5.62157031
,  5.33891594,  5.04676316,  4.77618646,  4.50937371,  4.24917796,  4.00202176
,  3.76373436,  3.53992844,  3.31897292,  3.11097665,  2.90305927,  2.70965619
,  2.5290938,  2.35453961,  2.19164573,  2.04242966,  1.92028101,  1.85367483
,  1.76657536,  1.70492112,  1.68938454,  1.6387508,  1.61527339,  1.62868824
,  1.59023822,  1.59395766,  1.62022451,  1.65856554,  1.78821641,  1.82034693
,  1.92460069,  1.96259906,  2.06459703,  2.04649815,  2.11812968,  2.12436663
,  2.15862543,  2.18193741,  2.22007382,  2.30842776,  2.36184358,  2.52212438
,  2.54606072,  2.60065413,  2.76579385,  2.79210374,  2.78788399,  2.8998527
,  2.86781786,  2.89589962,  2.88682149,  2.88469145,  2.9143968,  2.92618101
,  2.95290248,  2.90306702,  2.93950239,  2.87745588,  2.89392329,  2.85427998
,  2.85139902,  2.84800219,  2.8392252,  2.85327331,  2.82383223,  2.84153693
,  2.85700118,  2.86017793,  2.88681593,  2.91355302,  2.96947649,  3.00351705
,  3.05075799,  3.1194526,  3.18452363,  3.27340813,  3.36464221,  3.46217517
,  3.57266676,  3.69324948,  3.82259073,  3.96349462,  4.1168513,  4.27962866
,  4.45339379,  4.63770261,  4.83095724,  5.03430945,  5.24619767,  5.46640966
,  5.69451954,  5.93046452,  6.17393788,  6.42340053,  6.6790845,  6.93979326
,  7.20492571,  7.47408405,  7.74602387,  8.0201452,  8.29554845,  8.57222446
,  8.84810691,  9.12373446,  9.39745043,  9.66834927,  9.93565566, 10.19913055
, 10.45623067, 10.705655,  10.94684931, 11.17844771, 11.399753,  11.61005453
, 11.80855042, 11.99390572, 12.16492512, 12.3194005,  12.45391325, 12.57109221
, 12.66668081, 12.74198935, 12.79621355, 12.82829378, 12.8373249,  12.82284413
, 12.77848573, 12.71179701, 12.61902069, 12.49991254, 12.35845569, 12.19267353
, 12.0034048,  11.79219166, 11.55730328, 11.30339693, 11.02998888, 10.73870116
, 10.43183045, 10.11158714,  9.77882603,  9.43551415,  9.08275111,  8.72430235
,  8.36002591,  7.99329407,  7.62495876,  7.25700907,  6.89147975,  6.52978933
,  6.17237784,  5.82163713,  5.47857179,  5.14405128,  4.81944571,  4.50567673
,  4.20267572,  3.91209668,  3.63385797,  3.36832299,  3.11589482,  2.87664301
,  2.65076915,  2.43809392,  2.23843039,  2.05150688,  1.87707629,  1.7147344
,  1.56393832,  1.42431672,  1.29523601,  1.17627651,  1.06683796,  0.96632381
,  0.87420511,  0.78997069,  0.71300111,  0.64284212,  0.57897822,  0.52091731
,  0.46820653,  0.42043738,  0.3772056,  0.33810151,  0.30279414,  0.27094825
    ],
    'dy': [
        0.00235888, 0.00234909, 0.00401945, 0.00569966, 0.00592143, 0.00779714
        , 0.00921613, 0.00977046, 0.01094624, 0.01310575, 0.0146819, 0.0158457
        , 0.01752906, 0.01866204, 0.02031568, 0.0211073, 0.02187735, 0.02355228
        , 0.02445912, 0.02560293, 0.02678634, 0.02779131, 0.0287703, 0.02969265
        , 0.03070924, 0.03180022, 0.03258733, 0.03367721, 0.03435772,
        0.03510928
        , 0.03578286, 0.03650593, 0.03712457, 0.03775059, 0.03834302,
        0.03882841
        , 0.03940247, 0.04002895, 0.04054601, 0.04121413, 0.04178362,
        0.04236471
        , 0.04315074, 0.04367645, 0.04475704, 0.04598934, 0.04781606,
        0.04950378
        , 0.05171577, 0.05390601, 0.05673236, 0.06010318, 0.06303977,
        0.06620972
        , 0.07002342, 0.07301465, 0.0769838, 0.08071167, 0.08432752, 0.08843925
        , 0.09238908, 0.09664201, 0.10040864, 0.10484039, 0.10863789,
        0.11283446
        , 0.11676672, 0.1203971, 0.12429478, 0.12790194, 0.13105735, 0.13440463
        , 0.13714006, 0.13960741, 0.14209957, 0.14369921, 0.14572019,
        0.14710253
        , 0.14866303, 0.14982841, 0.15079287, 0.15172687, 0.15208843,
        0.15277021
        , 0.15305107, 0.15325112, 0.15321891, 0.15312386, 0.15317967,
        0.15308373
        , 0.15323442, 0.15275609, 0.15272881, 0.15230139, 0.15146175,
        0.15043597
        , 0.14902732, 0.14728737, 0.14571865, 0.14331723, 0.14082565,
        0.13831476
        , 0.13515891, 0.13214864, 0.12808322, 0.12418921, 0.12027461,
        0.11619338
        , 0.11194376, 0.10801487, 0.1038071, 0.09989958, 0.09579771, 0.09174581
        , 0.08798985, 0.08391353, 0.08019919, 0.07640965, 0.07267852,
        0.06906388
        , 0.06552513, 0.06216723, 0.05876194, 0.05556174, 0.05234213,
        0.04931459
        , 0.04640686, 0.0435652, 0.04101631, 0.04921451, 0.05247942, 0.05498395
        , 0.06727915, 0.06843886, 0.08732949, 0.0456023, 0.05485943, 0.05855987
        , 0.06231902, 0.08230355, 0.0621109, 0.05512222, 0.06017081, 0.05730026
        , 0.06155109, 0.0568558, 0.06171111, 0.05656287, 0.08803736, 0.05851605
        , 0.07793641, 0.0536779, 0.05128056, 0.07530208, 0.05246809, 0.05420648
        , 0.06538222, 0.05090974, 0.05067258, 0.05101395, 0.05008017,
        0.05444524
        , 0.0474591, 0.047798, 0.04686741, 0.04628257, 0.04590483, 0.0454182
        , 0.04523406, 0.04379325, 0.04378022, 0.04204853, 0.04159698,
        0.04026272
        , 0.03938778, 0.0387576, 0.03795885, 0.03763334, 0.03663563, 0.0365205
        , 0.03650347, 0.03630913, 0.03666295, 0.03722051, 0.03829504,
        0.03939333
        , 0.04089187, 0.04277416, 0.04977124, 0.05259031, 0.05545887,
        0.05883842
        , 0.0623804, 0.066024, 0.06784709, 0.07108123, 0.07432145, 0.07755623
        , 0.08075299, 0.08387039, 0.09243257, 0.09631425, 0.10008495,
        0.10375996
        , 0.10731353, 0.11072309, 0.11399959, 0.11715503, 0.12012384,
        0.12300693
        , 0.12574932, 0.12835908, 0.13085263, 0.13321728, 0.13545792,
        0.13763895
        , 0.13970581, 0.14170012, 0.14363438, 0.14551738, 0.14736746,
        0.14919376
        , 0.15100512, 0.15281201, 0.15462739, 0.15646079, 0.15844811,
        0.16054318
        , 0.16262909, 0.1645389, 0.16658395, 0.16867685, 0.17060487, 0.17188265
        , 0.1737448, 0.17554026, 0.17723966, 0.17880872, 0.18022293, 0.18147527
        , 0.18198042, 0.18231937, 0.18238958, 0.18229732, 0.18202856,
        0.18154587
        , 0.18084514, 0.17993609, 0.17876541, 0.17924537, 0.17734952,
        0.17514222
        , 0.17263259, 0.16983643, 0.16676057, 0.16341511, 0.15980239,
        0.15597263
        , 0.15190223, 0.14764285, 0.14320301, 0.13859162, 0.13384515, 0.1289862
        , 0.12401489, 0.11897802, 0.1138832, 0.10875914, 0.10362452, 0.09852004
        , 0.09341777, 0.0883757, 0.08279331, 0.07797734, 0.07385477, 0.06922691
        , 0.06473248, 0.06038662, 0.05620236, 0.05218805, 0.0483553, 0.04470951
        , 0.04125296, 0.0379906, 0.03491888, 0.03204007, 0.02934931, 0.02683881
        , 0.02450516, 0.02234151, 0.02033781, 0.01848826, 0.016784, 0.01521657
        , 0.01377709, 0.01245838, 0.01125281, 0.01015102, 0.00914709,
        0.00823286           ]
  },
  'Model3body': {
    'x': [1.00000000e-01, 1.03128316e-01, 1.06354496e-01, 1.09681601e-01,
          1.13112788e-01, 1.16651313e-01, 1.20300535e-01, 1.24063916e-01,  1.27945027e-01, 1.31947552e-01, 1.36075289e-01, 1.40332154e-01,  1.44722187e-01, 1.49249555e-01, 1.53918552e-01, 1.58733611e-01,  1.63699300e-01, 1.68820332e-01, 1.74101565e-01, 1.79548012e-01,  1.85164842e-01, 1.90957383e-01, 1.96931134e-01, 2.03091762e-01,  2.09445114e-01, 2.15997219e-01, 2.22754295e-01, 2.29722754e-01,  2.36909207e-01, 2.44320476e-01, 2.51963593e-01, 2.59845810e-01,  2.67974609e-01, 2.76357701e-01, 2.85003044e-01, 2.93918840e-01,  3.03113550e-01, 3.12595900e-01, 3.22374888e-01, 3.32459793e-01,  3.42860186e-01, 3.53585937e-01, 3.64647222e-01, 3.76054540e-01,  3.87818715e-01, 3.99950910e-01, 4.12462638e-01, 4.25365773e-01,  4.38672559e-01, 4.52395623e-01, 4.66547988e-01, 4.81143084e-01,  4.96194760e-01, 5.11717301e-01, 5.27725435e-01, 5.44234355e-01,  5.61259726e-01, 5.78817704e-01, 5.96924951e-01, 6.15598650e-01,  6.34856522e-01, 6.54716840e-01, 6.75198452e-01, 6.96320794e-01,  7.18103909e-01, 7.40568469e-01, 7.63735792e-01, 7.87627861e-01,  8.12267350e-01, 8.37677640e-01, 8.63882844e-01, 8.90907830e-01,  9.18778243e-01, 9.47520530e-01, 9.77161967e-01, 1.00773068e+00,  1.03925568e+00, 1.07176689e+00, 1.10529514e+00, 1.13987227e+00,  1.17553107e+00, 1.21230540e+00, 1.25023015e+00, 1.28934130e+00,  1.32967597e+00, 1.37127243e+00, 1.41417017e+00, 1.45840988e+00,  1.50403355e+00, 1.55108448e+00, 1.59960730e+00, 1.64964807e+00,  1.70125428e+00, 1.75447489e+00, 1.80936041e+00, 1.86596292e+00,  1.92433614e+00, 1.98453546e+00, 2.04661800e+00, 2.11064268e+00,  2.17667025e+00, 2.24476338e+00, 2.31498667e+00, 2.38740677e+00,  2.46209240e+00, 2.53911443e+00, 2.61854596e+00, 2.70046235e+00,  2.78494135e+00, 2.87206312e+00, 2.96191033e+00, 3.05456825e+00,  3.15012480e+00, 3.24867066e+00, 3.35029934e+00, 3.45510729e+00,  3.56319397e+00, 3.67466194e+00, 3.78961698e+00, 3.90816818e+00,  4.03042803e+00, 4.15651256e+00, 4.28654141e+00, 4.42063797e+00,  4.55892950e+00, 4.70154722e+00, 4.84862648e+00, 5.00030684e+00,  5.15673224e+00, 5.31805113e+00, 5.48441658e+00, 5.65598646e+00,  5.83292359e+00, 6.01539588e+00, 6.20357648e+00, 6.39764396e+00,  6.59778248e+00, 6.80418197e+00, 7.01703829e+00, 7.23655342e+00,  7.46293569e+00, 7.69639990e+00, 7.93716762e+00, 8.18546731e+00,  8.44153460e+00, 8.70561248e+00, 8.97795155e+00, 9.25881025e+00,  9.54845510e+00, 9.84716096e+00, 1.01552113e+01, 1.04728984e+01,  1.08005237e+01, 1.11383983e+01, 1.14868426e+01, 1.18461873e+01,  1.22167735e+01, 1.25989528e+01, 1.29930878e+01, 1.33995527e+01,  1.38187331e+01, 1.42510267e+01, 1.46968439e+01, 1.51566076e+01,  1.56307542e+01, 1.61197336e+01, 1.66240098e+01, 1.71440614e+01,  1.76803818e+01, 1.82334800e+01, 1.88038809e+01, 1.93921257e+01,  1.99987727e+01, 2.06243975e+01, 2.12695939e+01, 2.19349740e+01,  2.26211693e+01, 2.33288310e+01, 2.40586305e+01, 2.48112606e+01,  2.55874352e+01, 2.63878910e+01, 2.72133877e+01, 2.80647085e+01,  2.89426612e+01, 2.98480792e+01, 3.07818214e+01, 3.17447741e+01,  3.27378510e+01, 3.37619944e+01, 3.48181763e+01, 3.59073989e+01,  3.70306958e+01, 3.81891330e+01, 3.93838098e+01, 4.06158599e+01,  4.18864524e+01, 4.31967930e+01, 4.45481252e+01, 4.59417313e+01,  4.73789339e+01, 4.88610967e+01, 5.03896263e+01, 5.19659730e+01,  5.35916329e+01, 5.52681486e+01, 5.69971109e+01, 5.87801607e+01,  6.06189899e+01, 6.25153435e+01, 6.44710211e+01, 6.64878784e+01,  6.85678294e+01, 7.07128478e+01, 7.29249692e+01, 7.52062927e+01,  7.75589832e+01, 7.99852734e+01, 8.24874655e+01, 8.50679342e+01,  8.77291280e+01, 9.04735724e+01, 9.33038717e+01, 9.62227117e+01,  9.92328623e+01, 1.02337180e+02, 1.05538610e+02, 1.08840192e+02,  1.12245057e+02, 1.15756437e+02, 1.19377664e+02, 1.23112175e+02,  1.26963513e+02, 1.30935333e+02, 1.35031404e+02, 1.39255613e+02,  1.43611969e+02, 1.48104605e+02, 1.52737785e+02, 1.57515906e+02,  1.62443501e+02, 1.67525247e+02, 1.72765966e+02, 1.78170632e+02,  1.83744372e+02, 1.89492477e+02, 1.95420401e+02, 2.01533769e+02,  2.07838382e+02, 2.14340223e+02, 2.21045463e+02, 2.27960464e+02,  2.35091788e+02, 2.42446202e+02, 2.50030685e+02, 2.57852435e+02,  2.65918874e+02, 2.74237657e+02, 2.82816678e+02, 2.91664078e+02,  3.00788252e+02, 3.10197859e+02, 3.19901828e+02, 3.29909369e+02,  3.40229977e+02, 3.50873446e+02, 3.61849876e+02, 3.73169684e+02,  3.84843611e+02, 3.96882735e+02, 4.09298482e+02, 4.22102632e+02,  4.35307336e+02, 4.48925126e+02, 4.62968923e+02, 4.77452054e+02,  4.92388263e+02, 5.07791724e+02, 5.23677054e+02, 5.40059328e+02,  5.56954091e+02, 5.74377375e+02, 5.92345715e+02, 6.10876161e+02,  6.29986298e+02, 6.49694260e+02, 6.70018750e+02, 6.90979055e+02,  7.12595063e+02, 7.34887289e+02, 7.57876886e+02, 7.81585671e+02,  8.06036141e+02, 8.31251499e+02, 8.57255673e+02, 8.84073340e+02,  9.11729948e+02, 9.40251743e+02, 9.69665789e+02, 1.00000000e+03],
    'y': [0.06827396, 0.06819649, 0.12414418, 0.18161386, 0.19302117, 0.26095639
, 0.32272144, 0.35278834, 0.40661878, 0.49796243, 0.57856151, 0.63856876
, 0.73245402, 0.79970929, 0.90199166, 0.96951476, 1.03919186, 1.14808591
, 1.22867966, 1.32267072, 1.42423368, 1.5128252,  1.60892228, 1.69971364
, 1.78842392, 1.90083716, 1.97785734, 2.07899324, 2.16446106, 2.24737933
, 2.32156172, 2.40287219, 2.48481375, 2.56401866, 2.64345923, 2.71707564
, 2.79809474, 2.88339219, 2.96281631, 3.0529774,  3.13292383, 3.21303372
, 3.30621619, 3.37575074, 3.49630893, 3.63411106, 3.8133492,  3.97355457
, 4.16879769, 4.36322038, 4.58907938, 4.8610155,  5.09233601, 5.33248994
, 5.61376548, 5.82245349, 6.1439236,  6.42147533, 6.6869273,  7.00291723
, 7.30230496, 7.59123436, 7.86490318, 8.17996417, 8.4098891,  8.69310835
, 8.96470006, 9.23708514, 9.47719608, 9.72319497, 10.00295517, 10.2041265,
 10.45488234, 10.68291544, 10.87389265, 11.06816194, 11.22747121, 11.34225861,
 11.47290109, 11.5542153, 11.60981419, 11.66252553, 11.65449503, 11.67209654,
 11.64998643, 11.61548529, 11.56308693, 11.50621454, 11.45696056, 11.3939211,
 11.34426684, 11.24976628, 11.17117868, 11.05122313, 10.90483221, 10.73619951,
 10.53584659, 10.31010403, 10.09265096, 9.83238733, 9.55300083, 9.27880128
, 8.97424012, 8.66983343, 8.32087962, 7.97497377, 7.6629335,  7.32934019
, 7.00486978, 6.69869179, 6.38443218, 6.08889507, 5.79681317, 5.5029825
, 5.2345372,  4.95933363, 4.70601592, 4.45858207, 4.21986137, 3.99603485
, 3.78321769, 3.5873342,  3.39716422, 3.22238922, 3.05144922, 2.89742883
, 2.75941028, 2.63108723, 2.51733677, 2.41787388, 2.32972028, 2.25556674
, 2.19139845, 2.13985299, 2.09728959, 2.06498792, 2.04294544, 2.02929912
, 2.02475898, 2.02689766, 2.03650736, 2.05147607, 2.07317617, 2.09928231
, 2.12999025, 2.16394835, 2.20104615, 2.24002863, 2.28072686, 2.322219
, 2.36382914, 2.40513375, 2.44537135, 2.48416315, 2.52066343, 2.55483893
, 2.5859614,  2.61370972, 2.63795036, 2.65817022, 2.6741372,  2.68586594
, 2.69295825, 2.69547996, 2.6935891,  2.68717606, 2.67650362, 2.66190531
, 2.64358986, 2.62204454, 2.59780929, 2.57148809, 2.54378527, 2.51548107
, 2.48761905, 2.46108716, 2.43695088, 2.41660888, 2.40108602, 2.39173634
, 2.39013286, 2.39738253, 2.41502025, 2.44456617, 2.48713094, 2.54422801
, 2.61708725, 2.70671516, 2.81436488, 2.94075435, 3.08667431, 3.2528834
, 3.439436, 3.64680818, 3.87507779, 4.12357828, 4.39240511, 4.68087279
, 4.98758789, 5.31226041, 5.65350555, 6.0091756,  6.37870126, 6.76013641
, 7.15059059, 7.54938726, 7.95416366, 8.36139773, 8.7704854,  9.17879274
, 9.58236375, 9.98081166, 10.37140415, 10.75000667, 11.11654729, 11.46838371,
 11.80148393, 12.11616461, 12.41006866, 12.67951172, 12.92527695, 13.14545508,
 13.33692613, 13.50098638, 13.63629334, 13.74045318, 13.81528202, 13.86006937,
 13.87322994, 13.85705795, 13.81147765, 13.73570721, 13.63243505, 13.50214991,
 13.34480246, 13.16333432, 12.95868418, 12.73140665, 12.48452394, 12.21928338,
 11.93666464, 11.63960935, 11.32951582, 11.00760374, 10.67658896, 10.33787423
, 9.99274049, 9.64357414, 9.2916609,  8.93820094, 8.58518637, 8.23369856
, 7.88475853, 7.53994001, 7.20007137, 6.86594419, 6.53871724, 6.21895894
, 5.90721334, 5.60426149, 5.31042483, 5.0260154,  4.75148872, 4.486954
, 4.23252046, 3.98838176, 3.75447745, 3.53075452, 3.31720715, 3.11365061
, 2.91991139, 2.73584322, 2.56117859, 2.39566321, 2.2390608,  2.09105607
, 1.95134889, 1.81965333, 1.6956344,  1.57897407, 1.46936691, 1.36647973
, 1.26999706, 1.17961565, 1.09501891, 1.01590799, 0.94199617, 0.8729928
, 0.80862424, 0.74862887, 0.69274697, 0.64073456, 0.59235933, 0.54739413
, 0.50562615, 0.46685349, 0.43088094, 0.39752624, 0.36661689, 0.33798772],
    'dy': [0.0026093, 0.00299487, 0.00445117, 0.00615263, 0.00685115, 0.0086294
, 0.0101373, 0.01110111, 0.01198604, 0.01416713, 0.01592044, 0.01710864
, 0.01887844, 0.02006913, 0.02166351, 0.02286556, 0.02381787, 0.02511644
, 0.02633452, 0.02720851, 0.02848743, 0.02944632, 0.03047418, 0.03138025
, 0.03237143, 0.03351092, 0.03440476, 0.03548135, 0.03623861, 0.03711523
, 0.03763308, 0.03848357, 0.03911314, 0.03986918, 0.04073222, 0.04133234
, 0.04208255, 0.04293111, 0.04373546, 0.04460798, 0.04551753, 0.04644938
, 0.04756529, 0.0486377, 0.05031028, 0.05251463, 0.05536494, 0.05803317
, 0.06171969, 0.0646868, 0.06877788, 0.07350232, 0.07750604, 0.0815644
, 0.08684024, 0.09121194, 0.09578223, 0.10036806, 0.10453911, 0.10905452
, 0.11306328, 0.11844322, 0.12222135, 0.12676924, 0.13160661, 0.1356726
, 0.13886782, 0.14164672, 0.14511375, 0.14777485, 0.14823846, 0.15121733
, 0.1513476, 0.15142335, 0.15241722, 0.15124802, 0.15208354, 0.1520806
, 0.15247983, 0.15302011, 0.15367964, 0.1545272, 0.15505994, 0.15615113
, 0.15698357, 0.1578299, 0.15851226, 0.15915327, 0.16012187, 0.16094167
, 0.1623388, 0.1627849, 0.16445938, 0.16573125, 0.16599765, 0.16622389
, 0.16584293, 0.16485458, 0.16440309, 0.1623345, 0.16029427, 0.15806842
, 0.15472188, 0.15179954, 0.14683437, 0.14221836, 0.13756167, 0.13269781
, 0.12752196, 0.12302935, 0.11802107, 0.1135605, 0.10873406, 0.10406279
, 0.09979468, 0.09495496, 0.09066783, 0.0861627, 0.08174999, 0.07748356
, 0.07332527, 0.06943681, 0.0655063, 0.06198361, 0.05850939, 0.05551205
, 0.05293584, 0.05078393, 0.04921812, 0.04832846, 0.04810504, 0.04865218
, 0.04990052, 0.05186962, 0.05447392, 0.05764983, 0.06134305, 0.06546051
, 0.06994566, 0.07471195, 0.07975121, 0.08497462, 0.09034269, 0.09580548
, 0.1013228, 0.10685475, 0.11237087, 0.11782709, 0.12318976, 0.12843763
, 0.13352557, 0.13842263, 0.14310389, 0.14755113, 0.15172697, 0.1556236
, 0.15919994, 0.16244378, 0.16534216, 0.16786501, 0.17000301, 0.1717599
, 0.17310441, 0.1740317, 0.17455511, 0.17464576, 0.17430915, 0.17355649
, 0.17236655, 0.1707524, 0.16873104, 0.1662921, 0.16346326, 0.16027388
, 0.15674644, 0.1529409, 0.14892595, 0.14479872, 0.14069007, 0.13676565
, 0.13324839, 0.13038824, 0.12848334, 0.12785435, 0.12879077, 0.13154556
, 0.13626416, 0.14298901, 0.15165943, 0.16208385, 0.17407152, 0.18736389
, 0.20167558, 0.2167891, 0.23243183, 0.24834711, 0.26436285, 0.2802457
, 0.29576896, 0.31081509, 0.3252028, 0.33872163, 0.35131505, 0.36285288
, 0.37315136, 0.38220753, 0.38995048, 0.39623951, 0.40112748, 0.40458523
, 0.40654013, 0.40709012, 0.40626446, 0.40406016, 0.40061969, 0.39603367
, 0.39037048, 0.38381831, 0.37652603, 0.36862816, 0.36035199, 0.35189298
, 0.34342922, 0.33520497, 0.32742363, 0.32025435, 0.31390432, 0.30851367
, 0.30415986, 0.30185121, 0.29885219, 0.29782557, 0.29781531, 0.29868454
, 0.30023839, 0.30235314, 0.30483205, 0.30745386, 0.3100996, 0.31255833
, 0.31472215, 0.31648878, 0.31774004, 0.31836473, 0.3183516, 0.31764171
, 0.31618538, 0.31401622, 0.31112228, 0.30750045, 0.30321291, 0.29827949
, 0.29272731, 0.2866324, 0.28003234, 0.27297041, 0.26552469, 0.25773953
, 0.24966352, 0.2413685, 0.2328978, 0.2242977, 0.21562901, 0.20692918
, 0.19823755, 0.18960187, 0.18105127, 0.17261606, 0.16433079, 0.15621577
, 0.14829186, 0.14058171, 0.13309739, 0.12585111, 0.11885563, 0.11211598
, 0.10563722, 0.09942436, 0.09347693, 0.08779455, 0.08237655, 0.07721851
, 0.07231621, 0.06766512, 0.06325829, 0.05908902, 0.05515047, 0.05143425
, 0.04793236, 0.04463674, 0.04153848, 0.0386291, 0.03590018, 0.03334292
, 0.03094893, 0.02871003, 0.02661789, 0.02466465, 0.02284269, 0.02114444]
},
  'ModelBosa': {
    'x': [1.00000000e-01, 1.03128316e-01, 1.06354496e-01, 1.09681601e-01,
          1.13112788e-01, 1.16651313e-01, 1.20300535e-01, 1.24063916e-01,  1.27945027e-01, 1.31947552e-01, 1.36075289e-01, 1.40332154e-01,  1.44722187e-01, 1.49249555e-01, 1.53918552e-01, 1.58733611e-01,  1.63699300e-01, 1.68820332e-01, 1.74101565e-01, 1.79548012e-01,  1.85164842e-01, 1.90957383e-01, 1.96931134e-01, 2.03091762e-01,  2.09445114e-01, 2.15997219e-01, 2.22754295e-01, 2.29722754e-01,  2.36909207e-01, 2.44320476e-01, 2.51963593e-01, 2.59845810e-01,  2.67974609e-01, 2.76357701e-01, 2.85003044e-01, 2.93918840e-01,  3.03113550e-01, 3.12595900e-01, 3.22374888e-01, 3.32459793e-01,  3.42860186e-01, 3.53585937e-01, 3.64647222e-01, 3.76054540e-01,  3.87818715e-01, 3.99950910e-01, 4.12462638e-01, 4.25365773e-01,  4.38672559e-01, 4.52395623e-01, 4.66547988e-01, 4.81143084e-01,  4.96194760e-01, 5.11717301e-01, 5.27725435e-01, 5.44234355e-01,  5.61259726e-01, 5.78817704e-01, 5.96924951e-01, 6.15598650e-01,  6.34856522e-01, 6.54716840e-01, 6.75198452e-01, 6.96320794e-01,  7.18103909e-01, 7.40568469e-01, 7.63735792e-01, 7.87627861e-01,  8.12267350e-01, 8.37677640e-01, 8.63882844e-01, 8.90907830e-01,  9.18778243e-01, 9.47520530e-01, 9.77161967e-01, 1.00773068e+00,  1.03925568e+00, 1.07176689e+00, 1.10529514e+00, 1.13987227e+00,  1.17553107e+00, 1.21230540e+00, 1.25023015e+00, 1.28934130e+00,  1.32967597e+00, 1.37127243e+00, 1.41417017e+00, 1.45840988e+00,  1.50403355e+00, 1.55108448e+00, 1.59960730e+00, 1.64964807e+00,  1.70125428e+00, 1.75447489e+00, 1.80936041e+00, 1.86596292e+00,  1.92433614e+00, 1.98453546e+00, 2.04661800e+00, 2.11064268e+00,  2.17667025e+00, 2.24476338e+00, 2.31498667e+00, 2.38740677e+00,  2.46209240e+00, 2.53911443e+00, 2.61854596e+00, 2.70046235e+00,  2.78494135e+00, 2.87206312e+00, 2.96191033e+00, 3.05456825e+00,  3.15012480e+00, 3.24867066e+00, 3.35029934e+00, 3.45510729e+00,  3.56319397e+00, 3.67466194e+00, 3.78961698e+00, 3.90816818e+00,  4.03042803e+00, 4.15651256e+00, 4.28654141e+00, 4.42063797e+00,  4.55892950e+00, 4.70154722e+00, 4.84862648e+00, 5.00030684e+00,  5.15673224e+00, 5.31805113e+00, 5.48441658e+00, 5.65598646e+00,  5.83292359e+00, 6.01539588e+00, 6.20357648e+00, 6.39764396e+00,  6.59778248e+00, 6.80418197e+00, 7.01703829e+00, 7.23655342e+00,  7.46293569e+00, 7.69639990e+00, 7.93716762e+00, 8.18546731e+00,  8.44153460e+00, 8.70561248e+00, 8.97795155e+00, 9.25881025e+00,  9.54845510e+00, 9.84716096e+00, 1.01552113e+01, 1.04728984e+01,  1.08005237e+01, 1.11383983e+01, 1.14868426e+01, 1.18461873e+01,  1.22167735e+01, 1.25989528e+01, 1.29930878e+01, 1.33995527e+01,  1.38187331e+01, 1.42510267e+01, 1.46968439e+01, 1.51566076e+01,  1.56307542e+01, 1.61197336e+01, 1.66240098e+01, 1.71440614e+01,  1.76803818e+01, 1.82334800e+01, 1.88038809e+01, 1.93921257e+01,  1.99987727e+01, 2.06243975e+01, 2.12695939e+01, 2.19349740e+01,  2.26211693e+01, 2.33288310e+01, 2.40586305e+01, 2.48112606e+01,  2.55874352e+01, 2.63878910e+01, 2.72133877e+01, 2.80647085e+01,  2.89426612e+01, 2.98480792e+01, 3.07818214e+01, 3.17447741e+01,  3.27378510e+01, 3.37619944e+01, 3.48181763e+01, 3.59073989e+01,  3.70306958e+01, 3.81891330e+01, 3.93838098e+01, 4.06158599e+01,  4.18864524e+01, 4.31967930e+01, 4.45481252e+01, 4.59417313e+01,  4.73789339e+01, 4.88610967e+01, 5.03896263e+01, 5.19659730e+01,  5.35916329e+01, 5.52681486e+01, 5.69971109e+01, 5.87801607e+01,  6.06189899e+01, 6.25153435e+01, 6.44710211e+01, 6.64878784e+01,  6.85678294e+01, 7.07128478e+01, 7.29249692e+01, 7.52062927e+01,  7.75589832e+01, 7.99852734e+01, 8.24874655e+01, 8.50679342e+01,  8.77291280e+01, 9.04735724e+01, 9.33038717e+01, 9.62227117e+01,  9.92328623e+01, 1.02337180e+02, 1.05538610e+02, 1.08840192e+02,  1.12245057e+02, 1.15756437e+02, 1.19377664e+02, 1.23112175e+02,  1.26963513e+02, 1.30935333e+02, 1.35031404e+02, 1.39255613e+02,  1.43611969e+02, 1.48104605e+02, 1.52737785e+02, 1.57515906e+02,  1.62443501e+02, 1.67525247e+02, 1.72765966e+02, 1.78170632e+02,  1.83744372e+02, 1.89492477e+02, 1.95420401e+02, 2.01533769e+02,  2.07838382e+02, 2.14340223e+02, 2.21045463e+02, 2.27960464e+02,  2.35091788e+02, 2.42446202e+02, 2.50030685e+02, 2.57852435e+02,  2.65918874e+02, 2.74237657e+02, 2.82816678e+02, 2.91664078e+02,  3.00788252e+02, 3.10197859e+02, 3.19901828e+02, 3.29909369e+02,  3.40229977e+02, 3.50873446e+02, 3.61849876e+02, 3.73169684e+02,  3.84843611e+02, 3.96882735e+02, 4.09298482e+02, 4.22102632e+02,  4.35307336e+02, 4.48925126e+02, 4.62968923e+02, 4.77452054e+02,  4.92388263e+02, 5.07791724e+02, 5.23677054e+02, 5.40059328e+02,  5.56954091e+02, 5.74377375e+02, 5.92345715e+02, 6.10876161e+02,  6.29986298e+02, 6.49694260e+02, 6.70018750e+02, 6.90979055e+02,  7.12595063e+02, 7.34887289e+02, 7.57876886e+02, 7.81585671e+02,  8.06036141e+02, 8.31251499e+02, 8.57255673e+02, 8.84073340e+02,  9.11729948e+02, 9.40251743e+02, 9.69665789e+02, 1.00000000e+03],
    'y': [
0.06077181,  0.0579915,  0.11417945,  0.17012878,  0.17937351,  0.24445854
,  0.30699244,  0.33443174,  0.38998178,  0.48412251,  0.56275102,  0.62501617
,  0.72167908,  0.78790795,  0.89558951,  0.96062071,  1.02800341,  1.14639025
,  1.2262235,  1.32599907,  1.43104386,  1.52365461,  1.62289524,  1.7172395
,  1.81139895,  1.92614895,  2.0057599,  2.11094511,  2.19653399,  2.2806221
,  2.35775074,  2.43889174,  2.52213353,  2.60186968,  2.68055872,  2.75444057
,  2.83485506,  2.92016596,  2.9983566,  3.08799389,  3.16587606,  3.24404769
,  3.33476784,  3.40113588,  3.51901183,  3.65322939,  3.82903258,  3.98619365
,  4.17712724,  4.36916849,  4.59039337,  4.85871857,  5.08560715,  5.32164995
,  5.59686899,  5.79902445,  6.11969968,  6.39407574,  6.65749423,  6.97289483
,  7.2730559,  7.55652158,  7.83027738,  8.14600255,  8.36852704,  8.65340527
,  8.92824632,  9.20883365,  9.4492276,  9.70038534, 10.00147186, 10.20107306
, 10.47588005, 10.72383975, 10.92715998, 11.15370101, 11.32768646, 11.46227621
, 11.61687947, 11.71458194, 11.78301771, 11.85464581, 11.86127081, 11.89344983
, 11.88437531, 11.86086335, 11.81823204, 11.76991647, 11.73012867, 11.67609299
, 11.63854154, 11.5494538,  11.48349461, 11.37512169, 11.23432867, 11.07182464
, 10.87447941, 10.64959986, 10.43449737, 10.17064481,  9.88773411,  9.60995008
,  9.29685004,  8.9849652,  8.62118315,  8.26161715,  7.93692883,  7.58989111
,  7.2518702,  6.93471884,  6.60755775,  6.30073448,  5.99626959,  5.69822962
,  5.42150111,  5.15880163,  4.87757243,  4.61081119,  4.36787882,  4.11174367
,  3.88021975,  3.66578177,  3.43046418,  3.24383548,  3.03276389,  2.84663028
,  2.67263814,  2.49601337,  2.34402391,  2.19203958,  2.06479983,  1.93724501
,  1.82892919,  1.73727554,  1.66727899,  1.60269327,  1.58375296,  1.4924222
,  1.44641404,  1.43528638,  1.42248484,  1.4733281,  1.51246922,  1.56014009
,  1.59751998,  1.60441809,  1.68549389,  1.59301707,  1.63056197,  1.63202912
,  1.61506571,  1.61859843,  1.6519404,  1.66420066,  1.73797295,  1.83050806
,  1.78900037,  1.85528392,  1.95934229,  1.96549986,  1.93801997,  2.02365993
,  2.00921804,  2.02457233,  1.98968657,  2.01916947,  2.05240519,  2.10693052
,  2.1264259,  2.11047628,  2.14420265,  2.12454746,  2.13807068,  2.14404464
,  2.13487052,  2.16486442,  2.13884491,  2.18301183,  2.16434711,  2.1859763
,  2.21404917,  2.2197846,  2.24855372,  2.26771314,  2.31952906,  2.34987165
,  2.38382369,  2.44248059,  2.49021336,  2.56100837,  2.62776055,  2.69719579
,  2.78898099,  2.88004964,  2.98119939,  3.09789229,  3.22535568,  3.35869897
,  3.50920357,  3.67325617,  3.85021326,  4.04243664,  4.24961945,  4.47275208
,  4.71157358,  4.9660515,  5.23704801,  5.52374436,  5.82609132,  6.14366226
,  6.47499439,  6.81961038,  7.1770264,  7.54427582,  7.92146699,  8.30654567
,  8.69663066,  9.09110565,  9.48760056,  9.88274017, 10.27582436, 10.66415529
, 11.04421143, 11.41530093, 11.77460344, 12.11840936, 12.44622914, 12.75559327
, 13.0429298,  13.3087076,  13.55039319, 13.7657331,  13.95534133, 14.11803135
, 14.25175311, 14.35681205, 14.43032106, 14.46635428, 14.46737205, 14.43376173
, 14.36255203, 14.25857664, 14.12152691, 13.95186037, 13.75229201, 13.52384285
, 13.26714066, 12.98612001, 12.68176855, 12.35516515, 12.01029531, 11.64845878
, 11.2717736,  10.88303526, 10.48483183, 10.07796677,  9.66597864,  9.24946372
,  8.83157074,  8.41430481,  7.99986907,  7.58794176,  7.18226871,  6.78359212
,  6.39274032,  6.01172174,  5.64098642,  5.28151271,  4.9344617,  4.60024994
,  4.27940826,  3.97265911,  3.68011565,  3.40202127,  3.13867752,  2.88987356
,  2.65564895,  2.43582138,  2.22995422,  2.03777769,  1.85889262,  1.69277754
,  1.53890139,  1.39675934,  1.26569987,  1.14515236,  1.03451956,  0.93319695
,  0.84058299,  0.75610825,  0.67919656,  0.60930237,  0.54590877,  0.4885085
,  0.43663215,  0.38982436,  0.34765948,  0.30973717,  0.27568389,  0.24514931
        ],
    'dy': [
        0.00292363, 0.0031838, 0.00489884, 0.00680256, 0.00710848, 0.00963638
        , 0.01120328, 0.01226058, 0.0135255, 0.01552225, 0.0175614, 0.01856515
        , 0.02006286, 0.02181119, 0.02298877, 0.02408795, 0.02549559,
        0.02651751
        , 0.02766893, 0.02844239, 0.0295012, 0.03027189, 0.03116555, 0.03194504
        , 0.03269092, 0.03367749, 0.03434949, 0.03528013, 0.03593248,
        0.03659719
        , 0.0371076, 0.03784324, 0.03843683, 0.03910332, 0.03975979, 0.0402846
        , 0.04087093, 0.04147616, 0.04201591, 0.04266459, 0.04328886,
        0.04390917
        , 0.04478276, 0.04538703, 0.04655051, 0.04789127, 0.04984494,
        0.05171863
        , 0.05409559, 0.05638155, 0.05941745, 0.06299608, 0.06622842, 0.0697099
        , 0.07389259, 0.07743829, 0.08196542, 0.08620538, 0.09025182,
        0.09481254
        , 0.099026, 0.10397198, 0.10800843, 0.1127196, 0.11722062, 0.12160517
        , 0.12556256, 0.12900858, 0.13294331, 0.13658506, 0.13848314,
        0.14234017
        , 0.14370937, 0.14524739, 0.14738211, 0.14754487, 0.14919798,
        0.15002915
        , 0.15081536, 0.15165062, 0.15239438, 0.15311865, 0.15337824,
        0.15404194
        , 0.15437278, 0.1546455, 0.15472203, 0.15474652, 0.15492701, 0.1550033
        , 0.15542319, 0.15503101, 0.15526217, 0.15516391, 0.1545614, 0.15380688
        , 0.1526354, 0.15116853, 0.14982111, 0.1475457, 0.14515792, 0.14272693
        , 0.13956434, 0.13660191, 0.13224506, 0.12807351, 0.12384036,
        0.11948193
        , 0.1149638, 0.11088453, 0.10650898, 0.10249491, 0.09824649, 0.0942151
        , 0.09042241, 0.08658334, 0.08267032, 0.07879726, 0.07527434,
        0.07144381
        , 0.06796415, 0.06469925, 0.06099191, 0.05812655, 0.05479248,
        0.05189333
        , 0.04904663, 0.04606301, 0.04348487, 0.04078102, 0.03852071, 0.0361547
        , 0.03415402, 0.03228791, 0.03073692, 0.02937377, 0.02884899, 0.027112
        , 0.02617004, 0.02598109, 0.02589091, 0.02742092, 0.02894463,
        0.03067563
        , 0.03192199, 0.03248363, 0.03528233, 0.03228249, 0.03335342,
        0.03323314
        , 0.03286766, 0.03322443, 0.03473939, 0.03564231, 0.03792132,
        0.04009277
        , 0.03899218, 0.04014832, 0.04200416, 0.0418143, 0.04030352, 0.04146973
        , 0.0403829, 0.04021187, 0.03882885, 0.03900089, 0.03890874, 0.03881165
        , 0.03867632, 0.0371217, 0.03675703, 0.03527386, 0.03449137, 0.03344281
        , 0.03238561, 0.03222037, 0.03094288, 0.03111477, 0.03006851,
        0.02990611
        , 0.02996257, 0.02983849, 0.03008713, 0.03048987, 0.03154414,
        0.03246946
        , 0.03364781, 0.03529483, 0.03701098, 0.03916057, 0.04148011,
        0.04404408
        , 0.04706103, 0.05030485, 0.05387994, 0.05782989, 0.0620905, 0.06663941
        , 0.07153073, 0.07671587, 0.08219244, 0.08794962, 0.09400171, 0.1002744
        , 0.10682462, 0.11358224, 0.12051324, 0.12754904, 0.13466715,
        0.14182547
        , 0.14894841, 0.15601394, 0.1629741, 0.16975026, 0.17633475, 0.18267982
        , 0.18872135, 0.19445961, 0.19985611, 0.20485952, 0.20948354,
        0.21370364
        , 0.21747644, 0.22095533, 0.22387728, 0.22633611, 0.22837072,
        0.22997603
        , 0.23112287, 0.23186812, 0.23220075, 0.23210887, 0.23164066, 0.2307981
        , 0.22956751, 0.22801228, 0.22613364, 0.22394011, 0.22148589,
        0.21876532
        , 0.21577828, 0.21256649, 0.20913809, 0.20548254, 0.20164827,
        0.19761698
        , 0.19343773, 0.18913298, 0.18470654, 0.18015588, 0.17551214,
        0.17077278
        , 0.1659429, 0.1610207, 0.15605959, 0.1510388, 0.14598113, 0.14087907
        , 0.1357556, 0.13062462, 0.12550144, 0.12036849, 0.11527045, 0.11020606
        , 0.10517937, 0.10021345, 0.09530739, 0.09047272, 0.0857226, 0.08105978
        , 0.07649215, 0.07204021, 0.06768456, 0.0635777, 0.05956206, 0.05537635
        , 0.05157544, 0.04792503, 0.04443907, 0.04112043, 0.03797326,
        0.03499909
        , 0.03219642, 0.02956359, 0.02709634, 0.02479076, 0.02264133, 0.0206409
        , 0.0187841, 0.01706943, 0.01548212, 0.01401883, 0.01267289, 0.01143767
        , 0.01030675, 0.00927356, 0.00833186, 0.00747541, 0.00669815,
        0.00599425           ]
}}

for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]

    # ebl_model = EBL.readascii('outputs/outputs_dust_final_new/'
    #                           +  model['callable_func']
    #                           + '.txt',
    #             model_name=working_model_name)

    data_i = data[working_model_name]

    # plt.loglog(waves_ebl, ebl_model.ebl_array(z=0, lmu=waves_ebl),
    #            c=model['color'], lw=3,
    #            zorder=2/(ni+1),
    #            label=model['label'],#  ls=':'
    #            )


    plt.loglog(data_i['x'], data_i['y'],
               c=model['color'], lw=3,
               zorder=2/(ni+1),
               label=model['label'],#  ls=':'
               )

    plt.fill_between(data_i['x'],
                     np.array(data_i['y']) - np.array(data_i['dy']),
                     np.array(data_i['y']) + np.array(data_i['dy']),
                     facecolor=f_color[working_model_name],
                     alpha=0.3)
    handles_lines.append(plt.Line2D(
        [], [], color=model['color'], lw=3))
    labels_lines.append(model['label'])



ax1.set_xlim(0.1, 1e3)
ax1.set_ylim(0.8, 20)

legend22 = plt.legend(handles_lines, labels_lines,
                      loc=8, fontsize=18,
                      ncol=3)
legend22.set_zorder(0)
ax1.add_artist(legend11)
ax1.add_artist(legend22)

ax1.set_xlabel(r'Wavelength (µm)')

ax1.set_xscale('log')
# ax1.set_yscale('log')
def tick_function(X):
    return (c.h * c.c / X / u.micron).to(u.eV).value
def tick_function_2(X):
    return (c.h * c.c / X / u.eV).to(u.micron).value

aaa = tick_function(2.48)
ax3 = ax1.secondary_xaxis('top',
                         functions=(tick_function, tick_function_2))
ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel('Photon energy (eV)', labelpad=12)


plt.savefig('outputs/figures_paper/cb_manydust.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_manydust.png',
            bbox_inches='tight', dpi=500)

plt.show()


fig, ax = plt.subplots(figsize=(8.75, 7))

z_array = [0., 0.2, 0.5, 1., 2., 5.]
lambda_array = np.geomspace(0.05, 10., num=500)

N = len(z_array)
cividis_mine = mpl.colormaps['cividis']._resample(N)
cividis_mine.colors[-1, 1] = 0.7
cividis_mine.colors[-1, 2] = 0.
cividis_mine.colors = cividis_mine.colors[::-1]
ax.set_prop_cycle(get_cycle(cividis_mine, N))


def spline_starburst(lambda_array, z_value):
    return 10 ** ebl_class.ebl_ssp_spline(
        np.log10(c.c.value * 1e6 / lambda_array), z_value,
                          grid=False)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array(z_array), lambda_array)

for ni, i in enumerate(z_array):
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(lambda_array, spline_starburst(lambda_array, i),
             color=color, linestyle='-', label=i, lw=2)
    plt.plot(lambda_array, nuInu['finke2022'][ni],
             color=color, linestyle='dotted', lw=2)


plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

plt.xlim(0.1, 1e1)
plt.ylim(5e-3, 20)

plt.xscale('log')
plt.yscale('log')

lines = ['-', 'dotted']
legend1 = plt.legend(ncol=3, loc=3,
                      fontsize=18,
                      title_fontsize=20, title='Redshift')
legend2 = plt.legend([plt.Line2D([], [], linestyle=lines[i],
                                 color='k')
                      for i in range(2)],
                     ['Our model', 'Finke22'],
                     loc=2, fontsize=16, framealpha=0.4)

ax.add_artist(legend1)
ax.add_artist(legend2)

plt.savefig('outputs/figures_paper/cb_redshifs.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_redshifs.png',
            bbox_inches='tight')

plt.show()
