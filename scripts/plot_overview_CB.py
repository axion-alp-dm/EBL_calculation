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
plt.show()
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
for ni, working_model_name in enumerate(list_working_models.keys()):
    model = list_working_models[working_model_name]

    ebl_model = EBL.readascii('outputs/outputs_dust_final_new/'
                              +  model['callable_func']
                              + '.txt',
                model_name=working_model_name)

    plt.loglog(waves_ebl, ebl_model.ebl_array(z=0, lmu=waves_ebl),
               c=model['color'], lw=3,
               zorder=2/(ni+1),
               label=model['label'],#  ls=':'
               )
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
