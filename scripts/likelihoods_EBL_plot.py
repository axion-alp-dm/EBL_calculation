# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RectBivariateSpline

from scipy.interpolate import UnivariateSpline
from ebl_codes.metall_models import metall_model
from ebl_codes.sfr_models import sfr_model
from ebl_codes.dust_absorption_models import calculate_dust
from ebl_codes.EBL_class import EBL_model

from data.emissivity_measurs.emissivity_read_data import emissivity_data
from data.cb_measurs.import_cb_measurs import import_cb_data
from data.sfr_measurs.sfr_read import *
from data.metallicity_measurs.import_metall import import_met_data

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

from ebltable.ebl_from_model import EBL

all_size = 34
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

# direct_name = str('outputs/outputs_dust_allparamsfree 2025-05-22 08:20:07')
direct_name = ('outputs/outputs_systematics_13perct/')
print(direct_name)

# Configuration file reading and data input/output ---------#
with open(direct_name + '/input_data.yml', 'r') as file:
    config_data = yaml.safe_load(file)

ebl_class = EBL_model.input_yaml_data_into_class(config_data)
# ebl_class.logging_prints = True

waves_ebl = np.logspace(-1, 3, num=300)

colors = {
    'bosa': 'b',
    'chary': 'orange',
    '2bb':'g',
}
f_color = {
    'bosa': 'dodgerblue',
    'chary': 'C1',
    '2bb':'green',
}

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)

# FIGURE: COB fit ------------------------------------------------
fig_ebl = plt.figure(figsize=(10, 7.5))
axes_ebl = fig_ebl.gca()

plt.yscale('log')
plt.xscale('log')

plt.ylim(0.1, 120)
plt.xlim(0.1, 1e3)

handles_cob, labels_cob = [], []

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

# COB measurements that we are going to use
import_cb_data(
    lambda_min_total=0.1, lambda_max_total=1e20,
    plot_measurs=True, ax1=axes_ebl)

# FIGURE: sfr fit ------------------------------------------------
fig_sfr = plt.figure(figsize=(12, 10.5))
axes_sfr = fig_sfr.gca()

x_sfr = np.linspace(0, 10, num=100)

sfr_data = sfr_data_dict()
plot_sfr_data(sfr_data)

handles, labels = axes_sfr.get_legend_handles_labels()
handles = [h[0] for h in handles]

legend1 = plt.legend(handles, labels,
                     title='Measurements', loc=1, fontsize=20,
                     title_fontsize=24, framealpha=0.8)
axes_sfr.add_artist(legend1)

handles_sfr, labels_sfr = [], []

plt.yscale('log')

plt.xlim(0, 10)
plt.ylim(1e-3, 0.5)

plt.xlabel('redshift $z$')
plt.ylabel(r'$\rho_{\star}$ (M$_{\odot}$ / yr / Mpc$^{3}$)')

# FIGURE: Z IN DIFFERENT REDSHIFTS ------------------

fig_Z, ax_met = plt.subplots(figsize=(8, 5))
x_Z = np.geomspace(1e-3, 10, num=100)
x_Z = np.insert(x_Z, 0, 0.)
plt.yscale('log')
aa = import_met_data(ax=ax_met)
plt.xlim(0, 5)
plt.ylim(1e-3, 2.5e-2)

plt.xlabel(r'redshift $z$', fontsize=26)
plt.ylabel(r'Z($z$)', fontsize=26)

plt.xticks(fontsize=26)
plt.yticks(fontsize=26)


# FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS ------------------

list_zz_finke = os.listdir('/home/porrassa/Downloads/lumdens/')

zz_bare = []
for i in list_zz_finke:
    i = i.replace('.dat', '')
    i = i.replace('lumdens_total_z', '')
    zz_bare.append(i)

zz_bare = np.array(zz_bare, dtype=str)
zz_floats = np.array(zz_bare, dtype=float)

zz_order = np.argsort(zz_floats)
zz_floats = zz_floats[zz_order]
zz_bare = zz_bare[zz_order]

wavelengths = np.loadtxt(
    '/home/porrassa/Downloads/lumdens/lumdens_total_z0.00.dat')
wavelengths = wavelengths[:, 0]

array_lumin = np.zeros((len(wavelengths), len(zz_bare)))

for ni, ii in enumerate(zz_bare):
    data = np.loadtxt(
    '/home/porrassa/Downloads/lumdens/lumdens_total_z' + ii + '.dat')
    array_lumin[:, ni] = data[:, 1]

spline_emiss_finke = RectBivariateSpline(
    x=np.log10(wavelengths), y=zz_floats, z=array_lumin,
    kx=1, ky=1, s=0)


fig_emiss_z, axes_emiss_z = plt.subplots(4, 3, figsize=(12, 16))

z_array = np.linspace(0, 10)

for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                               0.44, 0.55, 0.79,
                               1.22, 2.2, 3.6,
                               4.5, 5.8, 8.0]):
    plt.subplot(4, 3, n_lambda + 1)
    emissivity_data(z_min=None, z_max=None,
                    lambda_min=ll - 0.01, lambda_max=ll + 0.01,
                    take1ref=None, plot_fig=True)

    plt.plot(z_array, spline_emiss_finke(np.log10(ll), z_array, grid=False),
             ls='--', c='k')

    plt.annotate(r'%r$\,$µm' % ll, xy=(5, 1e35), fontsize=28)

    plt.xlim(min(z_array), max(z_array))
    plt.ylim(1e33, 3e35)

    plt.yscale('log')

fig_emiss_z.subplots_adjust(wspace=0, hspace=0)

handles_emiss, labels_emiss = [], []

plt.subplot(4, 3, 11)
plt.xlabel(r'redshift $z$', fontsize=34)

plt.subplot(4, 3, 4)
# plt.ylabel(r'$_{\nu} \varepsilon_{_{\nu} \,\,(\mathrm{W\, / \, Mpc}^3)}$',
#            fontsize=40)
plt.text(-0.45, 0.,
         r'$_{\nu} \varepsilon_{_{\nu} \,\,(\mathrm{W\, / \, Mpc}^3)}$',
         horizontalalignment='center',
         verticalalignment='center', transform=axes_emiss_z[1, 0].transAxes,
         rotation=90, fontsize=40)
# plt.subplot(3, 3, 9)
# plt.annotate(r'3.6$\,\mu m$', xy=(6, 1e34), fontsize=28)

ax = [plt.subplot(4, 3, i) for i in [2, 3, 5, 6, 8, 9, 11, 12]]
for a in ax:
    a.set_yticklabels([])

ax = [plt.subplot(4, 3, i + 1) for i in range(9)]
for a in ax:
    a.set_xticklabels([])

ax = [plt.subplot(4, 3, i + 1) for i in range(9, 12)]
for a in ax:
    a.set_xticks([0, 2, 4, 6, 8])

ax = [plt.subplot(4, 3, i) for i in range(1, 10)]
for a in ax:
    a.set_xticks([0, 2, 4, 6, 8, 10])

a = plt.subplot(4, 3, 12)
a.set_xticks([0, 2, 4, 6, 8, 10])

emiss_data = emissivity_data()
freq_emiss = c.value / (emiss_data['lambda'] * 1e-6)
# plt.show()

for nkey, key in enumerate(config_data['ssp_models']):

    if key == 'bosa':
        values_sfr = np.concatenate((
            config_data['ssp_models'][key]['sfr_params'],
            config_data['ssp_models'][key]['metall_params']))

    elif key == 'chary':
        values_sfr = np.concatenate((
            config_data['ssp_models'][key]['sfr_params'],
            config_data['ssp_models'][key]['metall_params'],
            [config_data['ssp_models'][key]['dust_reem_params']['f_tir']],
            [config_data['ssp_models'][key]['dust_reem_params']['wv_reem_min']],
        ))

    elif key == '2bb':
        values_sfr = np.concatenate((
            config_data['ssp_models'][key]['sfr_params'],
            config_data['ssp_models'][key]['metall_params'],
            config_data['ssp_models'][key]['dust_reem_params']['T'],
            [config_data['ssp_models'][key]['dust_reem_params']['fracts']]
        ))

    print(values_sfr)
    values_cov = config_data['ssp_models'][key]['cov_matrix']
    values_cov = np.array(values_cov).reshape(
        int(np.sqrt(np.shape(values_cov))),
        int(np.sqrt(np.shape(values_cov))))

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])


    def fit_igl(lambda_igl, params):
        config_data['ssp_models'][key]['sfr_params'] = params[0:4].copy()
        config_data['ssp_models'][key]['metall_params'] = params[4:8].copy()

        if key == 'chary':
            config_data['ssp_models'][key]['dust_reem_params']['f_tir'] = \
                params[8]
            config_data['ssp_models'][key]['dust_reem_params']['wv_reem_min'] = \
                params[9]

        if key == '2bb':
            config_data['ssp_models'][key]['dust_reem_params']['T'] = \
                params[8:10].copy()
            config_data['ssp_models'][key]['dust_reem_params']['fracts'] = \
                params[10]

        return ebl_class.ebl_ssp_individualData(
            yaml_data=config_data['ssp_models'][key],
            x_data=lambda_igl)


    def fit_emiss(x_all, params):
        lambda_emiss, z_emiss = x_all
        freq_emissions = (c.value / lambda_emiss * 1e6)

        config_data['ssp_models'][key]['sfr_params'] = params[0:4].copy()
        config_data['ssp_models'][key]['metall_params'] = params[4:8].copy()

        if key == 'chary':
            config_data['ssp_models'][key]['dust_reem_params']['f_tir'] = \
                params[8]
            config_data['ssp_models'][key]['dust_reem_params']['wv_reem_min'] = \
                params[9]

        if key == '2bb':
            config_data['ssp_models'][key]['dust_reem_params']['T'] = \
                params[8:10].copy()
            config_data['ssp_models'][key]['dust_reem_params']['fracts'] = \
                params[10]


        ebl_class.emiss_ssp_calculation(config_data['ssp_models'][key])

        return (ebl_class.emiss_ssp_spline(
                     lambda_emiss, z_emiss) * freq_emissions * 1e-7)


    def sfr(x, params):
        return sfr_model(
            zz_array=x,
            sfr_model=config_data['ssp_models'][key]['sfr_formula'],
            sfr_params=params[0:4])


    def metall(x, params):
        return metall_model(
            zz_array=x,
            metall_model=config_data['ssp_models'][key]['metall_formula'],
            metall_params=params[4:8])

    # FIGURE: cob fit
    axes_ebl.plot(waves_ebl,
                  ebl_class.ebl_ssp_spline(waves_ebl, 0.),
                  color=colors[key], lw=2)

    labels_cob.append(config_data['ssp_models'][key]['name'])
    handles_cob.append(plt.Line2D([], [], linewidth=2,
                                  linestyle='-',
                                  color=colors[key]))

    y, y_cov = propagate(lambda pars:
                         fit_igl(waves_ebl, pars),
                         values_sfr, values_cov)
    yerr_prop = np.diag(y_cov) ** 0.5
    axes_ebl.fill_between(waves_ebl, y - yerr_prop, y + yerr_prop,
                          facecolor=f_color[key], alpha=0.3)
    print(waves_ebl)
    print(y)
    print(yerr_prop)
    print(key, 'cb')

    # FIGURE: SFR
    plt.figure(fig_sfr)
    axes_sfr.plot(x_sfr, sfr(x_sfr, values_sfr), '-',
                  color=colors[key], lw=2)

    labels_sfr.append(config_data['ssp_models'][key]['name'])
    handles_sfr.append(plt.Line2D([], [], linewidth=3,
                                  linestyle='-',
                                  color=colors[key]))

    y, y_cov = propagate(lambda pars:
                         sfr(x_sfr, pars),
                         values_sfr, values_cov)
    yerr_prop = np.diag(y_cov) ** 0.5
    plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
                     facecolor=f_color[key], alpha=0.3)
    print(key, 'sfr')

    # Fig Z
    plt.figure(fig_Z)
    plt.plot(x_Z, metall(x_Z, params=values_sfr),
             color=colors[key],
             label=config_data['ssp_models'][key]['name'])

    y, y_cov = propagate(lambda pars:
                         metall(x_Z, pars),
                         values_sfr, values_cov)
    yerr_prop = np.diag(y_cov) ** 0.5
    plt.fill_between(x_Z, y - yerr_prop, y + yerr_prop,
                     facecolor=f_color[key], alpha=0.3)
    print(key, 'Z')


    # FIGURE: emissivities fit
    plt.figure(fig_emiss_z)
    for n_lambda, ll in enumerate([0.15, 0.17, 0.28,
                                   0.44, 0.55, 0.79,
                                   1.22, 2.2, 3.6,
                                   4.5, 5.8, 8.0]):
        plt.subplot(4, 3, n_lambda + 1)

        plt.plot(z_array,
                 (c.value / (ll * 1e-6)
                  * ebl_class.emiss_ssp_spline(
                     ll * np.ones(len(z_array)),
                     z_array)
                  * 1e-7),
                 linestyle='-', color=colors[key], lw=2)

    labels_emiss.append(config_data['ssp_models'][key]['name'])
    handles_emiss.append(plt.Line2D([], [], linewidth=2,
                                    linestyle='-',
                                    color=colors[key]))

    y, y_cov = propagate(lambda pars:
                         fit_emiss((ll * np.ones(len(z_array)), z_array),
                                   pars),
                         values_sfr, values_cov)
    yerr_prop = np.diag(y_cov) ** 0.5
    plt.fill_between(z_array, y - yerr_prop, y + yerr_prop,
                     facecolor=f_color[key], alpha=0.3)

    print(key, 'emiss')

# -------------------------------------------------------------
plt.figure(fig_ebl)

plt.plot(waves_ebl, nuInu['finke2022'], ls='--', color='magenta', lw=2.)
labels_cob.append('Finke22')
handles_cob.append(plt.Line2D([], [], linewidth=2,
                              linestyle='-',
                              color='magenta'))

plt.plot(waves_ebl, nuInu['cuba'], ls='dotted', color='k', lw=2.)
labels_cob.append('CUBA')
handles_cob.append(plt.Line2D([], [], linewidth=2,
                              linestyle='-',
                              color='k'))

plt.legend(handles_cob, labels_cob,
           loc=8,
           title=r'Models')

plt.figure(fig_Z)
plt.legend(handles_sfr, labels_sfr,
           title='Models', loc=1,# bbox_to_anchor=(0.01, 0.),
                     fontsize=20, title_fontsize=22,
           ncol=1)

plt.figure(fig_sfr)

axes_sfr.plot(x_sfr, sfr_model(
    zz_array=x_sfr, sfr_model='sfr_madau14'),
              color='k', linestyle='--', lw=2)

labels_sfr.append('MD14')
handles_sfr.append(plt.Line2D([], [], linewidth=2,
                              linestyle='--',
                              color='k'))

legend2 = plt.legend(handles_sfr, labels_sfr,
                     loc=3, bbox_to_anchor=(0.01, 0.),
                     fontsize=26, title='Models', title_fontsize=28
                     )
axes_sfr.add_artist(legend2)


plt.figure(fig_emiss_z)
plt.subplot(4, 3, 2)
plt.legend(handles_emiss, labels_emiss,
           loc=8, fontsize=28, bbox_to_anchor=(0.5, 1.01),
           ncol=3)

# Save the figures
fig_ebl.savefig(direct_name + '/ebl' + '.png',
                bbox_inches='tight')
fig_ebl.savefig(direct_name + '/ebl' + '.pdf',
                bbox_inches='tight')

fig_sfr.savefig(direct_name + '/sfr' + '.png',
                bbox_inches='tight', dpi=500)
fig_sfr.savefig(direct_name + '/sfr' + '.pdf',
                bbox_inches='tight')

fig_Z.savefig(direct_name + '/Zev' + '.png',
                bbox_inches='tight')
fig_Z.savefig(direct_name + '/Zev' + '.pdf',
                bbox_inches='tight')

fig_emiss_z.subplots_adjust(wspace=0, hspace=0)
fig_emiss_z.savefig(
    direct_name + '/emiss_redshift' + '.png',
    bbox_inches='tight')
fig_emiss_z.savefig(
    direct_name + '/emiss_redshift' + '.pdf',
    bbox_inches='tight')
plt.show()

