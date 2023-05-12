# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from emissivity_data.emissivity_read_data import emissivity_data
from ebl_measurements.EBL_measurs_plot import plot_ebl_measurement_collection
from ebl_measurements.read_ebl_biteau import dictionary_datatype
from sfr_data.sfr_read import *

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

from ebltable.ebl_from_model import EBL

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
plt.rc('legend', fontsize=all_size)
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

direct_name = str('chi2_with_emissivities' +
                  time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime()))
if not os.path.exists("outputs/"):
    # if the directory for outputs is not present, create it.
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    # if the directory for outputs is not present, create it.
    os.makedirs('outputs/' + direct_name)


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


def input_yaml_data_into_class(yaml_data, log_prints=False):
    z_array = np.linspace(float(yaml_data['redshift_array']['zmin']),
                          float(yaml_data['redshift_array']['zmax']),
                          yaml_data['redshift_array']['zsteps'])

    lamb_array = np.logspace(np.log10(float(
        yaml_data['wavelenght_array']['lmin'])),
        np.log10(float(
            yaml_data['wavelenght_array']['lmax'])),
        yaml_data['wavelenght_array']['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     h=float(yaml_data['cosmology_params']['cosmo'][0]),
                     omegaM=float(
                         yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(
                         yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'],
                     log_prints=log_prints)


def chi2_upperlims(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


def chi2_measurs(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2.)


def gamma_from_rest(mass, gay):
    return (((mass * u.eV) ** 3. * (gay * u.GeV ** -1) ** 2.
             / 32. / h_plank.to(u.eV * u.s)).to(u.s ** -1).value)


config_data = read_config_file('scripts/input_data_iminuit_test.yml')
ebl_class = input_yaml_data_into_class(config_data)

axion_mac2 = np.logspace(-1, 4, num=100)
axion_gay = np.logspace(-12, -8, num=70)

igl_ebldata = dictionary_datatype(
    'ebl_measurements/optical_data_2023', 'IGL', lambda_max=5.)

upper_lims_ebldata = dictionary_datatype(
    'ebl_measurements/optical_data_2023', 'UL', lambda_max=5.)

upper_lims_ebldata_woNH = dictionary_datatype(
    'ebl_measurements/optical_data_2023', 'UL', lambda_max=5.,
    obs_not_taken=['lauer2022.ecsv'])

emiss_data = emissivity_data(directory='emissivity_data/')
freq_emiss = 3e8 / (emiss_data['lambda'] * 1e-6)

sfr_data = sfr_data_dict('sfr_data/')

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))

xx_plot = np.logspace(-1, 1, num=100)
xx_plot_freq = np.log10(c.value / np.array(xx_plot) * 1e6)

colors = ['b', 'r', 'g', 'purple', 'k']
models = ['solid', 'dashed', 'dotted']

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)

spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)

fig_ebl = plt.figure(figsize=(11, 11))
axes_ebl = fig_ebl.gca()

plt.plot(waves_ebl, nuInu['finke2022'], ls='--', color='k', lw=2.,
         label='Finke22')
ebl_class.change_axion_contribution(
            1e3, gamma_from_rest(1e3, 3e-11))
print(gamma_from_rest(1e3, 3e-11))
plt.plot(waves_ebl,
         10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0., grid=False),
         linestyle=models[2], color='k')
plt.plot(waves_ebl,
         10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0., grid=False)
         + spline_finke(waves_ebl),
         linestyle=models[0], color='k')

plt.yscale('log')
plt.xscale('log')

plt.ylim(0.1, 120)
# plt.xlim(0.1, 20.)

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ sr)')

dictionary_datatype('ebl_measurements/optical_data_2023', 'UL',
                    plot_measurs=True, lambda_max=20.)
dictionary_datatype('ebl_measurements/optical_data_2023', 'IGL',
                    plot_measurs=True, lambda_max=20.)

values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))
values_gay_array_NH = np.zeros((len(axion_mac2), len(axion_gay)))

for na, aa in enumerate(axion_mac2):
    for nb, bb in enumerate(axion_gay):
        ebl_class.change_axion_contribution(
            aa, gamma_from_rest(aa, bb))

        values_gay_array[na, nb] += 2. * chi2_upperlims(
            x_model=(10 ** ebl_class.ebl_axion_spline(
                np.log10(3e8 / (upper_lims_ebldata['lambda'] * 1e-6)),
                0.,
                grid=False)
                     + spline_finke(upper_lims_ebldata['lambda'])),
            x_obs=upper_lims_ebldata['nuInu'],
            err_obs=upper_lims_ebldata['nuInu_errp'])

        # values_gay_array_NH[na, nb] += 2. * chi2_upperlims(
        #     x_model=10 ** ebl_class.ebl_total_spline(
        #         np.log10(3e8 / (upper_lims_ebldata_woNH['lambda']
        #                         * 1e-6)),
        #         0.,
        #         grid=False),
        #     x_obs=upper_lims_ebldata_woNH['nuInu'],
        #     err_obs=(upper_lims_ebldata_woNH['nuInu_errn'] +
        #              upper_lims_ebldata_woNH['nuInu_errp']) / 2.)

        values_gay_array_NH[na, nb] += (
                ((16.7 - 10 ** ebl_class.ebl_total_spline(
                            np.log10(3e8 / (0.608 * 1e-6)),
                            0.,
                            grid=False)) / 1.47) ** 2.)

fig_params = plt.figure(figsize=(12, 10))
axes_params = fig_params.gca()
plt.title('Finke 2022 model A')

plt.xscale('log')
plt.yscale('log')

bbb = plt.pcolor(axion_mac2, axion_gay,
                 (values_gay_array.T - np.min(values_gay_array)),
                 vmin=0., vmax=100., rasterized=True
                 )
aaa = plt.contour(axion_mac2, axion_gay,
                  (values_gay_array.T - np.min(values_gay_array)),
                  levels=[2.30, 5.99],
                  origin='lower',
                  colors=('r', 'cyan'))
plt.clabel(aaa, inline=True, fontsize=16, levels=[2.30, 5.99],
           fmt={2.30: r'69%', 5.99: r'95%'})
cbar = plt.colorbar(bbb)
cbar.set_label(r'$\Delta\chi^2_{total}$')

plt.xlabel(r'm$_a\,$c$^2$ [eV]')
plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')

plt.savefig('outputs/' + direct_name + '/'
            + 'z_Finke22' + '_param_space' + '.png')
plt.savefig('outputs/' + direct_name + '/'
            + 'z_Finke22' + '_param_space' + '.pdf')

# FIGURE: AXION PARAMETER SPACE FOR THE DIFFERENT AXION PARAMETERS WO NH
plt.figure(figsize=(12, 10))
plt.title('Finke 2022 model A')

plt.xscale('log')
plt.yscale('log')

bbb = plt.pcolor(axion_mac2, axion_gay,
                 (values_gay_array_NH.T - np.min(values_gay_array_NH)),
                 vmin=0., vmax=100., rasterized=True
                 )
aaa = plt.contour(axion_mac2, axion_gay,
                  (values_gay_array_NH.T - np.min(values_gay_array_NH)),
                  levels=[2.30, 5.99],
                  origin='lower',
                  colors=('r', 'cyan'))
plt.clabel(aaa, inline=True, fontsize=16, levels=[2.30, 5.99],
           fmt={2.30: r'69%', 5.99: r'95%'})
cbar = plt.colorbar(bbb)
cbar.set_label(r'$\Delta\chi^2_{total}$')

plt.xlabel(r'm$_a\,$c$^2$ [eV]')
plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')

plt.savefig('outputs/' + direct_name + '/'
            + 'z_Finke22' + '_param_space_NH' + '.png')
plt.savefig('outputs/' + direct_name + '/'
            + 'z_Finke22' + '_param_space_NH' + '.pdf')

# MINIMIZATION OF CHI2 OF SSPs
for nkey, key in enumerate(config_data['ssp_models']):

    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])


    def fit_igl(lambda_igl, params):
        config_data['ssp_models'][key]['sfr_params'] = params
        return ebl_class.ebl_ssp_individualData(
            yaml_data=config_data['ssp_models'][key],
            x_data=lambda_igl)


    def fit_emiss(x_all, params):
        lambda_emiss, z_emiss = x_all
        freq_emissions = np.log10(3e8 / lambda_emiss * 1e-6)

        config_data['ssp_models'][key]['sfr_params'] = params

        return 10 ** (freq_emissions
                      + ebl_class.emiss_ssp_spline(freq_emissions, z_emiss)
                      - 7)


    def sfr(x, params):
        return eval(config_data['ssp_models'][key]['sfr'])(x, params)


    combined_likelihood = (LeastSquares(igl_ebldata['lambda'],
                                        igl_ebldata['nuInu'],
                                        (igl_ebldata['nuInu_errn']
                                         + igl_ebldata['nuInu_errp']) / 2.,
                                        fit_igl)
                           + LeastSquares((emiss_data['lambda'],
                                           emiss_data['z']),
                                          emiss_data['eje'],
                                          (emiss_data['eje_n']
                                           + emiss_data['eje_p']) / 2.,
                                          fit_emiss)
                           + LeastSquares(sfr_data[:, 0],
                                          sfr_data[:, 3],
                                          (sfr_data[:, 4]
                                           + sfr_data[:, 5]) / 2.,
                                          sfr))

    init_time = time.process_time()

    m = Minuit(combined_likelihood, ([0.015, 2.57, 3, 6.]))
    m.limits = [[0.005, 0.019], [2., 3.5], [1., 3.5], [4., 8.]]

    m.migrad()  # finds minimum of least_squares function
    m.hesse()  # accurately computes uncertainties

    outputs_file = open('outputs/' + direct_name + '/z_fits_info', 'a+')
    outputs_file.write(str(key) + '\n')
    outputs_file.write('SSP model: '
                       + str(config_data['ssp_models'][key]['name'])
                       + '\n')
    outputs_file.write(str(m.params) + '\n')
    outputs_file.write(str(m.values) + '\n')
    outputs_file.write(str(m.covariance) + '\n')
    outputs_file.write('\n\n\n')
    outputs_file.close()

    print(m.params)
    print(m.values)
    print(m.covariance)

    print('Fit: %.2fs' % (time.process_time() - init_time))
    init_time = time.process_time()

    config_data['ssp_models'][key]['sfr_params'] = [m.params[0].value,
                                                    m.params[1].value,
                                                    m.params[2].value,
                                                    m.params[3].value]
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

    values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))
    values_gay_array_NH = np.zeros((len(axion_mac2), len(axion_gay)))

    for na, aa in enumerate(axion_mac2):
        for nb, bb in enumerate(axion_gay):
            ebl_class.change_axion_contribution(
                aa, gamma_from_rest(aa, bb))

            values_gay_array[na, nb] += 2. * chi2_upperlims(
                x_model=10 ** ebl_class.ebl_total_spline(
                    np.log10(3e8 / (upper_lims_ebldata['lambda'] * 1e-6)),
                    0.,
                    grid=False),
                x_obs=upper_lims_ebldata['nuInu'],
                err_obs=(upper_lims_ebldata['nuInu_errn'] +
                         upper_lims_ebldata['nuInu_errp']) / 2.)

            # values_gay_array_NH[na, nb] += 2. * chi2_upperlims(
            #     x_model=10 ** ebl_class.ebl_total_spline(
            #         np.log10(3e8 / (upper_lims_ebldata_woNH['lambda']
            #                         * 1e-6)),
            #         0.,
            #         grid=False),
            #     x_obs=upper_lims_ebldata_woNH['nuInu'],
            #     err_obs=(upper_lims_ebldata_woNH['nuInu_errn'] +
            #              upper_lims_ebldata_woNH['nuInu_errp']) / 2.)

            values_gay_array_NH[na, nb] += (
                ((16.7 - 10 ** ebl_class.ebl_total_spline(
                            np.log10(3e8 / (0.608 * 1e-6)),
                            0.,
                            grid=False))
                 / 1.47) ** 2.)

    print('Axion param space: %.2fs' % (time.process_time() - init_time))

    # -------------------------------------------------------------------
    # FIGURE: EBL FIT
    fig_ebl = plt.figure(figsize=(11, 11))
    axes_ebl = fig_ebl.gca()

    plt.title(config_data['ssp_models'][key]['name'])

    plt.plot(waves_ebl, nuInu['finke2022'], ls='--', color='k', lw=2.,
             label='Finke22')

    plt.plot(xx_plot,
             10 ** ebl_class.ebl_ssp_spline(xx_plot_freq, 0., grid=False),
             'r',
             label="fit")

    # y, y_cov = propagate(lambda pars:
    #                      fit_igl(xx_plot, pars),
    #                      m.values, m.covariance)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # plt.fill_between(xx_plot, y - yerr_prop, y + yerr_prop,
    #                  facecolor="C1", alpha=0.5)

    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} /"f" {m.ndof}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

    plt.legend(title="\n".join(fit_info))

    plt.yscale('log')
    plt.xscale('log')

    plt.ylim(0.1, 120)
    plt.xlim(0.1, 20.)

    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ sr)')

    dictionary_datatype('ebl_measurements/optical_data_2023', 'UL',
                        plot_measurs=True, lambda_max=20.)
    dictionary_datatype('ebl_measurements/optical_data_2023', 'IGL',
                        plot_measurs=True, lambda_max=20.)

    plt.savefig('outputs/' + direct_name + '/' + key + '_ebl' + '.png')
    plt.savefig('outputs/' + direct_name + '/' + key + '_ebl' + '.pdf')

    # FIGURE: SFR
    plt.figure(figsize=(12, 8))
    plt.title(config_data['ssp_models'][key]['name'])

    plot_sfr_data(sfr_data)

    x_sfr = np.linspace(0, 10)

    plt.plot(x_sfr, sfr(x_sfr, [0.015, 2.7, 2.9, 5.6]),
             color='b', label='MD14')
    plt.plot(x_sfr, sfr(x_sfr, [0.0092, 2.79, 3.10, 6.97]),
             color='g', label='MF17')

    m2 = [m.params[0].value, m.params[1].value,
          m.params[2].value, m.params[3].value]
    plt.plot(x_sfr, sfr(x_sfr, m2), '-r', label='fit')

    # y, y_cov = propagate(lambda pars:
    #                      sfr(x_sfr, pars),
    #                      m.values, m.covariance)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
    #                  facecolor="C1", alpha=0.5)

    plt.yscale('log')
    plt.xlabel('redshift z')
    plt.ylabel(r'SFR [M$_{\odot}$ yr$^{-1}$Mpc$^{-3}$]')
    plt.legend(fontsize=16)

    plt.savefig('outputs/' + direct_name + '/'
                + key + '_sfr' + '.png')
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_sfr' + '.pdf')
    # ------------------------------------------------------------------------

    if config_data['ssp_models'][key]['dust_abs_models'] == ['finke2022_2']:
        linestyle = models[0]
    elif config_data['ssp_models'][key]['dust_abs_models'] == ['finke2022']:
        linestyle = models[1]
    else:
        linestyle = models[2]

    if config_data['ssp_models'][key]['file_name'] \
            == '0.0001':
        color = colors[0]
    elif config_data['ssp_models'][key]['file_name'] \
            == '0.008':
        color = colors[1]
    elif config_data['ssp_models'][key]['file_name'] \
            == '0.004':
        color = colors[2]
    elif config_data['ssp_models'][key]['file_name'] \
            == '0.02':
        color = colors[3]
    else:  # SB99
        color = colors[4]

    # FIGURE: AXION PARAMETER SPACE FOR THE DIFFERENT AXION PARAMETERS
    fig_params = plt.figure(figsize=(12, 10))
    plt.title(config_data['ssp_models'][key]['name'])

    plt.xscale('log')
    plt.yscale('log')

    bbb = plt.pcolor(axion_mac2, axion_gay,
                     (values_gay_array.T - np.min(values_gay_array)),
                     vmin=0., vmax=100., rasterized=True
                     )
    aaa = plt.contour(axion_mac2, axion_gay,
                      (values_gay_array.T - np.min(values_gay_array)),
                      levels=[2.30, 5.99],
                      origin='lower',
                      colors=('r', 'cyan'))
    plt.clabel(aaa, inline=True, fontsize=16, levels=[2.30, 5.99],
               fmt={2.30: r'69%', 5.99: r'95%'})
    cbar = plt.colorbar(bbb)
    cbar.set_label(r'$\Delta\chi^2_{total}$')

    plt.xlabel(r'm$_a\,$c$^2$ [eV]')
    plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')

    plt.savefig('outputs/' + direct_name + '/'
                + key + '_param_space' + '.png')
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_param_space' + '.pdf')

    # FIGURE: AXION PARAMETER SPACE FOR THE DIFFERENT AXION PARAMETERS WO NH
    plt.figure(figsize=(12, 10))
    plt.title(config_data['ssp_models'][key]['name'])

    plt.xscale('log')
    plt.yscale('log')

    bbb = plt.pcolor(axion_mac2, axion_gay,
                     (values_gay_array_NH.T - np.min(values_gay_array_NH)),
                     vmin=0., vmax=100., rasterized=True
                     )
    aaa = plt.contour(axion_mac2, axion_gay,
                      (values_gay_array_NH.T - np.min(values_gay_array_NH)),
                      levels=[2.30, 5.99],
                      origin='lower',
                      colors=('r', 'cyan'))
    plt.clabel(aaa, inline=True, fontsize=16, levels=[2.30, 5.99],
               fmt={2.30: r'69%', 5.99: r'95%'})
    cbar = plt.colorbar(bbb)
    cbar.set_label(r'$\Delta\chi^2_{total}$')

    plt.xlabel(r'm$_a\,$c$^2$ [eV]')
    plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')

    plt.savefig('outputs/' + direct_name + '/'
                + key + '_param_space_NH' + '.png')
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_param_space_NH' + '.pdf')

    # FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS
    plt.subplots(4, 3, figsize=(15, 15))

    z_array = np.linspace(0, 10)

    for n_lambda, ll in enumerate([0.15, 0.17, 0.28, 0.44, 0.55, 0.79,
                                   1.22, 2.2, 3.6, 4.5, 5.8, 8.]):
        plt.subplot(4, 3, n_lambda + 1)
        emissivity_data(directory='emissivity_data/', z_min=None, z_max=None,
                        lambda_min=ll - 0.05, lambda_max=ll + 0.05,
                        take1ref=None, plot_fig=True)

        plt.plot(z_array,
                 (3e8 / (ll * 1e-6))
                 * 10 ** ebl_class.emiss_ssp_spline(
                     np.log10(3e8 / (ll * 1e-6)) * np.ones(len(z_array)),
                     z_array)
                 * 1e-7,
                 linestyle='-', color='k')

        # y, y_cov = propagate(lambda pars:
        #                      fit_emiss((ll * np.ones(len(z_array)), z_array),
        #                                pars),
        #                      m.values, m.covariance)
        # yerr_prop = np.diag(y_cov) ** 0.5
        # plt.fill_between(z_array, y - yerr_prop, y + yerr_prop,
        #                  facecolor="C1", alpha=0.5)

        plt.annotate(r'%r$\,\mu m$' % ll, xy=(7, 3e34))

        plt.xlim(min(z_array), max(z_array))
        plt.ylim(1e33, 3e35)

        plt.yscale('log')

    plt.subplot(4, 3, 11)
    plt.xlabel(r'redshift z')

    plt.subplot(4, 3, 7)
    plt.ylabel(r'$\nu \mathrm{L}_{\nu}$ (W / Mpc$^3$)')

    plt.subplot(4, 3, 2)
    plt.title(config_data['ssp_models'][key]['name'])

    ax = [plt.subplot(4, 3, i) for i in [2, 3, 5, 6, 8, 9, 11, 12]]
    for a in ax:
        a.set_yticklabels([])

    ax = [plt.subplot(4, 3, i + 1) for i in range(9)]
    for a in ax:
        a.set_xticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_emiss_redshift' + '.png')
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_emiss_redshift' + '.pdf')

    # EMISSIVIY AT Z=0 FIGURE
    fig_emiss = plt.figure(figsize=(12, 8))
    axes_emiss = fig_emiss.gca()

    plt.title(config_data['ssp_models'][key]['name'])
    data_emiss = emissivity_data(directory='emissivity_data/', z_max=0.1)

    plt.errorbar(x=data_emiss['lambda'], y=data_emiss['eje'],
                 yerr=(data_emiss['eje_n'], data_emiss['eje_p']),
                 linestyle='', marker='o')

    plt.plot(waves_ebl,
             10 ** (freq_array_ebl
                    + ebl_class.emiss_ssp_spline(
                         freq_array_ebl, np.zeros(len(waves_ebl)))
                    - 7),
             linestyle=linestyle, color=color)

    # y, y_cov = propagate(lambda pars:
    #                      fit_emiss((xx_plot, np.zeros(len(xx_plot))), pars),
    #                      m.values, m.covariance)
    # yerr_prop = np.diag(y_cov) ** 0.5
    # plt.fill_between(xx_plot, y - yerr_prop, y + yerr_prop,
    #                  facecolor="C1", alpha=0.5)

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(r'$\nu \mathrm{L}_{\nu}$ (W / Mpc$^3$)')

    plt.xlim(0.09, 10)
    plt.ylim(1e33, 1e35)

    legend22 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle='-',
                                      color=colors[i])
                           for i in range(5)],
                          ['Pegase 0.0001', 'Pegase 0.008', 'Pegase 0.004',
                           'Pegase 0.02', 'StarBurst99'],
                          loc=8,
                          title=r'Population')
    legend33 = plt.legend([plt.Line2D([], [], linewidth=2, linestyle=models[i],
                                      color='k') for i in range(1, 3)],
                          ['Finke model A', 'Kneiske02'],
                          title=r'Dust absorption model',
                          # bbox_to_anchor=(1.04, 0.1),
                          loc=1)
    axes_emiss.add_artist(legend22)
    axes_emiss.add_artist(legend33)

    # plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_luminosities' + '.png')
    plt.savefig('outputs/' + direct_name + '/'
                + key + '_luminosities' + '.pdf')

    # plt.close('all')
    plt.show()

plt.show()
