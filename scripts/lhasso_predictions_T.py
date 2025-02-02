import os
import yaml
import numpy as np
import scipy.optimize
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import astropy.units as u

from e_disp_code import EDispGauss

from iminuit import Minuit
from iminuit.cost import LeastSquares

from ebltable.ebl_from_model import EBL
from ebltable.tau_from_model import OptDepth

from data.cb_measurs.import_cb_measurs import import_cb_data

from ebl_codes.EBL_class import EBL_model

if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


def likelihood_poisson(mu_i_array_obs, mu_i_array_asimov):
    return sum(mu_i_array_asimov * np.log(mu_i_array_obs)
               - mu_i_array_obs)

# ----------------------------------------------------------------------
fig_spectra = plt.figure()
mkr501_flux_1 = np.loadtxt(
    'data/lhasso_characteristics/mkr501_flare1997_paper1.txt')
mkr501_flux_1[:, 1] = (mkr501_flux_1[:, 1]
                       * mkr501_flux_1[:, 0] ** 2. * (u.TeV.to(u.erg))
                       * 1e12)
mkr501_flux_1[:, 2] = (mkr501_flux_1[:, 2]
                       * mkr501_flux_1[:, 0] ** 2. * (u.TeV.to(u.erg))
                       * 1e12)
mkr501_flux = np.loadtxt(
    'data/lhasso_characteristics/mkr501_flare1997_table_reanalysis.txt')
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='o', label='Reanalysis')
plt.errorbar(mkr501_flux_1[:, 0], mkr501_flux_1[:, 1],
             yerr=mkr501_flux_1[:, 2],
             ls='', marker='o', label='Original analysis')

mkr501_flux = np.concatenate((mkr501_flux, mkr501_flux_1[:, :3]))

zz = 0.034
# model_ebl = 'finke2022'
# ebl_finke = OptDepth.readmodel(model=model_ebl)
#
#
# def funct_mk501(ee_array, N, gamma):
#     opacity = ebl_finke.opt_depth(zz, ee_array)
#     return (N * ee_array ** 2.
#             * ee_array ** (-gamma)
#             * np.exp(-opacity))
e_array = np.geomspace(
    mkr501_flux[0, 0] - 1., mkr501_flux[-1, 0] + 1., num=500)

ebl_finke = EBL.readascii('outputs/dust_reem_different_models/'
                          '3_grey_bodies.txt',
                          model_name='mine')
opacity = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
opacity = UnivariateSpline(np.log10(e_array), opacity, k=1, s=0)


def funct_mk501(ee_array, N, gamma):
    print(opacity(np.log10(ee_array[0])),
          opacity(np.log10(ee_array[-1])))
    return (N * ee_array ** 2.
            * ee_array ** (-gamma)
            * np.exp(-opacity(np.log10(ee_array)))
            )


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501)

m_best_fit = Minuit(combined_likelihood,
                    N=152., gamma=2.03)

m_best_fit.migrad()
m_best_fit.hesse()

print(m_best_fit.params)

xxx = np.linspace(0.99 * min(mkr501_flux[:, 0]),
                  1.01 * max(mkr501_flux[:, 0]),
                  num=200)

plt.plot(xxx, funct_mk501(xxx, *m_best_fit.values),
         label='Fit to power law + optical depth, model finke2022')
plt.plot(xxx, funct_mk501(xxx, 150., 2.))

plt.legend()

plt.title('Mkr 501 flare 1997')

plt.xlabel('E [TeV]')
plt.ylabel('E2dN/dE [10−12 erg cm−2 s−1]')

plt.yscale('log')
plt.xscale('log')
# plt.show()
plt.savefig('outputs/lhaaso/mk501flare.png',
            bbox_inches='tight')
# ----------------------------------------------------------------------

xxx_bins = np.linspace(
    mkr501_flux[0, 0] - 0.5,
    mkr501_flux[-1, 0] + 0.5,
    num=1000) * u.TeV
xxx_means = (xxx_bins[1:] + xxx_bins[:-1]) / 2.

recovered_bins = xxx_bins[:-2]
recovered_means = xxx_means[:-2]

edisp_obj = EDispGauss(sigma=0.2, bias=0.)
edisp_obj.fill(e_true_edges=xxx_bins.value,
               e_reco_edges=recovered_bins.value)
# plt.figure()
# plt.title('e disp matrix we use in the calculations')
# edisp_obj.plot()

eff_area = np.loadtxt('data/lhasso_characteristics/'
                      'WCD_eff_area_0to15deg.txt')
eff_area = eff_area[np.argsort(eff_area[:, 0]), :]
eff_area[:, 1] = eff_area[:, 1] * (u.m ** 2).to(u.cm ** 2)

spline_eff_area = UnivariateSpline(
    x=eff_area[:, 0] - 3., y=np.log10(eff_area[:, 1]),
    s=0, k=1, ext=1
)

energy_bins_obs = np.geomspace(3.5, 20., num=20) * u.TeV
energy_means_obs = np.sqrt(energy_bins_obs[1:] * energy_bins_obs[:-1])

total_int_time = (110. * u.h).to(u.s)


def calculate_number_counts(flux_spectrum_funct, iminuit_object):
    integral_kernel = flux_spectrum_funct(
        xxx_means.value, *iminuit_object.values)
    integral_kernel = (integral_kernel * 1e-12
                       * u.erg * u.cm ** -2 * u.s ** -1)
    integral_kernel = (integral_kernel / xxx_means ** 2.).to(
        u.TeV ** -1 * u.cm ** -2 * u.s ** -1)

    integral_kernel *= 10 ** spline_eff_area(
        x=np.log10(xxx_means.value)) * u.cm ** 2

    integral_kernel *= np.log(10) * xxx_means
    integral_kernel = integral_kernel.to(1 / u.s)[:, np.newaxis]

    integral_kernel = integral_kernel * edisp_obj._pdf_matrix

    integral_kernel = simpson(
        y=integral_kernel, x=np.log10(xxx_means.value), axis=0)

    count_numb = []

    integral_kernel = integral_kernel * np.log(10) * recovered_means

    for i in range(len(energy_bins_obs) - 1):
        where_bin = ((recovered_means > energy_bins_obs[i])
                     * (recovered_means < energy_bins_obs[i + 1]))

        if sum(where_bin) > 0:
            count_numb.append(simpson(
                y=integral_kernel[where_bin],
                x=np.log10(recovered_means[where_bin].value)))
        else:
            count_numb.append(0.)

    count_numb = (count_numb * total_int_time).value

    return count_numb

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            # print(pa)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


config_data = read_config_file('outputs/lhaaso/input_data.yml')
ebl_class = EBL_model.input_yaml_data_into_class(
    config_data, log_prints=False)


def fit_igl(lambda_igl, params):
    config_data['ssp_models'][key]['dust_reem_params']['fracts'] = \
        params.copy()

    return ebl_class.ebl_ssp_individualData(
        yaml_data=config_data['ssp_models'][key],
        x_data=lambda_igl)

upper_lims_ebldata, igl_ebldata = import_cb_data(
    lambda_min_total=0.1, lambda_max_total=1.e4,
    plot_measurs=False)

igl_ebldata = igl_ebldata[igl_ebldata['ref'] != 'ISO/ISOCAM (Clements+ ‘99)']
igl_ebldata = igl_ebldata[igl_ebldata['ref'] != 'SCUBA-2 (Hsu+ ‘16)']
igl_ebldata = igl_ebldata[igl_ebldata['ref'] != 'ALMA (Fujimoto+ ‘16)']
# ----------------------------------------------------------------------
tt_array = np.arange(30., 90., 5)
e_array = np.geomspace(
    np.min(mkr501_flux[:, 0])*0.5, np.max(mkr501_flux[:, 0])*1.5,
    num=500)

ebl_finke = EBL.readascii('outputs/dust_reem_different_models/'
                          '3_grey_bodies.txt',
                          model_name='mine')
opacity = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
opacity = UnivariateSpline(np.log10(e_array), opacity)



def funct_mk501_mine(ee_array, N, gamma):
    return (N * ee_array ** 2.
            * ee_array ** (-gamma)
            * np.exp(-opacity(np.log10(ee_array))))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501_mine)

m_best_fit_mine = Minuit(combined_likelihood, N=211., gamma=2.0)

m_best_fit_mine.migrad()
m_best_fit_mine.hesse()

print(m_best_fit_mine.params)

count_number_best_mine = calculate_number_counts(
    funct_mk501_mine, m_best_fit_mine)

nn_array_mine = []
nn_errors_mine = []
likelihoods_array_mine = []

fig_ebl = plt.figure()


def funct_mk501_cutoff(ee_array, N, gamma, Ecut):
    return (N * ee_array ** 2.
            * ee_array ** (-gamma)
            * np.exp(-(ee_array/Ecut) - opacity(np.log10(ee_array))))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501_cutoff)

m_best_fit_mine_cutoff = Minuit(combined_likelihood,
                                N=211., gamma=2.03, Ecut=18.)

m_best_fit_mine_cutoff.migrad()
m_best_fit_mine_cutoff.hesse()

print(m_best_fit_mine_cutoff.params)

count_number_best_mine_cutoff = calculate_number_counts(
    funct_mk501_cutoff, m_best_fit_mine_cutoff)

nn_array_mine_cutoff = []
nn_errors_mine_cutoff = []
likelihoods_array_mine_cutoff = []

key = 'SB99_3body_fittodata_70K450Kfixed'
ebl_cuba = EBL.readmodel('finke2022')

# plt.figure()
waves_ebl = np.logspace(-1, 3, num=500)

for d in tt_array:
    print(d)

    config_data['ssp_models'][key] \
    ['dust_reem_params']['T'][2] = d

    combined_likelihood = (LeastSquares(igl_ebldata['lambda'],
                                        igl_ebldata['nuInu'],
                                        igl_ebldata['1 sigma'],
                                        fit_igl))

    mm = Minuit(combined_likelihood,
               config_data['ssp_models'][key]['dust_reem_params']['fracts'])
    mm.limits = [[0, 0.7], [0., 0.7]]

    mm.migrad()
    mm.hesse()

    print(mm.params)
    config_data['ssp_models'][key]['dust_reem_params']['fracts'] = \
        [mm.params[0].value, mm.params[1].value]

    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

    aa, bb = np.meshgrid(10 ** ebl_cuba.x, ebl_cuba.y)

    ebl_axion = EBL(z=ebl_cuba.y, lmu=10 ** ebl_cuba.x,
                    nuInu=ebl_class.ebl_ssp_spline(aa, bb).T,
                    model='cuba+axion')

    opacity = UnivariateSpline(
        np.log10(e_array),
        ebl_axion.optical_depth(z0=zz, ETeV=e_array),
        k=1, s=0)


    plt.figure(fig_ebl)
    plt.plot(e_array, opacity(np.log10(e_array)))
    plt.plot(waves_ebl, ebl_axion.ebl_array(z=zz, lmu=waves_ebl),
             ls='--')
    plt.plot(e_array, ebl_axion.optical_depth(z0=zz, ETeV=e_array),
             ls=':')


    def funct_mk501_mine(ee_array, N, gamma):
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity(np.log10(ee_array))))


    def funct_mk501_cutoff(ee_array, N, gamma, Ecut):
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-(ee_array / Ecut) - opacity(np.log10(ee_array))))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_mine)

    m = Minuit(combined_likelihood, N=211., gamma=2.03)

    m.migrad()
    m.hesse()

    print(m.params)
    plt.figure(fig_spectra)
    plt.plot(xxx, funct_mk501_mine(xxx, *m.values))
    plt.figure(fig_ebl)
    plt.plot(waves_ebl, ebl_class.ebl_ssp_spline(
        waves_ebl, zz))
    nn_array_mine.append([*m.values])
    nn_errors_mine.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_mine, m)

    likelihoods_array_mine.append(
        likelihood_poisson(count_number, count_number_best_mine))
    print(count_number)
    # -------------------------------------------------------
    print(d, 'cutoff')

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_cutoff)

    m = Minuit(combined_likelihood,
               N=211., gamma=2.03, Ecut=18.)

    m.limits['Ecut'] = (0., 100)

    m.migrad()
    m.hesse()

    print(m.params)
    nn_array_mine_cutoff.append([*m.values])
    nn_errors_mine_cutoff.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_cutoff, m)

    likelihoods_array_mine_cutoff.append(
        likelihood_poisson(count_number, count_number_best_mine_cutoff))

# ----------------------------------------------------------------------

plt.figure()

max_value = np.max(likelihoods_array_mine)
max_value_cutoff = np.max(likelihoods_array_mine_cutoff)

plt.title(r'$\phi(E) = N \left(\frac{E}{1TeV}\right)^{-\Gamma}'
             r' e^{-E/E_\mathrm{cut} - \tau}$')

plt.scatter(tt_array, 2. * (likelihoods_array_mine - max_value),
            marker='.', s=400,
            c='b')
plt.scatter(tt_array, 2. * (likelihoods_array_mine_cutoff - max_value_cutoff),
            marker='x', label='w/ cutoff', s=300,
            c='orange')

plt.ylabel(r'$-2\Delta L$')
plt.xlabel('T [K]')

plt.legend()

plt.savefig('outputs/lhaaso/T_likelihoods_models.png',
            bbox_inches='tight')


# ----------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

plt.suptitle(r'$\phi(E) = N \left(\frac{E}{1TeV}\right)^{-\Gamma}'
             r' e^{-E/E_\mathrm{cut} - \tau}$')

plt.subplots_adjust(wspace=0, hspace=0)
ms = 15

nn_array_mine = np.array(nn_array_mine)
nn_errors_mine = np.array(nn_errors_mine)
nn_array_mine_cutoff = np.array(nn_array_mine_cutoff)
nn_errors_mine_cutoff = np.array(nn_errors_mine_cutoff)

# arg_max = np.where(np.array(my_ebl) == '3_grey_bodies.txt')[0][0]
arg_max = np.argmax(likelihoods_array_mine)

plt.subplot(131)
ii = 0
plt.errorbar(nn_array_mine[:, ii], tt_array,
             xerr=nn_errors_mine[:, ii],
             marker='', capsize=3, label='wto/ cutoff', ms=ms,
             ls='', c='b')
plt.errorbar(nn_array_mine_cutoff[:, ii], tt_array,
             xerr=nn_errors_mine_cutoff[:, ii],
             marker='', capsize=3, label='w/ cutoff', ms=ms,
             ls='', c='orange')

plt.axvspan(
    nn_array_mine[arg_max, ii] - nn_errors_mine[arg_max, ii],
    nn_array_mine[arg_max, ii] + nn_errors_mine[arg_max, ii],
    color='b', alpha=0.3, zorder=0, lw=0)
plt.axvspan(
    nn_array_mine_cutoff[arg_max, ii] - nn_errors_mine_cutoff[arg_max, ii],
    nn_array_mine_cutoff[arg_max, ii] + nn_errors_mine_cutoff[arg_max, ii],
    color='orange', alpha=0.3, zorder=0, lw=0)

plt.xlabel(r'$N$')

plt.legend(bbox_to_anchor=(0.9, 0.85), loc='upper left',
           bbox_transform=plt.gcf().transFigure)

plt.subplot(132)
ii = 1
plt.errorbar(nn_array_mine[:, ii], tt_array,
             xerr=nn_errors_mine[:, ii],
             marker='', capsize=3, label='wto/ cutoff', ms=ms,
             ls='', c='b')
plt.errorbar(nn_array_mine_cutoff[:, ii], tt_array,
             xerr=nn_errors_mine_cutoff[:, ii],
             marker='', capsize=3, label='w/ cutoff', ms=ms,
             ls='', c='orange')

plt.axvspan(
    nn_array_mine[arg_max, ii] - nn_errors_mine[arg_max, ii],
    nn_array_mine[arg_max, ii] + nn_errors_mine[arg_max, ii],
    color='b', alpha=0.3, zorder=0, lw=0)
plt.axvspan(
    nn_array_mine_cutoff[arg_max, ii] - nn_errors_mine_cutoff[arg_max, ii],
    nn_array_mine_cutoff[arg_max, ii] + nn_errors_mine_cutoff[arg_max, ii],
    color='orange', alpha=0.3, zorder=0, lw=0)

plt.tick_params('y', labelleft=False)

plt.xlabel(r'$\Gamma$')


plt.subplot(133)
ii = 2
plt.errorbar(nn_array_mine_cutoff[:, ii], tt_array,
             xerr=nn_errors_mine_cutoff[:, ii],
             marker='', capsize=3, label='w/ cutoff', ms=ms,
             ls='', c='orange')

plt.axvspan(
    nn_array_mine_cutoff[arg_max, ii] - nn_errors_mine_cutoff[arg_max, ii],
    nn_array_mine_cutoff[arg_max, ii] + nn_errors_mine_cutoff[arg_max, ii],
    color='orange', alpha=0.3, zorder=0, lw=0)

plt.tick_params('y', labelleft=False)

plt.xlabel(r'$E_\mathrm{cut}$')
plt.xscale('log')

plt.margins(y=0.2)

plt.savefig('outputs/lhaaso/T_parameters.png',
            bbox_inches='tight')

plt.show()
