import os
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

if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


def likelihood_poisson(mu_i_array_obs, mu_i_array_asimov):
    return sum(mu_i_array_asimov * np.log(mu_i_array_obs)
               - mu_i_array_obs)

# ----------------------------------------------------------------------
plt.figure()
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
model_ebl = 'finke2022'
ebl_finke = OptDepth.readmodel(model=model_ebl)


def funct_mk501(ee_array, N, gamma):
    opacity = ebl_finke.opt_depth(zz, ee_array)
    return (N * ee_array ** 2.
            * ee_array ** (-gamma)
            * np.exp(-opacity))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501)

m_best_fit = Minuit(combined_likelihood,
                    N=152., gamma=2.03)

m_best_fit.migrad()
m_best_fit.hesse()

print(m_best_fit.params)

xxx = np.linspace(0.99 * min(mkr501_flux[:, 0]),
                  1.01 * max(mkr501_flux[:, 0]))

plt.plot(xxx, funct_mk501(xxx, *m_best_fit.values),
         label='Fit to power law + optical depth, model finke2022')

plt.legend()

plt.title('Mkr 501 flare 1997')

plt.xlabel('E [TeV]')
plt.ylabel('E2dN/dE [10−12 erg cm−2 s−1]')

plt.yscale('log')
plt.xscale('log')

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
plt.figure()
plt.title('e disp matrix we use in the calculations')
edisp_obj.plot()

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


    count_number = []

    integral_kernel = integral_kernel * np.log(10) * recovered_means

    for i in range(len(energy_bins_obs) - 1):
        where_bin = ((recovered_means > energy_bins_obs[i])
                     * (recovered_means < energy_bins_obs[i + 1]))

        if sum(where_bin) > 0:
            count_number.append(simpson(
                y=integral_kernel[where_bin],
                x=np.log10(recovered_means[where_bin].value)))
        else:
            count_number.append(0.)

    count_number = (count_number * total_int_time).value

    return count_number

# ----------------------------------------------------------------------

def funct_mk501_mine(ee_array, N, gamma):
    ebl_finke = EBL.readascii('outputs/dust_reem_different_models/'
                              '3_grey_bodies.txt',
                              model_name='mine')
    opacity = ebl_finke.optical_depth(z0=zz, ETeV=ee_array)
    return (N * ee_array ** 2.
            * ee_array ** (-gamma)
            * np.exp(-opacity))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501_mine)

m_best_fit_mine = Minuit(combined_likelihood,
                         N=211., gamma=2.0)

m_best_fit_mine.migrad()
m_best_fit_mine.hesse()

print(m_best_fit_mine.params)

count_number_best_mine = calculate_number_counts(
    funct_mk501_mine, m_best_fit_mine)

nn_array_mine = []
nn_errors_mine = []
likelihoods_array_mine = []

my_ebl = ['3_grey_bodies.txt',
          'Chary.txt',
          'BOSA.txt']

for d in my_ebl:
    print(d)

    def funct_mk501_inside(ee_array, N, gamma):
        ebl_finke = EBL.readascii(
            'outputs/dust_reem_different_models/' + d,
                              model_name='mine')
        opacity = ebl_finke.optical_depth(z0=zz, ETeV=ee_array)
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               N=211., gamma=2.03)

    m.migrad()
    m.hesse()

    print(m.params)
    nn_array_mine.append([*m.values])
    nn_errors_mine.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_inside, m)

    likelihoods_array_mine.append(
        likelihood_poisson(count_number, count_number_best_mine))

# ----------------------------------------------------------------------

def funct_mk501(ee_array, N, gamma, Ecut):
    ebl_finke = EBL.readascii('outputs/dust_reem_different_models/'
                              '3_grey_bodies.txt',
                              model_name='mine')
    opacity = ebl_finke.optical_depth(z0=zz, ETeV=ee_array)
    return (N * ee_array ** 2.
            * ee_array ** (-gamma)
            * np.exp(-(ee_array/Ecut) - opacity))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501)

m_best_fit_mine_cutoff = Minuit(combined_likelihood,
                                N=211., gamma=2.03, Ecut=18.)

m_best_fit_mine_cutoff.migrad()
m_best_fit_mine_cutoff.hesse()

print(m_best_fit_mine_cutoff.params)

count_number_best_mine_cutoff = calculate_number_counts(
    funct_mk501, m_best_fit_mine_cutoff)

nn_array_mine_cutoff = []
nn_errors_mine_cutoff = []
likelihoods_array_mine_cutoff = []

my_ebl = ['3_grey_bodies.txt',
          'Chary.txt',
          'BOSA.txt']

for d in my_ebl:
    print(d)

    def funct_mk501_inside(ee_array, N, gamma, Ecut):
        ebl_finke = EBL.readascii(
            'outputs/dust_reem_different_models/' + d,
                              model_name='mine')
        opacity = ebl_finke.optical_depth(z0=zz, ETeV=ee_array)
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-(ee_array/Ecut) - opacity))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               N=211., gamma=2.03, Ecut=18.)

    m.limits['Ecut'] = (0., 100)

    m.migrad()
    m.hesse()

    print(m.params)
    nn_array_mine_cutoff.append([*m.values])
    nn_errors_mine_cutoff.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_inside, m)

    likelihoods_array_mine_cutoff.append(
        likelihood_poisson(count_number, count_number_best_mine_cutoff))

# ----------------------------------------------------------------------

fig_counts = plt.figure()
fig_spectrum = plt.figure()
alpha_array = ebl_finke.get_models()
# alpha_array = alpha_array[6:8]


nn_array = []
nn_errors = []
likelihoods_array = []

for d in alpha_array:
    print(d)
    ebl_finke = OptDepth.readmodel(model=d)


    def funct_mk501_inside(ee_array, N, gamma):
        ebl_finke = OptDepth.readmodel(model=d)
        opacity = ebl_finke.opt_depth(zz, ee_array)
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood, N=152., gamma=2.03)

    m.migrad()
    m.hesse()

    print(m.params)

    nn_array.append([*m.values])
    nn_errors.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_inside, m)
    plt.figure(fig_counts)
    plt.scatter(energy_means_obs, count_number,
                label=d)

    plt.figure(fig_spectrum)
    plt.loglog(
        energy_means_obs,
        funct_mk501_inside(energy_means_obs.value, *m.values),
        label=d)

    likelihoods_array.append(
        likelihood_poisson(count_number, count_number_best_mine))

# ----------------------------------------------------------------------

nn_array_cutoff = []
nn_errors_cutoff = []
likelihoods_array_cutoff = []

for d in alpha_array:
    print(d)
    ebl_finke = OptDepth.readmodel(model=d)

    def funct_mk501_inside(ee_array, N, gamma, Ecut):
        ebl_finke = OptDepth.readmodel(model=d)
        opacity = ebl_finke.opt_depth(zz, ee_array)
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-(ee_array/Ecut) - opacity))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               N=152., gamma=2.03, Ecut=8.21)

    m.limits['Ecut'] = (0., 100)

    m.migrad()
    m.hesse()

    print(m.params)
    nn_array_cutoff.append([*m.values])
    nn_errors_cutoff.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_inside, m)

    likelihoods_array_cutoff.append(
        likelihood_poisson(count_number, count_number_best_mine_cutoff))

# ----------------------------------------------------------------------
plt.figure(fig_counts)
plt.scatter(energy_means_obs, count_number_best_mine_cutoff, marker='x',
            zorder=1e3, s=200)
plt.ylabel('count number')
plt.xlabel('E [TeV]')
plt.legend()

plt.figure(fig_spectrum)
ebl_finke = OptDepth.readmodel(model=model_ebl)
plt.loglog(energy_means_obs,
           funct_mk501(energy_means_obs.value,
                       *m_best_fit_mine_cutoff.values),
           marker='x', ms=20,
           zorder=0)
plt.ylabel('flux')
plt.xlabel('E [TeV]')

plt.legend(loc=3)

# ----------------------------------------------------------------------

plt.subplots(2, 1, height_ratios=(len(alpha_array), len(my_ebl)*1.5))

plt.suptitle(r'$\phi(E) = N \left(\frac{E}{1TeV}\right)^{-\Gamma}'
             r' e^{-E/E_\mathrm{cut} - \tau}$')

plt.subplots_adjust(wspace=0, hspace=0)

plt.subplot(211)
plt.scatter(2. * (likelihoods_array - np.max(likelihoods_array)),
            alpha_array, marker='.', label='wto/ cutoff', s=400,
            c='b')
plt.scatter(2. * (likelihoods_array_cutoff
            - np.max(likelihoods_array_cutoff)),
            alpha_array, marker='x', label='w/ cutoff', s=300,
            c='orange')

plt.legend(loc=9)
plt.subplot(212)

plt.scatter(2. * (likelihoods_array_mine
                  - np.max(likelihoods_array_mine)),
            my_ebl, marker='.', s=400,
            c='b')
plt.scatter(2. * (likelihoods_array_mine_cutoff
            - np.max(likelihoods_array_mine_cutoff)),
            my_ebl, marker='x', label='w/ cutoff', s=300,
            c='orange')

plt.xlabel(r'$-2\Delta L$')

plt.margins(y=0.3)

plt.savefig('outputs/lhaaso/likelihoods_models.png',
            bbox_inches='tight')


# ----------------------------------------------------------------------
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    2, 3, height_ratios=(len(alpha_array), len(my_ebl)*1.5))

plt.suptitle(r'$\phi(E) = N \left(\frac{E}{1TeV}\right)^{-\Gamma}'
             r' e^{-E/E_\mathrm{cut} - \tau}$')

plt.subplots_adjust(wspace=0, hspace=0)
ms = 15

nn_array = np.array(nn_array)
nn_errors = np.array(nn_errors)
nn_array_cutoff = np.array(nn_array_cutoff)
nn_errors_cutoff = np.array(nn_errors_cutoff)

nn_array_mine = np.array(nn_array_mine)
nn_errors_mine = np.array(nn_errors_mine)
nn_array_mine_cutoff = np.array(nn_array_mine_cutoff)
nn_errors_mine_cutoff = np.array(nn_errors_mine_cutoff)

arg_max = np.where(np.array(my_ebl) == '3_grey_bodies.txt')[0][0]

plt.subplot(231)
ii = 0
plt.errorbar(nn_array[:, ii], alpha_array,
             xerr=nn_errors[:, ii],
             marker='', capsize=3, label='wto/ cutoff', ms=ms,
             ls='', c='b')
plt.errorbar(nn_array_cutoff[:, ii], alpha_array,
             xerr=nn_errors_cutoff[:, ii],
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

plt.legend(bbox_to_anchor=(0.9, 0.85), loc='upper left',
           bbox_transform=plt.gcf().transFigure)

plt.tick_params('x', labelbottom=False)


plt.subplot(232)
ii = 1
plt.errorbar(nn_array[:, ii], alpha_array,
             xerr=nn_errors[:, ii],
             marker='', capsize=3, label='wto/ cutoff', ms=ms,
             ls='', c='b')
plt.errorbar(nn_array_cutoff[:, ii], alpha_array,
             xerr=nn_errors_cutoff[:, ii],
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

plt.tick_params('x', labelbottom=False)
plt.tick_params('y', labelleft=False)


plt.subplot(233)
ii = 2
plt.errorbar(nn_array_cutoff[:, ii], alpha_array,
             xerr=nn_errors_cutoff[:, ii],
             marker='', capsize=3, label='w/ cutoff', ms=ms,
             ls='', c='orange')

plt.axvspan(
    nn_array_mine_cutoff[arg_max, ii] - nn_errors_mine_cutoff[arg_max, ii],
    nn_array_mine_cutoff[arg_max, ii] + nn_errors_mine_cutoff[arg_max, ii],
    color='orange', alpha=0.3, zorder=0, lw=0)

plt.tick_params('x', labelbottom=False)
plt.tick_params('y', labelleft=False)

plt.xscale('log')

plt.subplot(234, sharex=ax1)
ii = 0
plt.errorbar(nn_array_mine[:, ii], my_ebl,
             xerr=nn_errors_mine[:, ii],
             marker='', capsize=3, label='wto/ cutoff', ms=ms,
             ls='', c='b')
plt.errorbar(nn_array_mine_cutoff[:, ii], my_ebl,
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

plt.margins(y=0.2)

plt.subplot(235, sharex=ax2)
ii = 1
plt.errorbar(nn_array_mine[:, ii], my_ebl,
             xerr=nn_errors_mine[:, ii],
             marker='', capsize=3, label='wto/ cutoff', ms=ms,
             ls='', c='b')
plt.errorbar(nn_array_mine_cutoff[:, ii], my_ebl,
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

plt.margins(y=0.2)


plt.subplot(236, sharex=ax3)
ii = 2
plt.errorbar(nn_array_mine_cutoff[:, ii], my_ebl,
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

plt.savefig('outputs/lhaaso/parameters.png',
            bbox_inches='tight')

plt.show()
