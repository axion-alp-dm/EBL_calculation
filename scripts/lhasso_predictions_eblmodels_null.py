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


mkr501_flux = np.concatenate((mkr501_flux_1[:8, :3], mkr501_flux))

zz = 0.034

e_array = np.geomspace(
    np.min(mkr501_flux[:, 0])*0.5, np.max(mkr501_flux[:, 0])*1.5,
    num=500)

model_ebl = 'finke2022'
ebl_finke = OptDepth.readmodel(model=model_ebl)
opacity = ebl_finke.opt_depth(z=zz, ETeV=e_array)
opacity = UnivariateSpline(
        np.log10(e_array), opacity, k=1, s=0)


# def funct_mk501(ee_array, N, gamma):
#     opacity = ebl_finke.opt_depth(zz, ee_array)
#     return (N * ee_array ** 2.
#             * ee_array ** (-gamma)
#             * np.exp(-opacity))


def funct_mk501(ee_array, phi0, E0, alpha, beta):
    return (phi0 * ee_array ** 2.
            * (ee_array / E0) ** (
                    - alpha - beta * np.log(ee_array/E0)))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501)

# m_best_fit = Minuit(combined_likelihood,
#                     N=152., gamma=2.03)
m_best_fit = Minuit(combined_likelihood,
                    phi0=1e-12, E0=430., alpha=5., beta=0.5)

m_best_fit.migrad()
m_best_fit.hesse()

print(m_best_fit.params)
print(m_best_fit.values)

xxx = np.geomspace(0.99 * min(mkr501_flux[:, 0]),
                   1.01 * max(mkr501_flux[:, 0]))

plt.plot(xxx, funct_mk501(xxx, *m_best_fit.values),
         label='Fit to LogParabola')

plt.legend()

plt.title('Mkr 501 flare 1997')

plt.xlabel('E [TeV]')
plt.ylabel('E2dN/dE [10−12 erg cm−2 s−1]')

plt.yscale('log')
plt.xscale('log')
# plt.show()
plt.savefig('outputs/lhaaso/mk501flare_logparabola.png',
            bbox_inches='tight')
# ----------------------------------------------------------------------

xxx_bins = np.geomspace(
    min(mkr501_flux[:, 0]) - 0.5,
    max(mkr501_flux[:, 0]) + 0.5,
    num=1000) * u.TeV
# xxx_means = (xxx_bins[1:] + xxx_bins[:-1]) / 2.
xxx_means = np.sqrt(xxx_bins[1:] * xxx_bins[:-1])

recovered_bins = xxx_bins.copy()#[:-2]
recovered_means = xxx_means.copy()#[:-2]

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

energy_bins_obs = np.geomspace(0.5, 21., num=20) * u.TeV
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

# ----------------------------------------------------------------------
fig_counts = plt.figure()
fig_spectrum = plt.figure()

e_array = np.geomspace(
    np.min(mkr501_flux[:, 0])*0.5, np.max(mkr501_flux[:, 0])*1.5,
    num=500)

# ebl_finke = EBL.readascii('outputs/lhaaso/'
#                               '3_grey_bodies.txt',
#                               model_name='mine')
ebl_finke = EBL.readmodel('finke2022')
opacity = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
opacity_best = UnivariateSpline(
        np.log10(e_array), opacity, k=1, s=0)

def funct_mk501_mine(ee_array, phi0, E0, alpha, beta):
    return (phi0 * ee_array ** 2.
            * (ee_array / E0) ** (
                    - alpha - beta * np.log(ee_array/E0))
            * np.exp(-opacity_best(np.log10(ee_array))))


combined_likelihood = LeastSquares(
    mkr501_flux[:, 0], mkr501_flux[:, 1],
    mkr501_flux[:, 2], funct_mk501_mine)

m_best_fit_mine = Minuit(combined_likelihood,
                    phi0=1e-12, E0=430., alpha=5., beta=0.5)

m_best_fit_mine.migrad()
m_best_fit_mine.hesse()

print(m_best_fit_mine.params)

count_number_best_mine = calculate_number_counts(
    funct_mk501_mine, m_best_fit_mine)

nn_array_mine = []
nn_errors_mine = []
likelihoods_array_mine = []


# def funct_mk501_cutoff(ee_array, N, gamma, Ecut):
#     return (N * ee_array ** 2.
#             * ee_array ** (-gamma)
#             * np.exp(-(ee_array/Ecut) - opacity(np.log10(ee_array))))
#
#
# combined_likelihood = LeastSquares(
#     mkr501_flux[:, 0], mkr501_flux[:, 1],
#     mkr501_flux[:, 2], funct_mk501_cutoff)
#
# m_best_fit_mine_cutoff = Minuit(combined_likelihood,
#                                 N=211., gamma=2.03, Ecut=18.)
#
# m_best_fit_mine_cutoff.migrad()
# m_best_fit_mine_cutoff.hesse()
#
# print(m_best_fit_mine_cutoff.params)
#
# count_number_best_mine_cutoff = calculate_number_counts(
#     funct_mk501_cutoff, m_best_fit_mine_cutoff)
#
nn_array_mine_cutoff = []
nn_errors_mine_cutoff = []
likelihoods_array_mine_cutoff = []

my_ebl = ['3_grey_bodies.txt',
          'Chary.txt',
          'BOSA.txt']

# for d in my_ebl:
#     print(d)
#     ebl_finke = EBL.readascii(
#         'outputs/lhaaso/' + d, model_name='mine')
#     opacity = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
#     opacity = UnivariateSpline(
#         np.log10(e_array), opacity, k=1, s=0)
#
#     def funct_mk501_inside(ee_array, phi0, E0, alpha, beta):
#         return (phi0 * ee_array ** 2.
#                 * (ee_array / E0) ** (
#                         - alpha - beta * np.log(ee_array / E0))
#                 * np.exp(-opacity(np.log10(ee_array))))
#
#
#     combined_likelihood = LeastSquares(
#         mkr501_flux[:, 0], mkr501_flux[:, 1],
#         mkr501_flux[:, 2], funct_mk501_inside)
#
#     m = Minuit(combined_likelihood,
#                phi0=1e-12, E0=430., alpha=5., beta=0.5)
#
#     m.migrad()
#     m.hesse()
#
#     print(m.params)
#     nn_array_mine.append([*m.values])
#     nn_errors_mine.append([*m.errors])
#
#     count_number = calculate_number_counts(funct_mk501_inside, m)
#
#     likelihoods_array_mine.append(
#         likelihood_poisson(count_number, count_number_best_mine))

    # ----------------------------------------------------------------------

    # print(d)
    #
    # def funct_mk501_inside(ee_array, N, gamma, Ecut):
    #     return (N * ee_array ** 2.
    #             * ee_array ** (-gamma)
    #             * np.exp(-(ee_array/Ecut) - opacity(np.log10(ee_array))))
    #
    #
    # combined_likelihood = LeastSquares(
    #     mkr501_flux[:, 0], mkr501_flux[:, 1],
    #     mkr501_flux[:, 2], funct_mk501_inside)
    #
    # m = Minuit(combined_likelihood,
    #            N=211., gamma=2.03, Ecut=18.)
    #
    # m.limits['Ecut'] = (0., 100)
    #
    # m.migrad()
    # m.hesse()
    #
    # print(m.params)
    # nn_array_mine_cutoff.append([*m.values])
    # nn_errors_mine_cutoff.append([*m.errors])
    #
    # count_number = calculate_number_counts(funct_mk501_inside, m)
    #
    # likelihoods_array_mine_cutoff.append(
    #     likelihood_poisson(count_number, count_number_best_mine_cutoff))


# ----------------------------------------------------------------------
alpha_array = OptDepth.get_models()
# alpha_array = alpha_array[6:8]


nn_array = []
nn_errors = []
likelihoods_array = []

for d in alpha_array:
    print(d)
    ebl_finke = OptDepth.readmodel(model=d)
    opacity = ebl_finke.opt_depth(z=zz, ETeV=e_array)
    opacity = UnivariateSpline(
        np.log10(e_array), opacity, k=1, s=0)

    def funct_mk501_inside(ee_array, phi0, E0, alpha, beta):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (
                        - alpha - beta * np.log(ee_array / E0))
                * np.exp(-opacity(np.log10(ee_array))))

    # def funct_mk501_inside(ee_array, N, gamma):
    #     ebl_finke = OptDepth.readmodel(model=d)
    #     opacity = ebl_finke.opt_depth(zz, ee_array)
    #     return (N * ee_array ** 2.
    #             * ee_array ** (-gamma)
    #             * np.exp(-opacity))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               phi0=1e-12, E0=430., alpha=5., beta=0.5)

    m.migrad()
    m.hesse()

    print(m.params)

    nn_array.append([*m.values])
    nn_errors.append([*m.errors])

    count_number = calculate_number_counts(funct_mk501_inside, m)
    print(count_number)
    likelihoods_arr_one = []
    figg, axgg = plt.subplots()
    plt.figure()
    plt.xscale('log')
    plt.scatter(energy_means_obs, count_number, marker='x', zorder=20)

    for nn in range(1000):
        rng = np.random.default_rng()
        poiss = rng.poisson(lam=count_number)
        likelihoods_arr_one.append(likelihood_poisson(poiss, count_number))
        if nn%250==0:
            print(poiss)
            print(count_number * np.log(poiss) - poiss)
            print(likelihoods_arr_one[-1])
            print()
            plt.scatter(energy_means_obs, poiss)

            axgg.scatter(poiss, count_number * np.log(poiss) - poiss)


    plt.figure()
    plt.hist(likelihoods_arr_one, bins=20)
    plt.axvline(np.max(likelihoods_arr_one) - 2.71, c='k',
                label='Max ln(L) - 2.71')
    plt.xlabel(r'$ln(L)$')
    plt.legend()
    plt.figure()
    plt.hist(np.max(likelihoods_arr_one) - likelihoods_arr_one, bins=20)

    plt.axvline(2.71, c='k', label='2.71')

    perc95 = np.percentile(
        np.max(likelihoods_arr_one) - likelihoods_arr_one, 95)
    plt.axvline(perc95, label='5 and 95 percentiles')
    perc95 = np.percentile(
        np.max(likelihoods_arr_one) - likelihoods_arr_one, 5)
    plt.axvline(perc95)
    plt.xlabel(r'$\Delta ln(L)$')
    plt.legend()
    plt.show()

    if d in ['finke', 'kneiske', 'dominguez-lower', 'inoue']:
        plt.figure(fig_counts)
        plt.scatter(energy_means_obs, count_number,
                    label=d)

        plt.figure(fig_spectrum)
        plt.loglog(xxx, funct_mk501_inside(xxx, *m.values),
            label=d)

    likelihoods_array.append(
        likelihood_poisson(count_number, count_number_best_mine))

# ----------------------------------------------------------------------

nn_array_cutoff = []
nn_errors_cutoff = []
likelihoods_array_cutoff = []
#
# for d in alpha_array:
#     print(d)
#     ebl_finke = OptDepth.readmodel(model=d)
#
#     def funct_mk501_inside(ee_array, N, gamma, Ecut):
#         ebl_finke = OptDepth.readmodel(model=d)
#         opacity = ebl_finke.opt_depth(zz, ee_array)
#         return (N * ee_array ** 2.
#                 * ee_array ** (-gamma)
#                 * np.exp(-(ee_array/Ecut) - opacity))
#
#
#     combined_likelihood = LeastSquares(
#         mkr501_flux[:, 0], mkr501_flux[:, 1],
#         mkr501_flux[:, 2], funct_mk501_inside)
#
#     m = Minuit(combined_likelihood,
#                N=152., gamma=2.03, Ecut=8.21)
#
#     m.limits['Ecut'] = (0., 100)
#
#     m.migrad()
#     m.hesse()
#
#     print(m.params)
#     nn_array_cutoff.append([*m.values])
#     nn_errors_cutoff.append([*m.errors])
#
#     count_number = calculate_number_counts(funct_mk501_inside, m)
#
#     likelihoods_array_cutoff.append(
#         likelihood_poisson(count_number, count_number_best_mine_cutoff))

    # if d in ['finke', 'kneiske', 'dominguez-lower']:
    #     plt.figure(fig_counts)
    #     plt.scatter(energy_means_obs, count_number,
    #                 label=d, marker='s')
    #
    #     plt.figure(fig_spectrum)
    #     plt.loglog(
    #         energy_means_obs,
    #         funct_mk501_inside(energy_means_obs.value, *m.values),
    #         label=d, ls='--')

# ----------------------------------------------------------------------
plt.figure(fig_counts)
plt.scatter(energy_means_obs, count_number_best_mine, marker='x',
            zorder=1e3, s=200)
plt.ylabel('count number')
plt.xlabel('E [TeV]')
plt.xscale('log')
plt.legend()

plt.savefig('outputs/lhaaso/counts_logparabola.png',
            bbox_inches='tight')

plt.figure(fig_spectrum)
ebl_finke = OptDepth.readmodel(model=model_ebl)

plt.loglog(xxx,
           funct_mk501_mine(xxx, *m_best_fit_mine.values),
           marker='x', ms=20,
           zorder=0, label='Best fit 3 body')

plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='o', label='Data', zorder=100)
plt.ylabel('flux')
plt.xlabel('E [TeV]')

plt.legend(loc=3)

plt.savefig('outputs/lhaaso/spectrum_logparabola.png',
            bbox_inches='tight')

plt.figure()
for d in ['kneiske', 'finke', 'dominguez-lower', 'inoue']:
    ebl_finke = OptDepth.readmodel(model=d)
    plt.plot(xxx, ebl_finke.opt_depth(z=zz, ETeV=xxx),
                    label=d)

ebl_finke = EBL.readascii(file_name='outputs/lhaaso/'
                              '3_grey_bodies.txt',
                              model_name='mine')
plt.plot(xxx, ebl_finke.optical_depth(z0=zz, ETeV=xxx),
         label='3 grey body')
plt.ylabel('opacity at redshift')
plt.xlabel('energy [TeV]')
plt.xscale('log')
plt.legend()

plt.savefig('outputs/lhaaso/ebl_models_logparabola.png',
            bbox_inches='tight')
# ----------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(
#     2, 1, height_ratios=(len(alpha_array), len(my_ebl)*1.5))
gs_kw = dict(height_ratios=(len(alpha_array), len(my_ebl)*1.5))
fig, (ax1, ax2) = plt.subplots(
    2, 1, gridspec_kw=gs_kw)

max_value = np.max(likelihoods_array_mine)
# max_value_cutoff = np.max(likelihoods_array_mine_cutoff)



plt.suptitle(r'$\phi(E) = \phi_0 \left(\frac{E}{E_0}\right)^{-\alpha '
             r'- \beta \mathrm{ln}\left(E/E_0\right)} e^{-\tau}$')

plt.subplots_adjust(wspace=0, hspace=0)

plt.subplot(211)
plt.scatter(2. * (likelihoods_array - max_value),
            alpha_array, marker='.', label='wto/ cutoff', s=400,
            c='b')
# plt.scatter(2. * (likelihoods_array_cutoff - max_value_cutoff),
#             alpha_array, marker='x', label='w/ cutoff', s=300,
#             c='orange')

plt.legend(loc=9)
plt.subplot(212, sharex=ax1)

plt.scatter(2. * (likelihoods_array_mine - max_value),
            my_ebl, marker='.', s=400,
            c='b')
# plt.scatter(2. * (likelihoods_array_mine_cutoff - max_value_cutoff),
#             my_ebl, marker='x', label='w/ cutoff', s=300,
#             c='orange')

plt.xlabel(r'$-2\Delta L$')

plt.margins(y=0.3)

plt.savefig('outputs/lhaaso/likelihoods_models_logparabola.png',
            bbox_inches='tight')


# ----------------------------------------------------------------------
# fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
#     2, 3, height_ratios=(len(alpha_array), len(my_ebl)*1.5))

fig, ax = plt.subplots(2, m.npar, gridspec_kw=gs_kw)

plt.suptitle(r'$\phi(E) = \phi_0 \left(\frac{E}{E_0}\right)^{-\alpha '
             r'- \beta \mathrm{ln}\left(\frac{E}{E_0}\right)} e^{-\tau}$')

plt.subplots_adjust(wspace=0, hspace=0)

ms = 15

nn_array = np.array(nn_array)
nn_errors = np.array(nn_errors)
# nn_array_cutoff = np.array(nn_array_cutoff)
# nn_errors_cutoff = np.array(nn_errors_cutoff)

nn_array_mine = np.array(nn_array_mine)
nn_errors_mine = np.array(nn_errors_mine)
# nn_array_mine_cutoff = np.array(nn_array_mine_cutoff)
# nn_errors_mine_cutoff = np.array(nn_errors_mine_cutoff)

arg_max = np.where(np.array(my_ebl) == '3_grey_bodies.txt')[0][0]
plot_cutoff = False
logscale_x = [True, False, False, False]

# xx_labels = m.parameters
xx_labels = [r'$\phi_0$', r'$E_0$', r'$\alpha$', r'$\beta$']

for col in range(m.npar):

    ax[1, col].set_xlabel(xx_labels[col])

    minn = np.min((
        np.min(nn_array[:, col] - nn_errors[:, col]),
        np.min(nn_array_mine[:, col] - nn_errors_mine[:, col])))
    maxx = np.max((
        np.max(nn_array[:, col] + nn_errors[:, col]),
        np.max(nn_array_mine[:, col] + nn_errors_mine[:, col])))

    if logscale_x[col]:
        ax[0, col].set_xscale('log')
        ax[1, col].set_xscale('log')

        ax[0, col].set_xlim(0.8 * minn, 1.1 * maxx)
        ax[1, col].set_xlim(0.8 * minn, 1.1 * maxx)

    else:
        aaa = (maxx - minn) * 0.1
        ax[0, col].set_xlim(minn - aaa, maxx + aaa)
        ax[1, col].set_xlim(minn - aaa, maxx + aaa)

    ax[1, col].margins(y=0.2)

    ax[0, col].errorbar(nn_array[:, col], alpha_array,
                        xerr=nn_errors[:, col],
                        marker='', capsize=3, label='wto/ cutoff', ms=ms,
                        ls='', c='b')

    ax[1, col].errorbar(nn_array_mine[:, col], my_ebl,
                        xerr=nn_errors_mine[:, col],
                        marker='', capsize=3, label='wto/ cutoff', ms=ms,
                        ls='', c='b')

    ax[0, col].axvspan(
        nn_array_mine[arg_max, col] - nn_errors_mine[arg_max, col],
        nn_array_mine[arg_max, col] + nn_errors_mine[arg_max, col],
        color='b', alpha=0.3, zorder=0, lw=0)

    ax[1, col].axvspan(
        nn_array_mine[arg_max, col] - nn_errors_mine[arg_max, col],
        nn_array_mine[arg_max, col] + nn_errors_mine[arg_max, col],
        color='b', alpha=0.3, zorder=0, lw=0)

    if plot_cutoff:
        ax[0, col].errorbar(nn_array_cutoff[:, col], alpha_array,
                            xerr=nn_errors_cutoff[:, col],
                            marker='', capsize=3, label='w/ cutoff', ms=ms,
                            ls='', c='orange')

        ax[0, col].axvspan(
            nn_array_mine_cutoff[arg_max, col]
            - nn_errors_mine_cutoff[arg_max, col],
            nn_array_mine_cutoff[arg_max, col]
            + nn_errors_mine_cutoff[arg_max, col],
            color='orange', alpha=0.3, zorder=0, lw=0)

        ax[1, col].errorbar(nn_array_mine_cutoff[:, col], my_ebl,
                            xerr=nn_errors_mine_cutoff[:, col],
                            marker='', capsize=3, label='w/ cutoff', ms=ms,
                            ls='', c='orange')
        ax[1, col].axvspan(
            nn_array_mine_cutoff[arg_max, col] - nn_errors_mine_cutoff[
                arg_max, col],
            nn_array_mine_cutoff[arg_max, col] + nn_errors_mine_cutoff[
                arg_max, col],
            color='orange', alpha=0.3, zorder=0, lw=0)
    # plt.subplot(2, m.npar, col + 1)
    # plt.tick_params('x', labelbottom=False)
    if col > 0:
        ax[0, col].tick_params('y', labelleft=False)
        ax[1, col].tick_params('y', labelleft=False)

    if col == 0:
        plt.legend(bbox_to_anchor=(0.9, 0.85), loc='upper left',
                   bbox_transform=plt.gcf().transFigure)
plt.savefig('outputs/lhaaso/parameters_logparabola.png',
            bbox_inches='tight')

plt.show()
