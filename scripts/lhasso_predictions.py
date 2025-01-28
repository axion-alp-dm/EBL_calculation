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

if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

list_all = os.listdir('data/lhasso_characteristics')
print(list_all)
list_wcd = []
list_km2a = []
for i in list_all:
    if 'WCD' in i and 'area' in i:
        list_wcd.append(i)
    if 'KM2A' in i and 'area' in i:
        list_km2a.append(i)
list_all = []

all_info = {}
plt.subplots(1, 2, figsize=(10, 5))
plt.suptitle('WCD\nsensitive to gamma rays with energies '
             'between 100 GeV - 30 TeV')
plt.subplot(121)

for ii in list_wcd:
    aaa = np.loadtxt('data/lhasso_characteristics/' + ii)
    aaa_sort = np.argsort(aaa[:, 0])
    all_info[ii] = aaa[aaa_sort, :]
    plt.plot(all_info[ii][:, 0], all_info[ii][:, 1])
plt.yscale('log')
plt.xlabel(r'log$_{10}$(E) [GeV]')
plt.ylabel(r'Effective area [m$^2$]')

plt.subplot(122)
aaa = np.loadtxt('data/lhasso_characteristics/'
                 'WCD_energy_resolution.txt')
aaa = aaa[np.argsort(aaa[:, 0]), :]

plt.plot(aaa[:, 0], aaa[:, 1])

plt.xscale('log')
plt.xlabel('E [TeV]')
plt.ylabel(r'$\Delta$E/E')


plt.subplots(1, 2, figsize=(10, 5))
plt.suptitle('KM2A - gamma\n'
             ' energy range from 10 TeV to 100 PeV')
plt.subplot(121)
aaa = np.loadtxt('data/lhasso_characteristics/'
                 'KM2A_eff_area_gamma.txt')
aaa = aaa[np.argsort(aaa[:, 0]), :]
plt.plot(aaa[:, 0], aaa[:, 1])

plt.xlabel(r'log$_{10}$(E) [GeV]')
plt.ylabel(r'Effective area [km$^2$]')



plt.subplot(122)
aaa = np.loadtxt('data/lhasso_characteristics/'
                 'KM2A_energy_resolution.txt')
aaa = aaa[np.argsort(aaa[:, 0]), :]

plt.plot(aaa[:, 0], aaa[:, 1])

plt.xlabel(r'log$_{10}$(E) [GeV]')
plt.ylabel('Energy resolution (%)')

edisp_obj = EDispGauss(sigma=0.2, bias=0.)
xx = np.geomspace(0.1, 10., num=100)
plt.figure()
edisp_obj.fill(
    e_true_edges=xx,
    e_reco_edges=xx*10.)
edisp_obj.plot()

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
             ls='', marker='o')
plt.errorbar(mkr501_flux_1[:, 0], mkr501_flux_1[:, 1],
             yerr=mkr501_flux_1[:, 2],
             ls='', marker='o')
plt.yscale('log')
plt.xscale('log')

mkr501_flux = np.concatenate((mkr501_flux, mkr501_flux_1[:, :3]))

from ebltable.tau_from_model import OptDepth
from scipy.optimize import curve_fit
ebl_finke = OptDepth.readmodel(model='finke2022')
zz = 0.034


def funct_mk501(ee_array, N, Ecut, E0, tau=1.31):
    return (N * ee_array ** 2.
            * ee_array**Ecut * np.exp(-(ee_array/E0)**tau))


fit_mkr501 = curve_fit(funct_mk501,
                       ydata=mkr501_flux[:, 1],
                       xdata=mkr501_flux[:, 0],
                       p0=[150, -2., 8.2, 1.3],
                       )
print(fit_mkr501[0])
print(np.sqrt(np.diag(fit_mkr501[1])))
print()

xxx = np.linspace(0.99 * min(mkr501_flux[:, 0]),
                  1.01 * max(mkr501_flux[:, 0]))

plt.plot(xxx, funct_mk501(xxx, fit_mkr501[0][0], fit_mkr501[0][1],
                          fit_mkr501[0][2], fit_mkr501[0][3],))

def funct_mk501_norm(ee_array, N):
    return (funct_mk501(ee_array, N=N, Ecut=-2.03,
                       E0=8.21, tau=1.31))
fit_mkr501_norm = curve_fit(funct_mk501_norm,
                       ydata=mkr501_flux[:, 1],
                       xdata=mkr501_flux[:, 0],
                       p0=[150],
                       )
plt.plot(xxx, funct_mk501(xxx, fit_mkr501_norm[0][0],
                          -2.03, 8.21, 1.31), ls='--')

plt.title('Mkr 501 flare 1997')
plt.xlabel('E [TeV]')
plt.ylabel('E2dN/dE [10−12 erg cm−2 s−1]')
# plt.show()
# ----------------------------------------------------------------------

xxx_bins = np.linspace(
    mkr501_flux[0, 0] - 0.5,
    mkr501_flux[-1, 0] + 0.5,
    num=1000) * u.TeV
xxx_means = (xxx_bins[1:] + xxx_bins[:-1])/2.
integral_kernel = funct_mk501(xxx_means.value,
                              N=fit_mkr501_norm[0][0],
                              Ecut=-2.03, E0=8.21, tau=1.31)
integral_kernel = integral_kernel * 1e-12 * u.erg * u.cm**-2 * u.s**-1
integral_kernel = (integral_kernel / xxx_means**2.).to(
    u.TeV**-1 * u.cm**-2 * u.s**-1)
print(np.max(integral_kernel))

eff_area = np.loadtxt('data/lhasso_characteristics/'
                      'WCD_eff_area_0to15deg.txt')
eff_area = eff_area[np.argsort(eff_area[:, 0]), :]
eff_area[:, 1] = eff_area[:, 1] * (u.m**2).to(u.cm**2)

spline_eff_area = UnivariateSpline(
    x=eff_area[:, 0] - 3., y=np.log10(eff_area[:, 1]),
    s=0, k=1, ext=1
)

integral_kernel *= 10**spline_eff_area(x=np.log10(xxx_means.value)) * u.cm**2
print(integral_kernel[0])

recovered_bins = xxx_bins[:-2]
recovered_means = xxx_means[:-2]


edisp_obj = EDispGauss(sigma=0.2, bias=0.)
edisp_obj.fill(e_true_edges=xxx_bins.value,
               e_reco_edges=recovered_bins.value)
plt.figure()
plt.title('e disp matrix we use in the calculations')
edisp_obj.plot()


integral_kernel *= np.log(10) * xxx_means
integral_kernel = integral_kernel.to(1/u.s)[:, np.newaxis]

integral_kernel = integral_kernel * edisp_obj._pdf_matrix

integral_kernel = simpson(y=integral_kernel,
                          x=np.log10(xxx_means.value), axis=0)
print(np.max(integral_kernel))

energy_bins_obs = np.geomspace(3.5, 20., num=20) * u.TeV
energy_means_obs = np.sqrt(energy_bins_obs[1:] * energy_bins_obs[:-1])
count_number_best = []
bbb = 0

integral_kernel = integral_kernel * np.log(10) * recovered_means

for i in range(len(energy_bins_obs) - 1):
    where_bin = ((recovered_means > energy_bins_obs[i])
                 * (recovered_means < energy_bins_obs[i+1]))

    if sum(where_bin) > 0:
        count_number_best.append(simpson(
            y=integral_kernel[where_bin],
            x=np.log10(recovered_means[where_bin].value)))
    else:
        count_number_best.append(0.)


total_int_time = (110. * u.h).to(u.s)
count_number_best = (count_number_best * total_int_time).value

plt.figure()
plt.scatter(energy_means_obs, count_number_best)
plt.ylabel('Best fit count number')
plt.xlabel('E [TeV]')
print(count_number_best)
# plt.show()
likelihoods_array = []


def likelihood_poisson(mu_i_array_obs, mu_i_array_asimov):
    return sum(mu_i_array_asimov * np.log(mu_i_array_obs)
               - mu_i_array_obs)


total_int_time = (110. * u.h).to(u.s)

edisp_obj.fill(e_true_edges=xxx_bins.value,
               e_reco_edges=recovered_bins.value)


alpha_array = np.linspace(1., 2., num=500)
# alpha_array = fit_mkr501[0][-1] * np.array([0.5, 1., 2.])
nn_array = []

fig_counts = plt.figure()
fig_spectrum = plt.figure()

for d in alpha_array:
    def funct_mk501_inside(ee_array, N, Ecut, E0):
        return funct_mk501(ee_array, N=N, Ecut=Ecut, E0=E0, tau=d)

    popt, pcov = curve_fit(
        funct_mk501_inside,
        ydata=mkr501_flux[:, 1],
        xdata=mkr501_flux[:, 0],
        p0=[152, -2, 8.]
    )
    nn_array.append(popt)

    integral_kernel = funct_mk501_inside(xxx_means.value,
                                         popt[0], popt[1], popt[2])

    integral_kernel = integral_kernel * 1e-12 * u.erg * u.cm**-2 * u.s**-1
    integral_kernel = (integral_kernel / xxx_means**2.).to(
        u.TeV**-1 * u.cm**-2 * u.s**-1)

    integral_kernel *= (10**spline_eff_area(x=np.log10(xxx_means.value))
                        * u.cm ** 2)

    integral_kernel = integral_kernel * np.log(10) * xxx_means
    integral_kernel = integral_kernel.to(1 / u.s)[:, np.newaxis]

    integral_kernel = integral_kernel * edisp_obj._pdf_matrix

    integral_kernel = simpson(y=integral_kernel,
                              x=np.log10(xxx_means.value), axis=0)

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
    plt.figure(fig_counts)
    plt.scatter(energy_means_obs, count_number)

    plt.figure(fig_spectrum)
    plt.loglog(
        energy_means_obs,
        funct_mk501(energy_means_obs.value,
                    N=popt[0], Ecut=popt[1], E0=popt[2], tau=d))

    likelihoods_array.append(
        likelihood_poisson(count_number, count_number_best))

plt.figure(fig_counts)
plt.scatter(energy_means_obs, count_number_best, marker='x',
            zorder=1e3, s=200)
plt.ylabel('count number')
plt.xlabel('E [TeV]')

plt.figure(fig_spectrum)
plt.loglog(energy_means_obs,
            funct_mk501(energy_means_obs.value,
                        fit_mkr501[0][0], fit_mkr501[0][1],
                        fit_mkr501[0][2], fit_mkr501[0][3]),
            marker='x', ms=20,
            zorder=1e3)
plt.ylabel('flux')
plt.xlabel('E [TeV]')

plt.figure()
plt.plot(alpha_array, likelihoods_array, marker='.')
plt.ylabel('poisson likelihood')
plt.xlabel(r'alpha')

plt.figure()
plt.plot(alpha_array, nn_array)

plt.ylabel('N param')
plt.xlabel(r'alpha')


plt.figure()
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             # yerr=mkr501_flux[:, 2],
             ls='', marker='o')

where_alphamin = np.argmax(likelihoods_array)
plt.plot(xxx, funct_mk501(
    xxx, N=nn_array[where_alphamin][0], Ecut=nn_array[where_alphamin][1],
    E0=nn_array[where_alphamin][2], tau=alpha_array[where_alphamin]))

plt.plot(xxx, funct_mk501(xxx, fit_mkr501[0][0], fit_mkr501[0][1],
                          fit_mkr501[0][2], fit_mkr501[0][3],), ls=':')
plt.plot(xxx, funct_mk501(xxx, fit_mkr501_norm[0][0],
                          -2.03, 8.21, 1.31), ls='--')

plt.title(('mk501 with minimum likelihood alpha=%.2f  \narr='
          % alpha_array[where_alphamin]) + str(nn_array[where_alphamin]))
plt.ylabel('mk501 spectrum')
plt.xlabel('E [TeV]')

plt.yscale('log')
plt.xscale('log')

fig_counts = plt.figure()
fig_spectrum = plt.figure()
alpha_array = ebl_finke.get_models()
nn_array = []
likelihoods_array = []

for d in alpha_array:
    print(d)
    ebl_finke = OptDepth.readmodel(model=d)
    def funct_mk501_inside(ee_array, N, Ecut, E0, tau):
        opacity = ebl_finke.opt_depth(zz, ee_array)
        return (funct_mk501(ee_array, N=N, Ecut=Ecut, E0=E0, tau=tau)
                * np.exp(-opacity))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        0.1*mkr501_flux[:, 1], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               N=152., Ecut=-2.03, E0=8.21, tau=1.31)
    # m.limits = [[0., 5.], [0., 10.], [0., 10.], [0., 10.]]
    # m.fixed[0] = True
    # m.fixed[1] = True
    # m.fixed[2] = True
    # m.fixed[3] = True

    m.migrad()  # finds minimum of least_squares function
    m.hesse()  # accurately

    print(m.params)
    popt = [*m.values]
    nn_array.append(popt)

    integral_kernel = funct_mk501_inside(
        xxx_means.value, popt[0], popt[1], popt[2], popt[3])

    integral_kernel = integral_kernel * 1e-12 * u.erg * u.cm**-2 * u.s**-1
    integral_kernel = (integral_kernel / xxx_means**2.).to(
        u.TeV**-1 * u.cm**-2 * u.s**-1)

    integral_kernel *= (10**spline_eff_area(x=np.log10(xxx_means.value))
                        * u.cm ** 2)

    integral_kernel = integral_kernel * np.log(10) * xxx_means
    integral_kernel = integral_kernel.to(1 / u.s)[:, np.newaxis]

    integral_kernel = integral_kernel * edisp_obj._pdf_matrix

    integral_kernel = simpson(y=integral_kernel,
                              x=np.log10(xxx_means.value), axis=0)

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
    plt.figure(fig_counts)
    plt.scatter(energy_means_obs, count_number)

    plt.figure(fig_spectrum)
    plt.loglog(
        energy_means_obs,
        funct_mk501_inside(energy_means_obs.value,
            N=popt[0], Ecut=popt[1], E0=popt[2], tau=popt[3]))

    likelihoods_array.append(
        likelihood_poisson(count_number, count_number_best))

plt.figure(fig_counts)
plt.scatter(energy_means_obs, count_number_best, marker='x',
            zorder=1e3, s=200)
plt.ylabel('count number')
plt.xlabel('E [TeV]')

plt.figure(fig_spectrum)
plt.loglog(energy_means_obs,
            funct_mk501(energy_means_obs.value,
                        fit_mkr501[0][0], fit_mkr501[0][1],
                        fit_mkr501[0][2], fit_mkr501[0][3]),
            marker='x', ms=20,
            zorder=1e3)
plt.ylabel('flux')
plt.xlabel('E [TeV]')

plt.figure(figsize=(10, 8))
plt.plot(alpha_array, likelihoods_array, marker='.')
plt.ylabel('poisson likelihood')
plt.xlabel(r'ebl model')
plt.gca().tick_params(axis='x', labelrotation=45)

plt.savefig('likelihoods_models.png',
                bbox_inches='tight')

plt.show()



