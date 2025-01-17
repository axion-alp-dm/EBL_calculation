import os
import numpy as np
import scipy.optimize
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import astropy.units as u

from e_disp_code import EDispGauss

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
mkr501_flux = np.loadtxt('data/lhasso_characteristics/mkr501.txt')
plt.scatter(mkr501_flux[:, 0], mkr501_flux[:, 1])
plt.yscale('log')

from ebltable.tau_from_model import OptDepth
from scipy.optimize import curve_fit
ebl_finke = OptDepth.readmodel(model='finke2022')
zz = 0.034
# def funct_mk501(ee_array, N, E0, tau):
#     return N * (ee_array/E0)**(-tau)

def funct_mk501(ee_array, N, E0, tau, Ecut, alpha):
    opacity = ebl_finke.opt_depth(zz, ee_array*1e-12)
    return (N * (ee_array/E0)**(-tau)
            * np.exp(#-ee_array/Ecut
                     - alpha * opacity)
            )

fit_mkr501 = curve_fit(funct_mk501, ydata=mkr501_flux[:, 1],
                       xdata=10**mkr501_flux[:, 0],
                       p0=[1e-13, 1e13, 2.57, 1e14, 1.2],
                       bounds=[(0., 0., 0., 0., 0.),
                               (np.inf, np.inf, 10, np.inf, 10.)]
                       )
print(fit_mkr501[0])
print(np.sqrt(np.diag(fit_mkr501[1])))
print(fit_mkr501[1])

fit_powerlaw = np.polyfit(mkr501_flux[:-3, 0],
                          np.log10(mkr501_flux[:-3, 1]), 1)
print(fit_powerlaw)

xxx = np.linspace(0.99*mkr501_flux[0, 0], 1.01*mkr501_flux[-1, 0])

plt.plot(xxx, 10**(fit_powerlaw[1] + xxx*fit_powerlaw[0]),
         ls='--')
plt.plot(xxx, funct_mk501(10**xxx,
                          fit_mkr501[0][0],
                          fit_mkr501[0][1],
                          fit_mkr501[0][2],
                          fit_mkr501[0][3],
                          fit_mkr501[0][4]
                          ))
plt.plot(xxx, funct_mk501(10**xxx,
                          1e-13, 1e13, 2.57, 1e20, 0.2
                          ))

plt.title('Mkr 501 flare 1997')
plt.xlabel('log10(E) [eV]')
plt.ylabel('Flux [photons/TeV/cm2/s]')

# ----------------------------------------------------------------------

xxx_bins = np.linspace(
    mkr501_flux[0, 0] - 0.5,
    mkr501_flux[-1, 0] + 0.5,
    num=1000)
xxx_means = (xxx_bins[1:] + xxx_bins[:-1])/2.
integral_kernel = 10**(fit_powerlaw[1] + xxx_means*fit_powerlaw[0])
integral_kernel /= u.TeV * u.cm**2 * u.s
print(np.max(integral_kernel))

eff_area = np.loadtxt('data/lhasso_characteristics/'
                      'WCD_eff_area_0to15deg.txt')
eff_area = eff_area[np.argsort(eff_area[:, 0]), :]
eff_area [:, 1] = eff_area[:, 1] * (u.m**2).to(u.cm**2)

spline_eff_area = UnivariateSpline(
    x=eff_area[:, 0] + 9., y=eff_area[:, 1], s=0, k=1, ext=1
)

integral_kernel *= spline_eff_area(x=xxx_means) * u.cm**2
print(integral_kernel[0])

edisp_obj = EDispGauss(sigma=0.2, bias=0.)
recovered_bins = xxx_bins[:-2]
recovered_means = xxx_means[:-2]
edisp_obj.fill(e_true_edges=10**xxx_bins,
               e_reco_edges=10**recovered_bins)
plt.figure()
plt.title('e disp matrix we use in the calculations')
edisp_obj.plot()

edisp_obj = EDispGauss(sigma=0.2, bias=0.)
edisp_obj.fill(e_true_edges=10**(xxx_bins-9.),
               e_reco_edges=10**(recovered_bins-9.))
plt.figure()
plt.title('e disp matrix we use in the calculations')
edisp_obj.plot()


integral_kernel *= np.log(10) * 10**xxx_means * u.eV
integral_kernel = integral_kernel.to(1/u.s)[:, np.newaxis]

integral_kernel = integral_kernel * edisp_obj._pdf_matrix

integral_kernel = simpson(y=integral_kernel, x=xxx_means, axis=0)
print(np.max(integral_kernel))

energy_bins_obs = np.linspace(11., 14., num=50)
energy_means_obs = (energy_bins_obs[1:] + energy_bins_obs[:-1])/2.
count_number = []
bbb = 0

integral_kernel *= np.log(10) * 10**recovered_means

for i in range(len(energy_bins_obs) - 1):
    where_bin = ((recovered_means > energy_bins_obs[i])
                 * (recovered_means < energy_bins_obs[i+1]))
    bbb += sum(where_bin)
    if sum(where_bin) > 0:
        count_number.append(simpson(
            y=integral_kernel[where_bin], x=recovered_means[where_bin]))
    else:
        count_number.append(0.)

print(np.shape(recovered_means), bbb)

total_int_time = (110. * u.h).to(u.s)
count_number = (count_number * total_int_time).value

plt.figure()
plt.scatter(energy_means_obs, count_number)
plt.ylabel('count number')
plt.xlabel(r'log$_{10}$(E) [eV]')
print(count_number)

likelihoods_array = []

def likelihood_poisson(mu_i_array):
    mu_i_array[mu_i_array <= 0] = 1e-43
    return sum(mu_i_array * (np.log10(mu_i_array) - 1.))

zz = 0.034

def funct_mk501(ee_array, N, E0, tau, alpha):
    opacity = ebl_finke.opt_depth(zz, ee_array*1e-12)
    return (N * (ee_array/E0)**(-tau)
            * np.exp(- alpha * opacity)
            )


total_int_time = (110. * u.h).to(u.s)

edisp_obj = EDispGauss(sigma=0.2, bias=0.)
edisp_obj.fill(e_true_edges=10 ** (xxx_bins - 9.),
                   e_reco_edges=10 ** (recovered_bins - 9.))

energy_bins_obs = np.linspace(11., 14., num=50)
energy_means_obs = (energy_bins_obs[1:] + energy_bins_obs[:-1]) / 2.

alpha_array = np.geomspace(1e-14, 1e-12, num=25)

plt.figure()

for d in alpha_array:
    print(d)

    def funct_mk501_inside(ee_array, N, E0, tau):
        opacity = ebl_finke.opt_depth(zz, ee_array * 1e-12)
        return (d * (ee_array / E0) ** (-tau) * np.exp(- N * opacity))


    popt, pcov = curve_fit(funct_mk501_inside, ydata=mkr501_flux[:, 1],
                       xdata=10**mkr501_flux[:, 0],
                       p0=[1., 1e13, 2.57],
                       bounds=[(0., 0., 0.),
                               (2., np.inf, 10)])

    integral_kernel = (funct_mk501_inside(10**xxx_means,
                          popt[0],
                          popt[1],
                          popt[2]
                          ))
    integral_kernel /= u.TeV * u.cm ** 2 * u.s

    integral_kernel *= spline_eff_area(x=xxx_means) * u.cm ** 2

    integral_kernel *= np.log(10) * 10 ** xxx_means * u.eV
    integral_kernel = integral_kernel.to(1 / u.s)[:, np.newaxis]

    integral_kernel = integral_kernel * edisp_obj._pdf_matrix

    integral_kernel = simpson(y=integral_kernel, x=xxx_means, axis=0)

    count_number = []

    integral_kernel *= np.log(10) * 10 ** recovered_means

    for i in range(len(energy_bins_obs) - 1):
        where_bin = ((recovered_means > energy_bins_obs[i])
                     * (recovered_means < energy_bins_obs[i + 1]))

        if sum(where_bin) > 0:
            count_number.append(simpson(
                y=integral_kernel[where_bin],
                x=recovered_means[where_bin]))
        else:
            count_number.append(0.)

    count_number = (count_number * total_int_time).value
    plt.scatter(energy_means_obs, count_number)

    likelihoods_array.append(likelihood_poisson(count_number))

plt.ylabel('count number')
plt.xlabel(r'log$_{10}$(E) [eV]')

plt.figure()
plt.plot(alpha_array, likelihoods_array, marker='.')
plt.ylabel('poisson likelihood')
plt.xlabel(r'alpha')



plt.show()



