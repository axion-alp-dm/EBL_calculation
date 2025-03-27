import os

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
import scipy.stats as stats
import matplotlib.pyplot as plt

import astropy.units as u

from iminuit import Minuit
from iminuit.cost import LeastSquares

from ebltable.ebl_from_model import EBL

if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

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
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

# ----------------------------------------------------------------------
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

mkr501_flux = np.concatenate((mkr501_flux_1[:8, :3], mkr501_flux))

zz = 0.034

e_array = np.geomspace(
    np.min(mkr501_flux[:, 0]) * 0.5,
    np.max(mkr501_flux[:, 0]) * 1.5,
    num=500)

# ----------------------------------------------------------------------
xx_array = np.geomspace(0.5, 25.)

plt.figure()
xx_plot = np.geomspace(0.5, 20)


plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2], ls='', marker='o',
             ms=7, zorder=20)


def doublePL(xx, N0, gamma1, gamma2, Ebreak, fi):
    return (N0 * xx ** 2.
            * xx ** (-gamma1)
            * (1. + (xx / Ebreak) ** fi) ** ((gamma1 - gamma2) / fi)
            )
plt.loglog(xx_plot, doublePL(xx_plot, 150, 2.1, 3., 3., 3.), c='k')

my_ebl = ['Chary.txt', 'BOSA.txt', '3_grey_bodies.txt']

colors = {'BOSA.txt': 'tab:blue',
          '3_grey_bodies.txt': 'tab:green',
          'Chary.txt': 'darkorange'}

opacities_array = {}
for d in my_ebl:
    ebl_finke = EBL.readascii(
            'outputs/lhaaso/' + d, model_name='mine')
    opacityy = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
    opacities_array[d] = UnivariateSpline(
            np.log10(e_array), opacityy, k=1, s=0)


    def doublePL(xx, N0, gamma1, gamma2, Ebreak, fi):
        return (N0 * xx ** 2.
                * xx ** (-gamma1)
                * (1. + (xx / Ebreak) ** fi) ** ((gamma1 - gamma2) / fi)
                * np.exp(-opacities_array[d](np.log10(xx))))
    plt.loglog(xx_plot, doublePL(xx_plot, 150, 2.1, 3., 3., 3.), c=colors[d])
plt.show()

fig_chi2, ax_chi2 = plt.subplots()

fig_spectrum, ax_spectrum = plt.subplots(figsize=(8, 8))
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2], ls='', marker='o',
             ms=7, zorder=20)


# plt.show()
print('LogParabola')
for nd, d in enumerate(my_ebl):
    print(d)

    opacity = opacities_array[d]

    def funct_mk501_inside(ee_array, phi0, E0, gamma, beta):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (
                        - gamma - beta * np.log(ee_array / E0))
                * np.exp(-opacity(np.log10(ee_array))))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               phi0=1e-12, E0=800.1, gamma=2., beta=0.)

    m.migrad()
    m.hesse()

    print(m.params)
    print(f"{m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    print(1 - stats.chi2.cdf(x=m.fval, df=m.ndof))

    ax_chi2.axvline(m.fval, c=colors[d])

    ax_spectrum.plot(xx_array,
                     funct_mk501_inside(xx_array, *m.values),
                     label=d + ' fit to obs', c=colors[d],
                     ls='--', lw=2)

xx_plot = np.linspace(0, 150, num=500)
ax_chi2.plot(xx_plot, stats.chi2.pdf(x=xx_plot, df=m.ndof))

print('\nPL + EBL cutoff')
for nd, d in enumerate(my_ebl):
    print(d)

    opacity = opacities_array[d]

    def funct_mk501_inside(ee_array, phi0, gamma):
        return (phi0 * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity(np.log10(ee_array)))
                )

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               phi0=150, gamma=2.)

    m.migrad()
    m.hesse()

    print(f"{m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    print(1 - stats.chi2.cdf(x=m.fval, df=m.ndof))

    # ax_spectrum.plot(xx_array,
    #                  funct_mk501_inside(xx_array, *m.values),
    #                  label=d + ' fit to obs', c=colors[d], lw=2)

print('\nPL + exp cutoff + EBL absorption')
for nd, d in enumerate(my_ebl):
    print(d)

    opacity = opacities_array[d]

    def funct_mk501_inside(ee_array, phi0, gamma, E0, Ecut):
        return (phi0 * ee_array ** 2.
                * (ee_array/E0) ** (-gamma)
                * np.exp(-(ee_array/Ecut) - opacity(np.log10(ee_array)))
                )

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               phi0=120, gamma=2.5, E0=1., Ecut=8.)

    m.migrad()
    m.hesse()

    print(f"{m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    print(1 - stats.chi2.cdf(x=m.fval, df=m.ndof))

    # ax_spectrum.plot(xx_array,
    #                  funct_mk501_inside(xx_array, *m.values),
    #                  label=d + ' fit to obs', c=colors[d],
    #                  ls=':', lw=2)

print('\nPL1 + CPL12 + EBL absorption')
for nd, d in enumerate(my_ebl):
    print(d)

    opacity = opacities_array[d]

    def funct_mk501_inside(xx, N0, gamma1, gamma2, Ebreak, fi):
        return (N0 * xx ** 2.
                * xx ** (-gamma1)
                * (1. + (xx / Ebreak) ** fi) ** ((gamma1 - gamma2) / fi)
                * np.exp(-opacity(np.log10(xx)))
                )

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               N0=200, gamma1=2., gamma2=2.45, Ebreak=2.1, fi=2.)

    m.limits['gamma1'] = (0., 10.)
    m.limits['gamma2'] = (0., 10.)
    m.limits['fi'] = (0., 10.)
    m.limits['Ebreak'] = (1., 10.)

    m.fixed['fi'] = True

    m.migrad()
    m.hesse()

    print(m.params)

    print(f"{m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    print(1 - stats.chi2.cdf(x=m.fval, df=m.ndof))

    ax_spectrum.plot(xx_array,
                     funct_mk501_inside(xx_array, *m.values),
                     label=d + ' fit to obs', c=colors[d],
                     ls='-.', lw=2)

plt.xscale('log')
plt.ylim(bottom=0)
plt.xlabel('E [TeV]')
plt.ylabel('E2dN/dE [10−12 erg cm−2 s−1]')
plt.savefig('outputs/lhaaso/fit_to_hegra_spectrum.png',
            bbox_inches='tight')
plt.show()
