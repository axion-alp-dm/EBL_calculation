import os
import time
import yaml

import iminuit.cost
import numpy as np
import scipy.optimize
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import astropy.units as u

from scripts.e_disp_code import EDispGauss

from iminuit import Minuit
from iminuit.cost import LeastSquares

from ebltable.ebl_from_model import EBL
from ebltable.tau_from_model import OptDepth

os.chdir("..")
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
print(os.getcwd())
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
mkr501_flux[:, 1] *= (u.erg.to(u.TeV))
mkr501_flux[:, 2] *= (u.erg.to(u.TeV))

zz = 0.034

e_array = np.geomspace(
    np.min(mkr501_flux[:, 0]) * 0.5,
    np.max(mkr501_flux[:, 0]) * 1.5,
    num=500)

my_ebl = ['bosa.txt', 'chary.txt', '2bb.txt']

opacities_array = {}
for d in my_ebl:
    ebl_finke = EBL.readascii(
            'outputs/lhaaso_syst10percent/' + d, model_name='mine')
    opacityy = ebl_finke.optical_depth(z0=zz, ETeV=e_array)

    # ebl_finke = OptDepth.readmodel('finke2022')
    # opacityy = ebl_finke.opt_depth(z=zz, ETeV=e_array)

    opacities_array[d] = UnivariateSpline(
            np.log10(e_array), opacityy, k=1, s=0)


plt.subplots(3, 2, sharex=True, figsize=(10, 12))
plt.subplots_adjust(hspace=0)

for nn, d in enumerate(my_ebl):
    print('\n\n')
    print(d)
    opacity = opacities_array[d]

    # ----------------------------------------------------------------------
    print('PL + EBL')
    def funct_mk501(ee_array, N, gamma):
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity(np.log10(ee_array))))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        N=152., gamma=2.03)

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)
    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
              % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)
    print(m_best_fit.values[0] * (u.erg/u.TeV**2).to(u.TeV**-1))

    plt.subplot(3, 2, 2*(nn+1))
    plt.loglog(e_array, funct_mk501(
        e_array,
        N=m_best_fit.values[0],
        gamma=m_best_fit.values[1]),
               label='PL + EBL')

    plt.subplot(3, 2, 2*(nn+1)-1)
    plt.ylabel(d)
    plt.plot(e_array, funct_mk501(
        e_array,
        N=m_best_fit.values[0],
        gamma=m_best_fit.values[1]),
               label='PL + EBL')

    # -----------------------------------------------------------------------
    print('\nLogParabola + EBL')
    def funct_mk501(ee_array, phi0, E0, alpha, beta):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (
                        - alpha - beta * np.log(ee_array/E0))
                * np.exp(-opacity(np.log10(ee_array))))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        phi0=100, E0=1.12, alpha=2., beta=0.088)

    # m_best_fit.limits['E0'] = (0., 10)
    # m_best_fit.limits['alpha'] = (0., 10)
    # m_best_fit.limits['beta'] = (0., 10)
    # m_best_fit.fixed['E0'] = True
    # m_best_fit.fixed['alpha'] = True
    # m_best_fit.fixed['beta'] = True

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)

    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
          % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
    %(m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(3, 2, 2 * (nn + 1))
    plt.loglog(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        E0=m_best_fit.values[1],
        alpha=m_best_fit.values[2],
        beta=m_best_fit.values[3]),
               label='LogParabola + EBL')

    plt.subplot(3, 2, 2 * (nn + 1) - 1)
    plt.plot(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        E0=m_best_fit.values[1],
        alpha=m_best_fit.values[2],
        beta=m_best_fit.values[3]),
               label='LogParabola + EBL')
    # -----------------------------------------------------------------------
    print('\nBPL + EBL')
    def funct_mk501(xx, N0, gamma1, gamma2, Ebreak, fi):
        return (N0 * xx ** 2.
                * xx ** (-gamma1)
                * (1. + (xx / Ebreak) ** fi) ** ((gamma1 - gamma2) / fi)
                * np.exp(-opacity(np.log10(xx)))
                )


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        N0=200, gamma1=2., gamma2=2.45, Ebreak=2.1, fi=2.)
    m_best_fit.limits['gamma1'] = (0., 10.)
    m_best_fit.limits['gamma2'] = (0., 10.)
    m_best_fit.limits['fi'] = (0., 10.)
    m_best_fit.limits['Ebreak'] = (0., 10.)

    m_best_fit.fixed['fi'] = True

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)
    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
              % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
          % (m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(3, 2, 2 * (nn + 1))
    plt.loglog(e_array, funct_mk501(
        e_array,
        N0=m_best_fit.values[0],
        gamma1=m_best_fit.values[1],
        gamma2=m_best_fit.values[2],
        Ebreak=m_best_fit.values[3],
        fi=2.),
               label='BPL + EBL')

    plt.subplot(3, 2, 2 * (nn + 1) - 1)
    plt.plot(e_array, funct_mk501(
        e_array,
        N0=m_best_fit.values[0],
        gamma1=m_best_fit.values[1],
        gamma2=m_best_fit.values[2],
        Ebreak=m_best_fit.values[3],
        fi=2.),
               label='BPL + EBL')
    # -----------------------------------------------------------------------
    print('\nPLE + EBL')

    def funct_mk501(ee_array, phi0, gamma, E0, Ecut):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (-gamma)
                * np.exp(-(ee_array / Ecut) - opacity(np.log10(ee_array))))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        phi0=137, gamma=1.9, E0=1.12, Ecut=7.)

    # m_best_fit.limits['Ecut'] = (0., 30)
    # m_best_fit.fixed['E0'] = True

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)
    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
              % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
          % (m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(3, 2, 2 * (nn + 1))
    plt.loglog(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        gamma=m_best_fit.values[1],
        E0=m_best_fit.values[2],
        Ecut=m_best_fit.values[3]),
               label='PLE + EBL')

    plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
                 yerr=mkr501_flux[:, 2],
                 ls='', marker='+', label='Data')
    plt.legend()

    plt.yscale('linear')
    plt.ylim(0., 220)

    plt.subplot(3, 2, 2 * (nn + 1) - 1)
    plt.plot(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        gamma=m_best_fit.values[1],
        E0=m_best_fit.values[2],
        Ecut=m_best_fit.values[3]),
               label='PLE + EBL')

    plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
                 yerr=mkr501_flux[:, 2],
                 ls='', marker='+', label='Data')

    plt.yscale('log')
    plt.ylim(1, 220)
    plt.legend()


for nn, d in enumerate(my_ebl):
    print('\n\n')
    print(d)
    opacity = opacities_array[d]

# ----------------------------------------------------------------------
print('PL + EBL')
plt.subplots(2, 1, figsize=(10, 6))
plt.suptitle('PL + EBL')

for nn, d in enumerate(my_ebl):

    opacity = opacities_array[d]

    def funct_mk501(ee_array, N, gamma):
        return (N * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity(np.log10(ee_array))))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        N=152., gamma=2.03)

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)
    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
              % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
          % (m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(1, 2, 1)
    plt.loglog(e_array, funct_mk501(
        e_array,
        N=m_best_fit.values[0],
        gamma=m_best_fit.values[1]),
               label=d)

    plt.legend()
    plt.xscale('log')

    plt.subplot(1, 2, 2)
    plt.ylabel(d)
    plt.plot(e_array, funct_mk501(
        e_array,
        N=m_best_fit.values[0],
        gamma=m_best_fit.values[1]),
               label=d)

plt.subplot(1, 2, 2)

plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
                 yerr=mkr501_flux[:, 2],
                 ls='', marker='+', label='Data')
plt.legend()
plt.yscale('linear')
plt.xscale('log')
plt.ylim(0., 220)

plt.subplot(1, 2, 1)
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='+', label='Data')

plt.yscale('log')
plt.xscale('log')
plt.ylim(1, 220)
plt.legend()
# -----------------------------------------------------------------------
print('\nLogParabola + EBL')

plt.subplots(2, 1, figsize=(10, 6))
plt.suptitle('\nLogParabola + EBL')

for nn, d in enumerate(my_ebl):

    opacity = opacities_array[d]

    def funct_mk501(ee_array, phi0, E0, alpha, beta):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (
                        - alpha - beta * np.log(ee_array/E0))
                * np.exp(-opacity(np.log10(ee_array))))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        phi0=150, E0=1., alpha=5., beta=0.5)

    # m_best_fit.limits['E0'] = (0., 10)
    # m_best_fit.limits['alpha'] = (0., 10)
    # m_best_fit.limits['beta'] = (0., 10)
    m_best_fit.fixed['E0'] = True

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)

    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
          % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
    %(m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(1, 2, 1)
    plt.loglog(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        E0=m_best_fit.values[1],
        alpha=m_best_fit.values[2],
        beta=m_best_fit.values[3]),
               label=d)

    plt.subplot(1, 2, 2)
    plt.plot(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        E0=m_best_fit.values[1],
        alpha=m_best_fit.values[2],
        beta=m_best_fit.values[3]),
               label=d)

plt.subplot(1, 2, 2)

plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
                 yerr=mkr501_flux[:, 2],
                 ls='', marker='+', label='Data')
plt.legend()
plt.yscale('linear')
plt.xscale('log')
plt.ylim(0., 220)

plt.subplot(1, 2, 1)
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='+', label='Data')

plt.yscale('log')
plt.xscale('log')
plt.ylim(1, 220)
plt.legend()

# -----------------------------------------------------------------------
print('\nBPL + EBL')
plt.subplots(2, 1, figsize=(10, 6))
plt.suptitle('\nBPL + EBL')

for nn, d in enumerate(my_ebl):

    opacity = opacities_array[d]

    def funct_mk501(xx, N0, gamma1, gamma2, Ebreak, fi):
        return (N0 * xx ** 2.
                * xx ** (-gamma1)
                * (1. + (xx / Ebreak) ** fi) ** ((gamma1 - gamma2) / fi)
                * np.exp(-opacity(np.log10(xx)))
                )


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        N0=200, gamma1=2., gamma2=2.45, Ebreak=2.1, fi=2.)
    m_best_fit.limits['gamma1'] = (0., 10.)
    m_best_fit.limits['gamma2'] = (0., 10.)
    m_best_fit.limits['fi'] = (0., 10.)
    m_best_fit.limits['Ebreak'] = (0., 10.)

    m_best_fit.fixed['fi'] = True

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)
    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
              % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
          % (m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(1, 2, 1)
    plt.loglog(e_array, funct_mk501(
        e_array,
        N0=m_best_fit.values[0],
        gamma1=m_best_fit.values[1],
        gamma2=m_best_fit.values[2],
        Ebreak=m_best_fit.values[3],
        fi=2.),
               label=d)

    plt.subplot(1, 2, 2)
    plt.plot(e_array, funct_mk501(
        e_array,
        N0=m_best_fit.values[0],
        gamma1=m_best_fit.values[1],
        gamma2=m_best_fit.values[2],
        Ebreak=m_best_fit.values[3],
        fi=2.),
               label=d)

plt.subplot(1, 2, 2)

plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
                 yerr=mkr501_flux[:, 2],
                 ls='', marker='+', label='Data')
plt.legend()
plt.yscale('linear')
plt.ylim(0., 220)
plt.xscale('log')

plt.subplot(1, 2, 1)
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='+', label='Data')

plt.yscale('log')
plt.ylim(1, 220)
plt.xscale('log')
plt.legend()
# -----------------------------------------------------------------------
print('\nPLE + EBL')
plt.subplots(2, 1, figsize=(10, 6))
plt.suptitle('\nPLE + EBL')

for nn, d in enumerate(my_ebl):

    opacity = opacities_array[d]

    def funct_mk501(ee_array, phi0, gamma, E0, Ecut):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (-gamma)
                * np.exp(-(ee_array / Ecut) - opacity(np.log10(ee_array))))


    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501)

    m_best_fit = Minuit(combined_likelihood,
                        phi0=137, gamma=2.5, E0=1., Ecut=19.)

    # m_best_fit.limits['Ecut'] = (0., 30)
    m_best_fit.fixed['E0'] = True

    m_best_fit.migrad()
    m_best_fit.hesse()

    print(m_best_fit.params)
    ss = ''
    for i in range(m_best_fit.npar):
        ss = (ss + r' & $ %.5f \pm %.5f $'
              % (m_best_fit.values[i], m_best_fit.errors[i]))
    print(ss)

    print('$%.3f / %i = %.2f $'
          % (m_best_fit.fval, m_best_fit.ndof, m_best_fit.fmin.reduced_chi2))

    plt.subplot(1, 2, 2)
    plt.loglog(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        gamma=m_best_fit.values[1],
        E0=m_best_fit.values[2],
        Ecut=m_best_fit.values[3]),
               label=d)

    plt.subplot(1, 2, 1)
    plt.plot(e_array, funct_mk501(
        e_array,
        phi0=m_best_fit.values[0],
        gamma=m_best_fit.values[1],
        E0=m_best_fit.values[2],
        Ecut=m_best_fit.values[3]),
               label=d)

plt.subplot(1, 2, 2)

plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
                 yerr=mkr501_flux[:, 2],
                 ls='', marker='+', label='Data')
plt.legend()
plt.yscale('linear')
plt.xscale('log')
plt.ylim(0., 220)

plt.subplot(1, 2, 1)
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='+', label='Data')

plt.yscale('log')
plt.ylim(1, 220)
plt.xscale('log')
plt.legend()


plt.show()
