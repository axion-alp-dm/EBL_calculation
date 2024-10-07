import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from astropy import units as u
import astropy.constants as c

from astropy.cosmology import Planck15 as cosmo

from ebltable.ebl_from_model2 import EBL as EBL_before
from ebltable.ebl_from_model import EBL
from ebltable_wrapper import EBL_with_axion

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

cosmic_max_axion = np.loadtxt(
    'outputs/figures_paper/cuba_cosmic_constraints.txt')
cosmic_max_axion = UnivariateSpline(
    cosmic_max_axion[:, 0], cosmic_max_axion[:, 1], k=1, s=0)


def cosmic_axion_contr(lmbd, zz, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = (c.c.value / lmbd * 1e6 * u.s ** -1)

    z_star = axion_mass / (2. * c.h.to(u.eV * u.s) * freq) - 1.

    ebl_axion_cube = ((cosmo.Odm(0.) * cosmo.critical_density0
                       * c.c ** 3. / (64. * np.pi * u.sr)
                       * axion_gayy ** 2. * axion_mass ** 2.
                       * freq
                       / cosmo.H(z_star)
                       ).to(u.nW * u.m ** -2 * u.sr ** -1)
                      ).value

    if type(zz) == np.float64 or len(zz) == 1:
        ebl_axion_cube = ebl_axion_cube * (z_star > zz)
    else:
        ebl_axion_cube = (ebl_axion_cube[np.newaxis, :]
                          * (z_star > zz[:, np.newaxis]))

    return ebl_axion_cube


ebl_finke = EBL.readmodel('finke2022')
# ebl_cuba = EBL.readmodel('finke2022')

z = np.array([0.9])#0.3, 0.4, 0.5, 0.65, 0.9])
lmu = np.logspace(-2, 1.5, int(1e3))
ETeV = np.logspace(-3, 1, int(50))

# --------------------------------------------------------------------
plt.figure()

mass_1 = 10.
gayy_1 = cosmic_max_axion(mass_1)

# yyy = 10 ** ebl_cuba.Z + (
#     cosmic_axion_contr(
#         lmbd=10 ** ebl_cuba.x, zz=ebl_cuba.y,
#         mass=mass_1, gayy=gayy_1).T
# )
#
# ebl_our2 = EBL(z=ebl_cuba.y, lmu=10 ** ebl_cuba.x, nuInu=yyy,
#                model='cuba+cosmic out')

ebl_our_inside = EBL_with_axion.readmodel(
    model='finke2022', axion_gayy=gayy_1, axion_mass=mass_1)

ebl_before = EBL_before.readmodel(model='finke2022',
              axion_mass=10., axion_gayy=1e-10
              )

for i, zz in enumerate(z):
    plt.loglog(lmu, cosmic_axion_contr(lmbd=lmu, zz=zz,
                                    mass=mass_1, gayy=gayy_1),
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2., marker='*',
               zorder=-1 * i)
    plt.loglog(lmu, ebl_finke.ebl_array(zz, lmu),
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)
    # plt.loglog(lmu, ebl_cuba.ebl_array(zz, lmu),
    #            ls='-', color=plt.cm.CMRmap(i / float(len(z))),
    #            lw=2.,
    #            zorder=-1 * i)
    # plt.loglog(lmu, ebl_cuba.ebl_array(zz, lmu)
    #            + cosmic_axion_contr(lmbd=lmu, zz=zz,
    #                                 mass=mass_1, gayy=gayy_1),
    #            ls='-', color=plt.cm.CMRmap(i / float(len(z))),
    #            lw=2.,
    #            zorder=-1 * i)

    plt.loglog(lmu, ebl_our_inside.ebl_array(zz, lmu),
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               marker='x',
               zorder=-1 * i)

    # plt.loglog(lmu, ebl_our2.ebl_array(zz, lmu),
    #            ls='--', color=plt.cm.CMRmap(i / float(len(z))),
    #            lw=2.,
    #            marker='o',
    #            zorder=-1 * i)

plt.gca().set_ylim((2e-5, 15.))
plt.gca().set_xlim((1e-2, 30))
plt.gca().set_xlabel('Wavelength ($\mu$m)', size='x-large')
plt.gca().set_ylabel(
    r'$\nu I_\nu (\mathrm{nW}\,\mathrm{sr}^{-1}\mathrm{m}^{-2})$',
    size='x-large')
aaa = plt.legend(loc='lower center', ncol=2)

markers = ['dotted', '-', '--']
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(3)],
                 ['Finke22', 'CUBA', 'CUBA + cosmic'],
                 loc=1, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)
# plt.show()

# --------------------------------------------------------------------

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(26, 8))


linesss = ['-', '--', 'dotted']
mass_array = [30]#, 20, 10]
colors = ['k', 'r', 'b']

for mi, mass_ii in enumerate(mass_array):
    mass_1 = mass_ii
    gayy_1 = cosmic_max_axion(mass_1)

    yyy = 10 ** ebl_finke.Z + (
        cosmic_axion_contr(
            lmbd=10 ** ebl_finke.x, zz=ebl_finke.y,
            mass=mass_1, gayy=gayy_1).T
    )

    ebl_our = EBL(z=ebl_finke.y, lmu=10 ** ebl_finke.x,
                  nuInu=yyy, model='finke2022+gaussian')

    ebl_inside = EBL_with_axion.readmodel(
        model='finke2022', axion_mass=mass_1, axion_gayy=gayy_1)

    ebl_before = EBL_before.readmodel(
        model='finke2022', axion_mass=mass_1, axion_gayy=gayy_1)

    plt.subplot(131)

    if mi == 0:
        for i, zz in enumerate(z):
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_our.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='o',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                     label='$z = {0:.2f}$'.format(zz),
                     lw=2)
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_inside.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='+', ms=20,
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                     lw=2)

            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_before.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='>', ms=10,
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                     lw=2)
            print(ebl_finke.optical_depth(zz, ETeV[-2:])
                  , ebl_our.optical_depth(zz, ETeV[-2:]))
            print(
                ebl_finke.optical_depth(zz, ETeV[-2:])
                - ebl_our.optical_depth(zz, ETeV[-2:]))
            print((np.exp(-ebl_our.optical_depth(zz, ETeV[-2:]))
                   / np.exp(-ebl_finke.optical_depth(zz, ETeV[-2:]))))
            print()
    else:
        for i, zz in enumerate(z):
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_our.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='o',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                     lw=2)
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_inside.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='+', ms=20,
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                     lw=2)
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_before.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='>', ms=10,
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                     lw=2)

    # plt.gca().set_ylim((1e-2, 15.))
    # plt.gca().set_xlim((1e-1, 1e1))
    plt.gca().set_xlabel('Energy (TeV)', size='x-large')
    plt.gca().set_ylabel(r'Attenuation ratio axion/CUBA',
                         size='x-large')

    plt.xscale('log')
    aaa = plt.legend(loc=3)
    # bbb = plt.legend([plt.Line2D([], [], linestyle=linesss[i],
    #                              color='k')
    #                   for i in range(3)],
    #                  ['%i eV, %.2e GeV-1'
    #                   % (mass_array[i], cosmic_max_axion(mass_array[i]))
    #                   for i in range(3)],
    #                  loc=6, fontsize=12, framealpha=0.4,
    #                  title='Axion params')
    # plt.gca().add_artist(aaa)
    # plt.gca().add_artist(bbb)
    # plt.show()
    # --------------------------------------------------------------------
    # plt.figure()
    plt.subplot(132)
    for i, zz in enumerate(z):
        plt.loglog(ETeV, np.exp(-ebl_finke.optical_depth(zz, ETeV)),
                   ls='dotted',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   label='$z = {0:.2f}$'.format(zz), lw=2)

        plt.loglog(ETeV, np.exp(-ebl_inside.optical_depth(zz, ETeV)),
                   ls='--', marker='+',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   lw=2, ms=20)
        plt.loglog(ETeV, np.exp(-ebl_our.optical_depth(zz, ETeV)),
                   ls='--',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   lw=2, marker='o')
        plt.loglog(ETeV, np.exp(-ebl_before.optical_depth(zz, ETeV)),
                   ls='--',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   lw=2, marker='>')
        # plt.axvline(Etau1GeV[i] / 1e3, ls=':', color = plt.cm.CMRmap(i / float(len(z))) )

    plt.gca().set_ylim((1e-4, 2.))
    plt.gca().set_xlim((1e-3, 1e1))
    plt.gca().set_xlabel('Energy (TeV)', size='x-large')
    plt.gca().set_ylabel(r'Attenuation $exp(-\tau)$', size='x-large')
    aaa = plt.legend(loc=3)
    markers = ['dotted', '-', '--']
    bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                                 color='k')
                      for i in range(3)],
                     ['Finke22', 'CUBA', 'CUBA + cosmic'],
                     loc=4, fontsize=16, framealpha=0.4)
    plt.gca().add_artist(aaa)
    plt.gca().add_artist(bbb)

    # --------------------------------------------------------------------
    # plt.figure()
    plt.subplot(133)
    for i, zz in enumerate(z):
        plt.loglog(ETeV, ebl_finke.optical_depth(zz, ETeV),
                   ls='dotted',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   label='$z = {0:.2f}$'.format(zz), lw=2)

        plt.loglog(ETeV, ebl_inside.optical_depth(zz, ETeV),
                   ls='--',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   # label = '$z = {0:.2f}$'.format(zz),
                   lw=2, marker='+', ms=20)
        plt.loglog(ETeV, ebl_our.optical_depth(zz, ETeV),
                   ls='--',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   # label = '$z = {0:.2f}$'.format(zz),
                   lw=2, marker='o')
        plt.loglog(ETeV, ebl_before.optical_depth(zz, ETeV),
                   ls='--',
                     # color=plt.cm.CMRmap(i / float(len(z))),
                     color=colors[mi],
                   # label = '$z = {0:.2f}$'.format(zz),
                   lw=2, marker='>')

    plt.gca().set_ylim((4e-8, 800.))
    plt.gca().set_xlim((1e-3, 1e1))
    plt.gca().set_xlabel('Energy (TeV)', size='x-large')
    plt.gca().set_ylabel(r'Optical depth $\tau$', size='x-large')
    aaa = plt.legend(loc=2)
    markers = ['dotted', '-', '--']
    bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                                 color='k')
                      for i in range(3)],
                     ['Finke22', 'CUBA', 'CUBA + cosmic'],
                     loc=4, fontsize=16, framealpha=0.4)
    plt.gca().add_artist(aaa)
    plt.gca().add_artist(bbb)

plt.savefig('outputs/figures_paper/opacities_test.png',
            bbox_inches='tight')

plt.show()
