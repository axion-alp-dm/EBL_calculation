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

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

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

    if len(zz) > 1:
        cube = np.ones((len(lmbd), len(zz)))
        freq = cube * (c.c.value / lmbd * 1e6 * u.s ** -1)
        zz = cube * zz[np.newaxis, :]

    z_star = (axion_mass / (2. * c.h.to(u.eV * u.s) * freq) * (1. + zz) - 1.)

    ebl_axion_cube = ((cosmo.Odm(0.) * cosmo.critical_density0
                       * c.c ** 3. / (64. * np.pi * u.sr)
                       * axion_gayy ** 2. * axion_mass ** 2.
                       * freq
                       / cosmo.H(z_star)
                       ).to(u.nW * u.m ** -2 * u.sr ** -1)
                      ).value

    ebl_axion_cube = ebl_axion_cube * (z_star > zz)

    return ebl_axion_cube


ebl_finke = EBL.readmodel('finke2022')
ebl_fit = EBL.readascii(
    file_name='outputs/outputs_dust_reem_wto_LOWdatapoints '
              '2024-10-30 10:04:08/SB99_dustFinke3e10.txt',
    model_name='Fit')

z = np.array([0., 0.15, 0.3, 0.65])
lmu = np.logspace(-2, 3., int(1e3))
ETeV = np.geomspace(1e-3, 30, int(50))

# --------------------------------------------------------------------
plt.subplots(1, 2, figsize=(20, 8))
plt.subplot(121)

mass_1 = 10.
gayy_1 = cosmic_max_axion(mass_1)

for i, zz in enumerate(z):

    plt.loglog(lmu, ebl_finke.ebl_array(zz, lmu),
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)

    plt.loglog(lmu, ebl_fit.ebl_array(zz, lmu),
               ls=':', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               zorder=-1 * i)


plt.gca().set_ylim((0.5, 50.))
plt.gca().set_xlim((1e-1, 1000))
plt.gca().set_xlabel('Wavelength ($\mu$m)')
plt.gca().set_ylabel(
    r'$\nu I_\nu (\mathrm{nW}\,\mathrm{sr}^{-1}\mathrm{m}^{-2})$')
aaa = plt.legend(loc='lower center', ncol=2)

markers = ['-', 'dotted']
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(2)],
                 ['Finke22', 'Our fit'],
                 loc=1, fontsize=16, framealpha=0.4)

plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

plt.subplot(122)

for i, zz in enumerate(z):
    plt.loglog(ETeV, ebl_fit.optical_depth(zz, ETeV),
               ls='dotted',
                 color=plt.cm.CMRmap(i / float(len(z))),
                 # color=colors[mi],
               lw=2)
    plt.loglog(ETeV, ebl_finke.optical_depth(zz, ETeV),
               ls='-',
                 color=plt.cm.CMRmap(i / float(len(z))),
                 # color=colors[mi],
               label='$z = {0:.2f}$'.format(zz), lw=2)

plt.gca().set_ylim((4e-8, 800.))
plt.gca().set_xlim((1e-3, 30.))
plt.gca().set_xlabel('Energy (TeV)')
plt.gca().set_ylabel(r'Optical depth $\tau$')
aaa = plt.legend(loc=2)
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(2)],
                 ['Finke22', 'Our fit'],
                 loc=4, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

# plt.show()

# --------------------------------------------------------------------

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(26, 8))


linesss = ['-', '--', 'dotted']
mass_array = [10]#, 20, 10]
colors = ['k', 'r', 'b']

for mi, mass_ii in enumerate(mass_array):
    mass_1 = mass_ii
    gayy_1 = cosmic_max_axion(mass_1)

    yyy = 10 ** ebl_finke.Z + (
        cosmic_axion_contr(
            lmbd=10 ** ebl_finke.x[:, np.newaxis], zz=ebl_finke.y,
            mass=mass_1, gayy=gayy_1)
    )

    ebl_our = EBL(z=ebl_finke.y, lmu=10 ** ebl_finke.x,
                  nuInu=yyy, model='finke2022+gaussian')

    ebl_inside = EBL_with_axion.readmodel(
        model='finke2022', axion_mass=mass_1, axion_gayy=gayy_1)

    ebl_before = EBL_before.readmodel(
        model='finke2022', axion_mass=mass_1, axion_gayy=gayy_1)

    plt.figure()
    for i, zz in enumerate(z):
        plt.loglog(lmu, ebl_our.ebl_array(zz, lmu),
                   ls='-', color=plt.cm.CMRmap(i / float(len(z))),
                   lw=2.,
                   label='$z = {0:.2f}$'.format(zz),
                   zorder=-1 * i)

        plt.loglog(lmu, ebl_inside.ebl_array(zz, lmu),
                   ls=':', color=plt.cm.CMRmap(i / float(len(z))),
                   lw=2.,
                   zorder=-1 * i)

        plt.loglog(lmu, ebl_before.ebl_array(zz, lmu),
                   ls='-.', color=plt.cm.CMRmap(i / float(len(z))),
                   lw=2.,
                   zorder=-1 * i)

    plt.gca().set_ylim((0.5, 50.))
    plt.gca().set_xlim((1e-1, 1000))
    plt.gca().set_xlabel('Wavelength ($\mu$m)')
    plt.gca().set_ylabel(
        r'$\nu I_\nu (\mathrm{nW}\,\mathrm{sr}^{-1}\mathrm{m}^{-2})$')
    aaa = plt.legend(loc='lower center', ncol=2)

    markers = ['-', 'dotted']
    bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                                 color='k')
                      for i in range(2)],
                     ['Finke22', 'Our fit'],
                     loc=1, fontsize=16, framealpha=0.4)

    plt.gca().add_artist(aaa)

    plt.figure(fig)
    plt.subplot(131)

    if mi == 0:
        for i, zz in enumerate(z):
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_our.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='o',
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     label='$z = {0:.2f}$'.format(zz),
                     lw=2)
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_inside.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='+', ms=20,
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     lw=2)

            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_before.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='>', ms=10,
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     lw=2)

    else:
        for i, zz in enumerate(z):
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_our.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='o',
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     lw=2)
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_inside.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='+', ms=20,
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     lw=2)
            plt.plot(ETeV,
                     (np.exp(ebl_finke.optical_depth(zz, ETeV)
                             - ebl_before.optical_depth(zz, ETeV))),
                     ls=linesss[mi], marker='>', ms=10,
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
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
                   ls='-',
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
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
    plt.gca().set_xlim((1e-3, 30.))
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
