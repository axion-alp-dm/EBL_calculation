import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from astropy import units as u
import astropy.constants as c
from astropy.constants import h as h_plank

from ebl_codes.EBL_class import EBL_model
from ebltable.ebl_from_model import EBL
from astropy.cosmology import FlatLambdaCDM

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

cosmic_max_axion = np.loadtxt(
    'outputs/figures_paper/cuba_cosmic_constraints.txt')
cosmic_max_axion = UnivariateSpline(
    cosmic_max_axion[:, 0], cosmic_max_axion[:, 1], k=1, s=0)

h = 0.7
omegaM = 0.3
omegaBar = 0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                      Ob0=omegaBar, Tcmb0=2.7255)


def cosmic_axion_contr(lmbd, zz, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = (c.c.value / lmbd * 1e6 * u.s ** -1)[:, np.newaxis]

    z_star = (axion_mass / (2. * h_plank.to(u.eV * u.s) * freq)
              - 1.)

    ebl_axion_cube = (
            ((cosmo.Odm(0.) * cosmo.critical_density0
              * c.c ** 3. / (64. * np.pi * u.sr)
              * axion_gayy ** 2. * axion_mass ** 2.
              * freq
              / cosmo.H(z_star)
              ).to(u.nW * u.m ** -2 * u.sr ** -1)
             ).value
            * (z_star > zz))  # [np.newaxis, :]))

    return ebl_axion_cube


# D_factor = 2.20656e22 * u.GeV * u.cm ** -2
# def host_axion_contr(xx, mass, gay, v_dispersion=220.):
#     sigma = (2. * 2.48 / mass
#              * (v_dispersion * u.km * u.s ** -1 / c.c).to(1))
#     nuInu_values = (
#             14.53 * mass ** 3. * (gay * 1e10) ** 2.
#             * (D_factor / (1.11e22 * u.GeV * u.cm ** -2)).to(1)
#             * (v_dispersion / 220.) ** -1
#             * np.exp(-0.5 * ((xx - 2.48 / mass) / sigma) ** 2.))
#
#     return nuInu_values


ebl_finke = EBL.readmodel('finke2022')
ebl_cuba1 = EBL.readmodel('cuba')

z = np.array([0.3, 0.4, 0.5, 0.65, 0.9])
lmu = np.logspace(-2, 1.5, int(1e3))
ETeV = np.logspace(-3, 1, int(50))

cuba_spline = RectBivariateSpline(
    x=ebl_cuba1.x, y=ebl_cuba1.y, z=ebl_cuba1.Z,
    kx=1, ky=1)

cuba_new = np.zeros((len(lmu), len(ebl_cuba1.y)))

for ni, ii in enumerate(ebl_cuba1.y):
    cuba_new[:, ni:ni + 1] = cuba_spline(np.log10(lmu), ii)

ebl_cuba = EBL(z=ebl_cuba1.y, lmu=lmu,
               nuInu=10 ** cuba_new, model='cuba')

mass_1 = 20.
gayy_1 = cosmic_max_axion(mass_1)

yyy = 10 ** ebl_cuba.Z + (
    cosmic_axion_contr(
        lmbd=10 ** ebl_cuba.x, zz=ebl_cuba.y,
        mass=mass_1, gayy=gayy_1)
)

ebl_our = EBL(z=ebl_cuba.y, lmu=10 ** ebl_cuba.x,
              nuInu=yyy, model='cuba+gaussian')

nuInu_finke = ebl_finke.ebl_array(z, lmu)
nuInu_cuba = ebl_cuba.ebl_array(z, lmu)
nuInu_our = ebl_our.ebl_array(z, lmu)

# --------------------------------------------------------------------
plt.figure()

for i, zz in enumerate(z):
    arrzz = np.zeros((1))
    arrzz[0] = zz

    plt.loglog(lmu, nuInu_finke[i],
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)
    plt.loglog(lmu, nuInu_cuba[i]
               + cosmic_axion_contr(
        lmbd=lmu, zz=arrzz,
        mass=mass_1, gayy=gayy_1).T[0, :]
               ,
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               zorder=-1 * i)

    plt.loglog(lmu, nuInu_our[i],
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               marker='x',
               zorder=-1 * i)

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
# plt.figure()

fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(26, 8))
plt.subplot(131)

linesss = ['-', '--', 'dotted']
mass_array = [30, 20, 10]

for mi, mass_ii in enumerate(mass_array):
    mass_1 = mass_ii
    gayy_1 = cosmic_max_axion(mass_1)

    yyy = 10 ** ebl_cuba.Z + (
        cosmic_axion_contr(
            lmbd=10 ** ebl_cuba.x, zz=ebl_cuba.y,
            mass=mass_1, gayy=gayy_1)
    )

    ebl_our = EBL(z=ebl_cuba.y, lmu=10 ** ebl_cuba.x,
                  nuInu=yyy, model='cuba+gaussian')

    if mi == 0:
        for i, zz in enumerate(z):
            plt.plot(ETeV,
                     (np.exp(ebl_cuba.optical_depth(zz,
                                                    ETeV)-ebl_our.optical_depth(zz, ETeV)))
                      # / np.exp(-))
                     ,
                     ls=linesss[mi],
                     color=plt.cm.CMRmap(i / float(len(z))),
                     label='$z = {0:.2f}$'.format(zz),
                     lw=2)
            print(ebl_cuba.optical_depth(zz, ETeV[-2:])
                  , ebl_our.optical_depth(zz, ETeV[-2:]))
            print(
                  ebl_cuba.optical_depth(zz, ETeV[-2:])
                  - ebl_our.optical_depth(zz, ETeV[-2:]))
            print((np.exp(-ebl_our.optical_depth(zz, ETeV[-2:]))
                      / np.exp(-ebl_cuba.optical_depth(zz, ETeV[-2:]))))
            print()
    else:
        for i, zz in enumerate(z):
            plt.plot(ETeV,
                     (np.exp(ebl_cuba.optical_depth(zz,
                                                    ETeV)-ebl_our.optical_depth(zz, ETeV)))
                      # / np.exp(-))
                     ,
                     ls=linesss[mi],
                     color=plt.cm.CMRmap(i / float(len(z))),
                     lw=2)

# plt.gca().set_ylim((1e-2, 15.))
# plt.gca().set_xlim((1e-1, 1e1))
plt.gca().set_xlabel('Energy (TeV)', size='x-large')
plt.gca().set_ylabel(r'Attenuation ratio axion/CUBA',
                     size='x-large')

plt.xscale('log')
aaa = plt.legend(loc=3)
bbb = plt.legend([plt.Line2D([], [], linestyle=linesss[i],
                             color='k')
                  for i in range(3)],
                 ['%i eV, %.2e GeV-1'
                  % (mass_array[i], cosmic_max_axion(mass_array[i]))
                  for i in range(3)],
                 loc=6, fontsize=12, framealpha=0.4,
                 title='Axion params')
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)
# plt.show()
# --------------------------------------------------------------------
# plt.figure()
plt.subplot(132)
for i, zz in enumerate(z):
    plt.loglog(ETeV, np.exp(-ebl_finke.optical_depth(zz, ETeV)),
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               label='$z = {0:.2f}$'.format(zz), lw=2)
    plt.loglog(ETeV,
               np.exp(-ebl_cuba.optical_depth(zz, ETeV)),
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.2f}$'.format(zz),
               lw=2)
    plt.loglog(ETeV, np.exp(-ebl_our.optical_depth(zz, ETeV)),
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.2f}$'.format(zz),
               lw=2)
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
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               label='$z = {0:.2f}$'.format(zz), lw=2)
    plt.loglog(ETeV,
               ebl_cuba.optical_depth(zz, ETeV),
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.2f}$'.format(zz),
               lw=2)
    plt.loglog(ETeV, ebl_our.optical_depth(zz, ETeV),
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.2f}$'.format(zz),
               lw=2)

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
