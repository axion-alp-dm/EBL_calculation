import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from astropy import units as u
import astropy.constants as c

from astropy.cosmology import Planck15 as cosmo

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


def cosmic_axion_contr(lmu, zz, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = (c.c.value / lmu * 1e6 * u.s ** -1)

    z_star = (axion_mass / (2. * c.h.to(u.eV * u.s) * freq) * (1. + zz)
              - 1.)

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

input_file_dir = ('outputs/outputs_dust_reem_wto_LOWdatapoints '
              '2024-10-30 10:04:08/')
ebl_finke = EBL.readmodel('finke2022')
ebl_fit = EBL.readascii(
    file_name='outputs/outputs_dust_reem_wto_LOWdatapoints '
              '2024-10-30 10:04:08/SB99_dustFinke3e10.txt',
    model_name='Fit')

z = np.array([0., 0.15, 0.3, 0.65])
lmu = np.logspace(-2, 3., int(1e4))
ETeV = np.geomspace(1e-3, 30, int(100))

# --------------------------------------------------------------------
plt.figure()
freq = (c.c.value / lmu * 1e6 * u.s ** -1)
z_array = np.arange(0., 20., 0.2)
for ni, ii in enumerate(z_array):
    z_star = (5. * u.eV / (2. * c.h.to(u.eV * u.s) * freq) * (1. + ii)
              - 1.)

    plt.loglog(lmu, ii*np.ones(len(lmu)),
             label=ii, color=plt.cm.CMRmap(ni / float(len(z_array))),)
    plt.loglog(lmu[z_star>ii], z_star[z_star>ii],
             label=ii, color=plt.cm.CMRmap(ni / float(len(z_array))),)
# plt.legend()
plt.show()

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
plt.gca().set_xlim((1e-1, 3000))
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

plt.savefig(input_file_dir + 'opt_depth.png',
                bbox_inches='tight')
plt.savefig(input_file_dir + 'opt_depth.pdf',
                bbox_inches='tight')

# plt.show()

# --------------------------------------------------------------------

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(26, 8))

linesss = ['-', '--', 'dotted']
mass_array = [30, 20, 10]
colors = ['k', 'r', 'b']

for mi, mass_ii in enumerate(mass_array):
    mass_1 = mass_ii
    gayy_1 = cosmic_max_axion(mass_1)

    ebl_inside = EBL_with_axion.readmodel(
        model='finke2022', axion_mass=mass_1, axion_gayy=gayy_1)

    opt_depth_finke = np.zeros((len(z), len(ETeV)))
    opt_depth_axion = np.zeros((len(z), len(ETeV)))

    for i, zz in enumerate(z):
        opt_depth_finke[i, :] = ebl_finke.optical_depth(zz, ETeV)
        opt_depth_axion[i, :] = ebl_inside.optical_depth(zz, ETeV)

    plt.figure()
    for i, zz in enumerate(z):
        plt.loglog(lmu, ebl_inside.ebl_array(zz, lmu),
                   ls=':', color=plt.cm.CMRmap(i / float(len(z))),
                   lw=2.,
                   zorder=-1 * i)

    plt.gca().set_ylim((0.5, 50.))
    plt.gca().set_xlim((1e-1, 3000))
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

    for i, zz in enumerate(z):
        if mi == 0:
            plt.plot(ETeV,
                     (np.exp(opt_depth_finke[i, :]
                             - opt_depth_axion[i, :])),
                     ls=linesss[mi],
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     lw=2,
                     label='$z = {0:.2f}$'.format(zz))
        else:
            plt.plot(ETeV,
                     (np.exp(opt_depth_finke[i, :]
                             - opt_depth_axion[i, :])),
                     ls=linesss[mi],
                     color=plt.cm.CMRmap(i / float(len(z))),
                     # color=colors[mi],
                     lw=2)

    # plt.gca().set_ylim((1e-2, 15.))
    plt.gca().set_xlim((8e-3, 3e1))
    plt.gca().set_xlabel('Energy (TeV)', size='x-large')
    plt.gca().set_ylabel(r'Attenuation ratio axion/Finke',
                         size='x-large')

    plt.xscale('log')
    aaa = plt.legend(loc=3)
    bbb = plt.legend([plt.Line2D([], [], linestyle=linesss[i],
                                 color='k')
                      for i in range(len(mass_array))],
                     ['%i eV, %.2e GeV-1'
                      % (mass_array[i], cosmic_max_axion(mass_array[i]))
                      for i in range(3)],
                     loc=6, fontsize=12, framealpha=0.4,
                     title='Axion params')
    plt.gca().add_artist(aaa)
    plt.gca().add_artist(bbb)

    # --------------------------------------------------------------------

plt.subplot(132, sharex=ax0)
plt.title(r'$m_ac^2 = $%i eV' %mass_ii)
for i, zz in enumerate(z):
    plt.loglog(ETeV, np.exp(-opt_depth_finke[i, :]),
               ls='-',
                 color=plt.cm.CMRmap(i / float(len(z))),
                 # color=colors[mi],
               label='$z = {0:.2f}$'.format(zz), lw=2)

    plt.loglog(ETeV, np.exp(-opt_depth_axion[i, :]),
               ls='--',
                 color=plt.cm.CMRmap(i / float(len(z))),
                 # color=colors[mi],
               lw=2)
    # plt.axvline(Etau1GeV[i] / 1e3, ls=':', color = plt.cm.CMRmap(i / float(len(z))) )

plt.gca().set_ylim((1e-4, 2.))
plt.gca().set_xlabel('Energy (TeV)', size='x-large')
plt.gca().set_ylabel(r'Attenuation $exp(-\tau)$', size='x-large')
aaa = plt.legend(loc=3)
markers = ['-', '--']
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(2)],
                 ['Finke22', 'Finke + cosmic axion'],
                 loc=4, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

# --------------------------------------------------------------------
# plt.figure()
plt.subplot(133, sharex=ax0)
plt.title(r'$m_ac^2 = $%i eV' %mass_ii)
for i, zz in enumerate(z):
    plt.loglog(ETeV, opt_depth_finke[i, :],
               ls='-',
                 color=plt.cm.CMRmap(i / float(len(z))),
                 # color=colors[mi],
               label='$z = {0:.2f}$'.format(zz), lw=2)

    plt.loglog(ETeV, opt_depth_axion[i, :],
               ls='--',
                 color=plt.cm.CMRmap(i / float(len(z))),
                 # color=colors[mi],
               lw=2)

plt.gca().set_ylim((4e-8, 800.))
plt.gca().set_xlabel('Energy (TeV)', size='x-large')
plt.gca().set_ylabel(r'Optical depth $\tau$', size='x-large')
aaa = plt.legend(loc=2)
markers = ['-', '--']
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(2)],
                 ['Finke22', 'Finke + cosmic axion'],
                 loc=4, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

plt.savefig('outputs/figures_paper/opacities_test.png',
            bbox_inches='tight')

plt.show()
