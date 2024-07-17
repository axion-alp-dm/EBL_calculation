import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline

from astropy import units as u
import astropy.constants as c

from ebl_codes.EBL_class import EBL_model
from ebltable.ebl_from_model import EBL
from data.cb_measurs.import_cb_measurs import import_cb_data

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

cuba_cosmic = np.loadtxt('outputs/figures_paper/cuba_cosmic_constraints.txt')
cuba_host = np.loadtxt('outputs/figures_paper/limits_cosmic_host.txt')
plt.figure()
plt.loglog(cuba_cosmic[:, 0], cuba_cosmic[:, 1], c='k')
plt.loglog(cuba_host[:, 0], cuba_host[:, 1], c='b')
# plt.show()
cosmic_magay_spline = UnivariateSpline(
    cuba_cosmic[:, 0], cuba_cosmic[:, 1],
    k=1, s=0)


ebl_cuba = EBL.readmodel('cuba')
ebl_finke = EBL.readmodel('finke2022')


sigma = 0.05
yyy = (21.98
           # * 1 / np.sqrt(2. * np.pi) / sigma
           * np.exp(
            -0.5 * ((10 ** ebl_cuba.x - 0.608) #*((1 + ebl_our2[0, 1:]))[np.newaxis,:])
                    / sigma) ** 2.)
       )[:, np.newaxis]

ebl_axion = EBL(z=ebl_cuba.y, lmu=10 ** ebl_cuba.x,
                nuInu=10**ebl_cuba.Z + yyy, model='cuba+axion')


z = np.arange(0., .5, .1)
lmu = np.logspace(-2, 1.5, 10000)
ETeV = np.logspace(-4, 1, 50)

nuInu_finke = ebl_finke.ebl_array(z, lmu)
nuInu_cuba = ebl_cuba.ebl_array(z, lmu)
nuInu_our = ebl_axion.ebl_array(z, lmu)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplot(121)

for i, zz in enumerate(z):
    plt.loglog(lmu, nuInu_finke[i],
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)
    plt.loglog(lmu, nuInu_cuba[i],
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               # label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)

    plt.loglog(lmu, nuInu_our[i],
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               # label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)

    # plt.axvline(lmu[np.argmax(nuInu_our2[i])],
    #             color=plt.cm.CMRmap(i / float(len(z))))

plt.gca().set_xlabel('Wavelength ($\mu$m)', size='x-large')
plt.gca().set_ylabel(
    r'$\nu I_\nu (\mathrm{nW}\,\mathrm{sr}^{-1}\mathrm{m}^{-2})$',
    size='x-large')
aaa = plt.legend(loc='lower center', ncol=2)

markers = ['-', '--', 'dotted']
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(3)],
                 ['Finke22', 'CUBA', 'CUBA + gaussian'],
                 loc=1, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax, plot_measurs=True)

plt.xlim(0.2, 2.)
plt.ylim(1.3, 150.)


plt.subplot(122)
for i, zz in enumerate(z):
    plt.loglog(ETeV, np.exp(-ebl_finke.optical_depth(zz, ETeV)),
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               label='$z = {0:.3f}$'.format(zz), lw=2)
    plt.loglog(ETeV, np.exp(-ebl_cuba.optical_depth(zz, ETeV)),
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.3f}$'.format(zz),
               lw=2)
    plt.loglog(ETeV, np.exp(-ebl_axion.optical_depth(zz, ETeV)),
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.3f}$'.format(zz),
               lw=2)
    # plt.axvline(Etau1GeV[i] / 1e3, ls=':', color = plt.cm.CMRmap(i / float(len(z))) )

plt.gca().set_ylim((1e-5, 2.))
plt.gca().set_xlim((1e-1, 2e1))
plt.gca().set_xlabel('Energy (TeV)', size='x-large')
plt.gca().set_ylabel(r'Attenuation $\exp(-\tau)$', size='x-large')
aaa = plt.legend(loc=5)
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                             color='k')
                  for i in range(3)],
                 ['Finke22', 'Our model', 'Ours + gaussian'],
                 loc=4, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

plt.show()
