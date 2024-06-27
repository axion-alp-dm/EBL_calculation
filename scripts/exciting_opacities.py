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

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

recalculate = False
if recalculate:

    def read_config_file(ConfigFile):
        with open(ConfigFile, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml


    # We initialize the class with the input file
    config_data = read_config_file('outputs/final_outputs_Zevol_fixezZsolar '
                                   '2024-06-05 09:10:05/' + 'input_data.yml')
    ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                     log_prints=True)
    ebl_class.ebl_ssp_calculation(
        config_data['ssp_models']['SB99_dustFinke'])

    waves_ebl = np.geomspace(5e-2, 50, num=int(1e4))

    zz = np.geomspace(1e-9, 10, num=100)

    lm, zm = np.meshgrid(waves_ebl, zz)
    data_nuInu = 10 ** ebl_class.ebl_ssp_spline(
        np.log10(c.c.value * 1e6 / lm), zm,
        grid=False)

    data_total = np.zeros((len(waves_ebl) + 1, len(zz) + 1))
    data_total[1:, 0] = waves_ebl
    data_total[0, 1:] = zz
    data_total[1:, 1:] = data_nuInu.T

    np.savetxt('outputs/final_outputs_Zevol_fixezZsolar '
               '2024-06-05 09:10:05/table_data.txt',
               data_total)

    # With a random gaussian
    data_total = np.zeros((len(waves_ebl) + 1, len(zz) + 1))
    data_total[1:, 0] = waves_ebl
    data_total[0, 1:] = zz
    data_total[1:, 1:] = (data_nuInu.T
                          + (
                                    21.98
                                    * 1 / np.sqrt(2. * np.pi) / 1.
                                    * np.exp(
                                -0.5 * ((waves_ebl - 0.608) / 1.) ** 2.))[:,
                            np.newaxis])

    np.savetxt('outputs/final_outputs_Zevol_fixezZsolar '
               '2024-06-05 09:10:05/table_data_gaussian.txt',
               data_total)

ebl_finke = EBL.readmodel('finke2022')
ebl_our = EBL.readascii('outputs/final_outputs_Zevol_fixezZsolar '
                        '2024-06-05 09:10:05/table_data.txt',
                        model_name='Our')
ebl_our2 = np.loadtxt('outputs/final_outputs_Zevol_fixezZsolar '
                      '2024-06-05 09:10:05/table_data.txt',
                      )
sigma = 0.005
ebl_our2[1:, 1:] = (
        ebl_our2[1:, 1:]
        + (21.98
           # * 1 / np.sqrt(2. * np.pi) / sigma
           * np.exp(
            -0.5 * ((ebl_our2[1:, 0] - 0.608) / sigma) ** 2.)
           )[:, np.newaxis])

np.savetxt('outputs/final_outputs_Zevol_fixezZsolar '
           '2024-06-05 09:10:05/table_data_gaussian.txt',
           ebl_our2)
ebl_our2 = EBL.readascii('outputs/final_outputs_Zevol_fixezZsolar '
                         '2024-06-05 09:10:05/table_data_gaussian.txt',
                         model_name='Our_withGaussian')
z = np.arange(0., 1.2, 0.2)
lmu = np.logspace(-1, 1.5, 10000)
ETeV = np.logspace(-1, 2, 50)

nuInu_finke = ebl_finke.ebl_array(z, lmu)
nuInu_our = ebl_our.ebl_array(z, lmu)
nuInu_our2 = ebl_our2.ebl_array(z, lmu)

tau_finke = ebl_finke.optical_depth(z, ETeV)

plt.figure()

for i, zz in enumerate(z):
    plt.loglog(lmu, nuInu_finke[i],
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)
    plt.loglog(lmu, nuInu_our[i],
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               # label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)

    plt.loglog(lmu, nuInu_our2[i],
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               lw=2.,
               # label='$z = {0:.2f}$'.format(zz),
               zorder=-1 * i)

plt.gca().set_xlabel('Wavelength ($\mu$m)', size='x-large')
plt.gca().set_ylabel(
    r'$\nu I_\nu (\mathrm{nW}\,\mathrm{sr}^{-1}\mathrm{m}^{-2})$',
    size='x-large')
aaa = plt.legend(loc='lower center', ncol=2)

markers = ['-', '--', 'dotted']
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                                 color='k')
                      for i in range(3)],
                     ['Finke22', 'Our model', 'Ours + gaussian'],
                     loc=2, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

plt.figure()
for i, zz in enumerate(z):
    plt.loglog(ETeV, ebl_finke.optical_depth(zz, ETeV),
               ls='-', color=plt.cm.CMRmap(i / float(len(z))),
               label='$z = {0:.3f}$'.format(zz), lw=2)
    plt.loglog(ETeV, ebl_our.optical_depth(zz, ETeV),
               ls='--', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.3f}$'.format(zz),
               lw=2)
    plt.loglog(ETeV, ebl_our2.optical_depth(zz, ETeV),
               ls='dotted', color=plt.cm.CMRmap(i / float(len(z))),
               # label = '$z = {0:.3f}$'.format(zz),
               lw=2)
    # plt.axvline(Etau1GeV[i] / 1e3, ls=':', color = plt.cm.CMRmap(i / float(len(z))) )

plt.gca().set_ylim((1e-2, 15.))
plt.gca().set_xlim((1e-1, 1e1))
plt.gca().set_xlabel('Energy (TeV)', size='x-large')
plt.gca().set_ylabel(r'Attenuation $\exp(-\tau)$', size='x-large')
aaa = plt.legend(loc=2)
bbb = plt.legend([plt.Line2D([], [], linestyle=markers[i],
                                 color='k')
                      for i in range(3)],
                     ['Finke22', 'Our model', 'Ours + gaussian'],
                     loc=4, fontsize=16, framealpha=0.4)
plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

plt.show()
