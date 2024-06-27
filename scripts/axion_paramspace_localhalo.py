# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebltable.ebl_from_model import EBL

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
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
direct_name = str('individuals_Dfact'
                  # + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# Choose the max and minimum wavelengthS of the data that we import
lambda_min_total = 3e-3  # [microns]
lambda_max_total = 5.  # [microns]

fig, ax = plt.subplots()

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)

D_factor = 2.20656e22 * u.GeV * u.cm ** -2
# D_factor = 1.11e22 * u.GeV * u.cm ** -2
def chi2_upperlims(x_model, x_obs, err_obs):
    """

    :param x_model:
    :param x_obs:
    :param err_obs:
    :return:
    """
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))

def gamma_def(mass_eV, gay_GeV):
    return ((mass_eV * u.eV) ** 3. * (gay_GeV * u.GeV ** -1) ** 2.
            / (32. * h_plank)).to(u.s ** -1)
def host_function_std(x_array, mass_eV, gay_GeV, v_dispersion=220.):
    lambda_decay = 2.48 / mass_eV * u.um
    luminosiy = (1 / (4 * np.pi * u.sr)
                 * gamma_def(mass_eV=mass_eV, gay_GeV=gay_GeV)
                 / mass_eV * u.eV ** -1
                 * D_factor
                 * h_plank * c).to(u.nW * u.m ** -1 * u.sr ** -1)
    # print(luminosiy)

    sigma = (2. * lambda_decay
             * (v_dispersion * u.km * u.s ** -1 / c).to(1))

    gaussian = (1 / np.sqrt(2. * np.pi) / sigma
                # * np.exp(
                # -0.5 * ((x_array * u.um - lambda_decay) / sigma) ** 2.)
                ).to(u.m ** -1)

    return (luminosiy * gaussian).to(u.nW * u.m ** -2 * u.sr ** -1)#.value


waves_ebl = np.geomspace(5e-6, 10, num=int(1e6))
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)

upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=ax, plot_measurs=True)

chi2_min = chi2_upperlims(
    x_model=spline_cuba(upper_lims_all['lambda']),
    x_obs=upper_lims_all['nuInu'],
    err_obs=upper_lims_all['1 sigma'])

chi2_delta = chi2_min + 4.61  # for upper limits

nuInu_extra = (upper_lims_all['nuInu']
               - spline_cuba(upper_lims_all['lambda'])
               + upper_lims_all['1 sigma'] * chi2_delta ** 0.5)

# sigma = (2. * upper_lims_all['lambda']
#          * (220. * u.km * u.s ** -1 / c).to(1))
g_ay_array = (1e-10 *
              (nuInu_extra
               / (14.53
                  * (2.48 / upper_lims_all['lambda']) ** 3.
                  * (D_factor / (1.11e22 * u.GeV * u.cm ** -2)).to(1))
               ) ** 0.5
              )

plt.plot(waves_ebl, spline_cuba(waves_ebl), 'k')
plt.scatter(upper_lims_all['lambda'],
            spline_cuba(upper_lims_all['lambda'])+host_function_std(
    upper_lims_all['lambda'].value,
    2.48/upper_lims_all['lambda'].value, g_ay_array.value).value,
            color='r')
plt.yscale('log')
plt.xscale('log')
def nuInu_maybe(mass, gayy):
    return (c/(512.*np.pi*h_plank*220*u.km/u.s* np.sqrt(2. * np.pi))
       * (mass*u.eV)**3.*(gayy*u.GeV**-1)**2.*D_factor).to(
        u.nW * u.m **-2)


print(host_function_std(2.48, 1., 1e-10))
aaa = nuInu_maybe(1., 1e-10)
print('aaa', aaa)

np.savetxt('outputs/' + direct_name + '/dipsD.txt',
           np.column_stack((2.48 / upper_lims_all['lambda'],
                            g_ay_array.value)))


h=0.7
omegaM=0.3
omegaBar=0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                                    Ob0=omegaBar, Tcmb0=2.7255)
axion_mass = 1. * u.eV
axion_gayy = 1e-10 * u.GeV ** -1
llhh = (cosmo.Odm(0.) * cosmo.critical_density0
        * c**2 * gamma_def(axion_mass.value, axion_gayy.value))
llhh = llhh.to(u.erg/u.m**3./u.s)
print(cosmo.Odm(0.))
print(cosmo.critical_density0)
print(gamma_def(axion_mass.value, axion_gayy.value))
print(llhh)


cosmic_decay = (cosmo.Odm(0.) * cosmo.critical_density0*c**3.
                / (128. * np.pi * h_plank * cosmo.H(0.))
                * axion_mass**3. * axion_gayy**2.)
cosmic_decay = cosmic_decay.to(u.nW * u.m **-2)
print(cosmo.H(0.))
print(cosmic_decay)
plt.show()
