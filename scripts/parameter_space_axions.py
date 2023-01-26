# IMPORTS --------------------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import h as h_plank

from ebl_codes.EBL_class import EBL_model

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


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


if not os.path.exists("../outputs/"):
    # if the directory for outputs is not present, create it.
    os.makedirs("../outputs/")


def input_yaml_data_into_class(yaml_data):
    z_array = np.linspace(float(yaml_data['redshift_array']['zmin']),
                          float(yaml_data['redshift_array']['zmax']),
                          yaml_data['redshift_array']['zsteps'])

    lamb_array = np.logspace(np.log10(float(yaml_data['wavelenght_array']['lmin'])),
                             np.log10(float(yaml_data['wavelenght_array']['lmax'])),
                             yaml_data['wavelenght_array']['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     h=float(yaml_data['cosmology_params']['cosmo'][0]),
                     omegaM=float(yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'])


# FIGURE: AXION MASS-GAYY AND MASS-GAMMA PARAMETER SPACES -----------------------------------
def gamma_from_rest(mass, gay):
    return ((mass * u.eV) ** 3. * (gay * u.GeV ** -1) ** 2. / 32. / h_plank.to(u.eV * u.s)).to(u.s ** -1).value


axion_mac2 = np.logspace(np.log10(3), 1, num=20)
axion_gay = np.logspace(np.log10(5e-11), -9, num=20)

axion_gamma = np.logspace(np.log10(gamma_from_rest(axion_mac2[0], axion_gay[0])),
                          np.log10(gamma_from_rest(axion_mac2[-1], axion_gay[-1])), num=len(axion_gay))

values_gamma_array = np.zeros((len(axion_mac2), len(axion_gamma)))
values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))

config_data = read_config_file('scripts/input_data.yml')
ebl_class = input_yaml_data_into_class(config_data)
ebl_class.logging_prints = False


for key in config_data['ssp_models']:
    print(key)
    ebl_class.ebl_all_calculations(ssp_yaml=config_data['ssp_models'][key],
                                   log10_Aihl=float(config_data['ihl_params']['A_ihl']),
                                   alpha=float(config_data['ihl_params']['alpha']),
                                   mass=float(config_data['axion_params']['axion_mass']),
                                   gamma=float(config_data['axion_params']['axion_gamma'])
                                   )

    for aa in range(len(axion_mac2)):
        for bb in range(len(axion_gamma)):
            ebl_class.change_axion_contribution(axion_mac2[aa], axion_gamma[bb])

            if 10 ** ebl_class.ebl_total_spline(np.log10(3e8 / 0.608 * 1e6), 0.) < 16.37:
                values_gamma_array[aa, bb] = 1.

            # --------------
            ebl_class.change_axion_contribution(axion_mac2[aa], gamma_from_rest(axion_mac2[aa], axion_gay[bb]))

            if 10 ** ebl_class.ebl_total_spline(np.log10(3e8 / 0.608 * 1e6), 0.) < 16.37:
                values_gay_array[aa, bb] = 1.

    plt.subplots(2, 1, figsize=(10, 8))

    plt.subplot(211)
    plt.title('SSP model: %s' % config_data['ssp_models'][key]['name'])
    aaa = plt.pcolor(axion_mac2, axion_gamma, values_gamma_array.T)

    plt.colorbar(aaa)

    plt.xscale('log')
    plt.yscale('log')

    plt.ylabel(r'$\Gamma_{a}$ [s$^{-1}$]')

    plt.xlim(axion_mac2[0], axion_mac2[-1])
    plt.ylim(axion_gamma[0], axion_gamma[-1])

    plt.subplot(212)

    bbb = plt.pcolor(axion_mac2, axion_gay, values_gay_array.T)
    plt.colorbar(bbb)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'm$_a\,$c$^2$ [eV]')
    plt.ylabel(r'$g_{a\gamma}$ [GeV$^{-1}$]')

    plt.xlim(axion_mac2[0], axion_mac2[-1])
    plt.ylim(axion_gay[0], axion_gay[-1])

plt.show()
