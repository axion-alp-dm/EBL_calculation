# IMPORTS --------------------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from EBL_class import EBL_model


# Configuration file reading and data input/output ---------#

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


if not os.path.exists("outputs/"):
    # if the directory for outputs is not present, create it.
    os.makedirs("outputs/")


def input_yaml_data_into_class(yaml_data):

    z_array = np.linspace(float(yaml_data['zmin']), float(yaml_data['zmax']), yaml_data['zsteps'])
    lamb_array = np.logspace(np.log10(float(yaml_data['lmin'])), np.log10(float(yaml_data['lmax'])),
                             yaml_data['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     sfr=yaml_data['sfr'], sfr_params=yaml_data['sfr_params'],
                     path_SSP=yaml_data['path_SSP'],
                     dust_abs_models=yaml_data['dust_abs_models'],
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'],
                     h=float(yaml_data['cosmo'][0]), omegaM=float(yaml_data['cosmo'][1]),
                     omegaBar=float(yaml_data['omegaBar']),
                     axion_decay=yaml_data['axion_decay'])


# Calculations of emissivity and EBL ----------------#

config_data = read_config_file('data.yml')

lambdas = np.array([1, 0.35, 0.2, 0.15])
freqs = np.log10(3e8/(lambdas * 1e-6))
zz_plotarray = np.linspace(0, 10)
emiss = np.zeros([3, len(zz_plotarray)])

waves_ebl = np.logspace(-1, 3)
freq_array_ebl = np.log10(3e8/(waves_ebl * 1e-6))

models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'k', 'g', 'orange']
j = 0

fig1 = plt.figure()
axes1 = fig1.gca()

fig2 = plt.figure()
axes2 = fig2.gca()

for key in config_data.keys():
    print('EBL model: ', key)
    test_stuff = input_yaml_data_into_class(config_data[key])
    test_stuff.calc_emissivity()
    test_stuff.calc_ebl()

    plt.figure(fig1)
    plt.plot(zz_plotarray, test_stuff.emi_spline(freqs[0], zz_plotarray)[0], linestyle=models[j],
             color=colors[0], label=config_data[key]['name'])
    for i in range(1, len(freqs)):
        plt.plot(zz_plotarray, test_stuff.emi_spline(freqs[i], zz_plotarray)[0], color=colors[i], linestyle=models[j])

    plt.figure(fig2)
    plt.plot(waves_ebl, 10**test_stuff.ebl_total_spline(freq_array_ebl, 0., grid=False), linestyle=models[0],
             color=colors[j], label=config_data[key]['name'])
    plt.plot(waves_ebl, 10**test_stuff.ebl_ssp_spline(freq_array_ebl, 0., grid=False), linestyle=models[1],
             color=colors[j])
    plt.plot(waves_ebl, 10**test_stuff.ebl_axion_spline(freq_array_ebl, 0., grid=False), linestyle=models[2],
             color=colors[j])

    j += 1

plt.figure(fig1)

plt.ylabel(r'$\epsilon_{\nu}$ [erg/s/Hz/Mpc3]')
plt.xlabel('z')
plt.title('Emissivity')

lines = axes1.get_lines()
legend1 = plt.legend(loc=1)
legend2 = plt.legend([lines[i] for i in range(len(lambdas))], lambdas, loc=4, title=r'$\lambda$ [$\mu$m]')
axes1.add_artist(legend1)
axes1.add_artist(legend2)

plt.figure(fig2)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'EBL SED (nW / m$^2$ sr)')

lines = axes2.get_lines()
legend11 = plt.legend(loc=1)
legend22 = plt.legend([lines[i] for i in range(3)], ['Total', 'SSP', 'Axion decay'], loc=3, title=r'Component')
axes2.add_artist(legend11)
axes2.add_artist(legend22)

plt.xlim([.1, 1E3])
plt.ylim(1E-4, 1.1e2)  # 1.5 * np.max(ebl_axion[:, i * 10] + ebl_SSP[:, i * 10])])

plt.show()
