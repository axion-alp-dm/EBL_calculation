# IMPORTS --------------------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from EBL_class import EBL_model
from EBL_measurements.EBL_measurs_plot import plot_ebl_measurement_collection

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
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

waves_ebl = np.logspace(-1, 3, num=700)
freq_array_ebl = np.log10(3e8/(waves_ebl * 1e-6))

models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'k', 'g', 'orange']
j = 0

axion_mac2 = np.logspace(-2, 1, num=20)
axion_gamma = np.logspace(-26, -22, num=20)
values_array = np.zeros((len(axion_mac2), len(axion_gamma)))
freq_Driver16 = np.log10(3e8*1e6/np.array([0.471900519751, 0.6190494643, 0.748572101049,
               0.904957124161, 1.03626837678, 1.25260698788, 1.66465227795, 2.15280842939, 3.36374217832, 4.65274441363]))
ebl_Driver16 = np.array([5.14360495045, 7.34847435842, 9.37197207554, 10, 10.3296238484,
          10.1634757088, 9.83915373687, 8.64210857596, 5.40000753964, 3.4293457049])

fig1 = plt.figure()
axes1 = fig1.gca()

fig2 = plt.figure(figsize=(14, 8))
axes2 = fig2.gca()

fig3 = plt.figure()
axes3 = fig3.gca()

for key in config_data.keys():
    print('EBL model: ', key)
    test_stuff = input_yaml_data_into_class(config_data[key])
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
    # plt.plot(waves_ebl, 10**test_stuff.ebl_intra_spline(freq_array_ebl, 0., grid=False), linestyle=models[2],
    #          color=colors[j])
    plt.plot(waves_ebl, 10**test_stuff.ebl_axion_spline(freq_array_ebl, 0., grid=False), linestyle=models[3],
             color=colors[j])
    j += 1

    for aa in range(len(axion_mac2)):
        test_stuff.axion_mass = axion_mac2[aa]
        for bb in range(len(axion_gamma)):
            test_stuff.axion_gamma = axion_gamma[bb]

            ebl_ourvalues = 10**test_stuff.ebl_total_spline(freq_Driver16, 0., grid=False)
            if np.all(ebl_ourvalues > ebl_Driver16) and\
                    10**test_stuff.ebl_total_spline(np.log10(3e8/0.608*1e6), 0.) < 16.37:
                values_array[aa, bb] = 1.


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

plot_ebl_measurement_collection('EBL_measurements/EBL_measurements.yml')

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu$ (nW / m$^2$ sr)')

lines = axes2.get_lines()
legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
legend22 = plt.legend([lines[i] for i in range(3)], ['Total', 'SSP', 'Axion decay', 'IHL'], loc=3, title=r'Component')
axes2.add_artist(legend11)
axes2.add_artist(legend22)

plt.xlim([.1, 1E3])
plt.ylim(1E-4, 1.1e2)  # 1.5 * np.max(ebl_axion[:, i * 10] + ebl_SSP[:, i * 10])])
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)


plt.figure(fig3)

aaa = plt.pcolor(axion_mac2, axion_gamma, values_array.T)

plt.colorbar(aaa)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'log$_{10}$(m$_a\,$c$^2$/eV)')
plt.ylabel(r'log$_{10}$($\Gamma_a$/s$^{-1}$)')

plt.xlim(axion_mac2[0], axion_mac2[-1])
plt.ylim(axion_gamma[0], axion_gamma[-1])


# plt.figure()
# plt.plot(waves_ebl, 10**test_stuff.ebl_total_spline(freq_array_ebl, 0., grid=False), linestyle=models[0],
#              color=colors[j], label=config_data['Kneiste_only']['name'])
# plt.plot(waves_ebl, 10**test_stuff.ebl_ssp_spline(freq_array_ebl, 0., grid=False), linestyle=models[1],
#              color=colors[j])
# plt.plot(waves_ebl, 10**test_stuff.ebl_axion_spline(freq_array_ebl, 0., grid=False), linestyle=models[2],
#              color=colors[j])
#
# plot_ebl_measurement_collection('EBL_measurements/EBL_measurements.yml')
#
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel(r'Wavelength ($\mu$m)')
# plt.ylabel(r'EBL SED (nW / m$^2$ sr)')
plt.show()
