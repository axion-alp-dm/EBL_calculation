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
                     #sfr=yaml_data['sfr'], sfr_params=yaml_data['sfr_params'],
                     #path_SSP=yaml_data['path_SSP'],
                     #dust_abs_models=yaml_data['dust_abs_models'],
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'])
                     #axion_decay=yaml_data['axion_decay'],
                     #log_prints=yaml_data['log_prints'])



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


def gamma_from_rest(mass, gay):
    return ((mass*u.eV)**3. * (gay*u.GeV**-1)**2. / 32. / h_plank.to(u.eV * u.s)).to(u.s**-1).value

axion_mac2 = np.logspace(np.log10(3), 1, num=15)
axion_gay = np.logspace(np.log10(5e-11), -9, num=20)

axion_gamma = np.logspace(np.log10(gamma_from_rest(axion_mac2[0], axion_gay[0])),
        np.log10(gamma_from_rest(axion_mac2[-1], axion_gay[-1])), num=len(axion_gay))

values_gamma_array = np.zeros((len(axion_mac2), len(axion_gamma)))
values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))


fig1 = plt.figure()
axes1 = fig1.gca()

fig2 = plt.figure(figsize=(14, 8))
axes2 = fig2.gca()

ebl_class = input_yaml_data_into_class(config_data)

for key in config_data['ssp_models']:
    print(key, config_data['ssp_models'][key])
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

'''
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

    test_stuff.logging_prints = False
    for aa in range(len(axion_mac2)):
        test_stuff.axion_mass = axion_mac2[aa]
        for bb in range(len(axion_gamma)):
            test_stuff.axion_gamma = axion_gamma[bb]

            if 10**test_stuff.ebl_total_spline(np.log10(3e8/0.608*1e6), 0.) < 16.37:
                values_gamma_array[aa, bb] = 1.

            #--------------
            test_stuff.axion_gamma = gamma_from_rest(axion_mac2[aa], axion_gay[bb])

            if 10**test_stuff.ebl_total_spline(np.log10(3e8/0.608*1e6), 0.) < 16.37:
                values_gay_array[aa, bb] = 1.



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
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')

lines = axes2.get_lines()
legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
legend22 = plt.legend([lines[i] for i in range(4)], ['Total', 'SSP', 'IHL', 'Axion decay'], loc=3, title=r'Component')
axes2.add_artist(legend11)
axes2.add_artist(legend22)

plt.xlim([.1, 1E3])
plt.ylim(1E-4, 1.1e2)  # 1.5 * np.max(ebl_axion[:, i * 10] + ebl_SSP[:, i * 10])])
#plt.xlim(0,5)
#plt.ylim(1e-2, 100)
plt.subplots_adjust(left=0.125, right=.65, top=.95, bottom=.13)


plt.subplots(2,1, figsize=(10, 8))

plt.subplot(211)
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


aaa = open('data.txt', 'w')

np.savetxt(aaa, axion_mac2.reshape(1, axion_mac2.shape[0]))
np.savetxt(aaa, axion_gay.reshape(1, axion_gay.shape[0]))
np.savetxt(aaa, values_gay_array)
aaa.close()


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
#plt.close('all')
'''
