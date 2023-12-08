# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.import_cb_measurs import import_cb_data

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank

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
direct_name = str('final_outputs_clumpy_vdispersion_NHonly'
                  # + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# Choose the max and minimum wavelengthS of the data that we import
lambda_min_total = 0.  # [microns]
lambda_max_total = 5.  # [microns]


# If the directory for outputs is not present, create it.
# if not os.path.exists("outputs/"):
#     os.makedirs("outputs/")
# if not os.path.exists('outputs/' + direct_name):
#     os.makedirs('outputs/' + direct_name)


def chi2_upperlims(x_model, x_obs, err_obs):
    """

    :param x_model:
    :param x_obs:
    :param err_obs:
    :return:
    """
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


# Configuration of EBL class from input yaml file
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


config_data = read_config_file(
    'scripts/input_files/input_data_paper.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)

# Parameter space for axion characteristics and rest of necessary arrays
# axion_mac2 = np.logspace(-1, 7, num=500)
# axion_gay = np.logspace(-20, -7, num=500)
axion_mac2 = np.logspace(0, 7, num=250)
axion_gay = np.logspace(-12, -9, num=250)

waves_ebl = np.logspace(np.log10(5e-6), 1, 3000)
# waves_ebl = np.linspace(0.1, 10, num=5000)
freq_array_ebl = np.log10(c.value / (waves_ebl * 1e-6))

# We introduce the Finke22 and CUBA splines
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


def spline_pegase0001(lambda_array):
    method_y = np.load('outputs/final_outputs 2023-10-04 16:15:16/'
                       'pegase0.0001_Finkespline.npy',
                       allow_pickle=True).item()
    return method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                    grid=False)


def spline_starburst(lambda_array):
    method_y = np.load('outputs/final_outputs 2023-10-04 16:15:16/'
                       'SB99_kneiskespline.npy',
                       allow_pickle=True).item()
    return method_y(np.log10(c.value * 1e6 / lambda_array), 0.,
                    grid=False)


list_working_models = {
    'ModelA': {'label': 'Model A', 'callable_func': spline_pegase0001},
    'ModelB': {'label': 'Model B', 'callable_func': spline_starburst},
    'CUBA': {'label': 'CUBA', 'callable_func': spline_cuba},
    'Finke22': {'label': 'Finke+22', 'callable_func': spline_finke}
}

# Beginning of figure specifications
plt.figure(figsize=(16, 10))  # figsize=(16, 10))
ax1 = plt.gca()


def gamma_def(mass_eV, gay_GeV):
    return ((mass_eV * u.eV) ** 3. * (gay_GeV * u.GeV ** -1) ** 2.
            / (32. * h_plank)).to(u.s ** -1)


# D_factor = 1.87e22 * u.GeV * u.cm ** -2 * u.sr ** -1
D_factor = 1.11e22 * u.GeV * u.cm ** -2


def host_function(x_array, mass_eV, gay_GeV, sigma):
    lambda_decay = 2.48 / mass_eV
    luminosiy = (D_factor
                 * gamma_def(mass_eV=mass_eV, gay_GeV=gay_GeV)).to(
        u.nW * u.m ** -2 * u.sr ** -1).value

    # print(lambda_decay, luminosiy)

    return (luminosiy / np.sqrt(2. * np.pi) / sigma
            * np.exp(-0.5 * ((x_array - lambda_decay) / sigma) ** 2.))


def host_function_std(x_array, mass_eV, gay_GeV, v_dispersion=220.):
    lambda_decay = 2.48 / mass_eV * u.um
    luminosiy = (1 / (4 * np.pi * u.sr)
                 * gamma_def(mass_eV=mass_eV, gay_GeV=gay_GeV)
                 / mass_eV * u.eV ** -1
                 * D_factor
                 * h_plank * c).to(u.nW * u.m ** -1 * u.sr ** -1)
    sigma = 2. * lambda_decay * (v_dispersion * u.km * u.s**-1 / c).to(1)


    gaussian = (1 / np.sqrt(2. * np.pi) / sigma
                * np.exp(-0.5 * ((x_array*u.um - lambda_decay) / sigma) ** 2.)
                ).to(u.m**-1)


    print(lambda_decay, luminosiy, sigma, gaussian)

    return (luminosiy * gaussian).to(u.nW * u.m ** -2 * u.sr ** -1)#.value


x_array = np.linspace(-5,5, num=int(1e7))
aaa = host_function_std(x_array, 1., 1e-10)


print(host_function_std(2.48, 1., 1e-10))

plt.figure()
plt.plot(x_array, aaa)
plt.show()
ma_ex = 4.079
gay_ex = 9.9e-13

plt.loglog(waves_ebl, spline_cuba(waves_ebl), c='k')
plt.loglog(waves_ebl, spline_cuba(waves_ebl)
           + host_function_std(waves_ebl, mass_eV=ma_ex,
                               gay_GeV=gay_ex), c='r')

ebl_class.change_axion_contribution(ma_ex, gay_ex)
plt.loglog(waves_ebl, spline_cuba(waves_ebl)
           + host_function_std(waves_ebl, mass_eV=ma_ex,
                               gay_GeV=gay_ex)
           + 10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0.,
                                              grid=False),
           c='g')

plt.loglog(waves_ebl, spline_cuba(waves_ebl), c='fuchsia')
plt.loglog(waves_ebl, spline_finke(waves_ebl), c='orange')
plt.loglog(waves_ebl, 10**spline_pegase0001(waves_ebl), c='r', lw=3)
plt.loglog(waves_ebl, 10**spline_starburst(waves_ebl), c='b', lw=3)

ebl_class.change_axion_contribution(1e2, 1e-13)
plt.loglog(waves_ebl,
           (10 ** ebl_class.ebl_axion_spline(freq_array_ebl, 0.,
                                             grid=False)
            + spline_cuba(waves_ebl)), c='green')


# We introduce all the EBL measurements
# We might want to take out 'matsuoka2011.ecsv' ???
upper_lims_all, _ = import_cb_data(
    lambda_min_total=lambda_min_total,
    lambda_max_total=lambda_max_total,
    ax1=ax1, plot_measurs=True)

upper_lims_all_woNH = upper_lims_all[
    upper_lims_all['ref'] != r'NH/LORRI (Lauer+ \'22)']

plt.xlim(5e-6, 1e1)
plt.ylim(5e-3, 120)

colors = ['b', 'r', 'orange', 'fuchsia', 'green']
linewidths = [3, 3, 2, 2, 2]
legend22 = plt.legend([plt.Line2D([], [],
                                  linewidth=linewidths[i],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(5)],
                      ['Model A', 'Model B',
                       'Finke22', 'CUBA', r'CUBA + axion decay'
                               '\n(example)'
                               '\n'
                               r'    m$_a = 10^2$ eV'
                               '\n'
                               r'    g$_{a\gamma} = 10^{-13}$ GeV$^{-1}$'],
                      loc=7, bbox_to_anchor=(1., 0.3),
                      title=r'Models', fontsize=16)

ax1.add_artist(legend22)
legend11 = plt.legend(title='Measurements', ncol=2, loc=2, fontsize=11.5,
                      title_fontsize=20)#, bbox_to_anchor=(1.001, 0.99))

# ax1.add_artist(legend11)


plt.annotate(text='', xy=(3e-3, 7e-3), xytext=(5e-6, 7e-3),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='', xy=(0.1, 7e-3), xytext=(3e-3, 7e-3),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='', xy=(10, 7e-3), xytext=(0.1, 7e-3),
             arrowprops=dict(arrowstyle='<->', color='grey'),
             alpha=0.7, zorder=-10)
plt.annotate(text='CXB', xy=(1e-4, 7.5e-3), alpha=0.7, color='grey')
plt.annotate(text='CUB', xy=(0.035, 7.5e-3), alpha=0.7, color='grey')
plt.annotate(text='COB', xy=(1, 7.5e-3), alpha=0.7, color='grey')

plt.savefig('outputs/cb.pdf', bbox_inches='tight')
plt.savefig('outputs/cb.png', bbox_inches='tight')
plt.show()



aaa = []
for working_model_name in list_working_models.keys():
    print(working_model_name)

    # working_model = list_working_models[
    #     working_model_name]['callable_func']

    values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))
    values_gay_array_NH = np.zeros((len(axion_mac2), len(axion_gay)))
    values_nuInu = np.zeros((len(axion_mac2), len(axion_gay)))

    for na, aa in enumerate(axion_mac2):
        if na % 25 == 0:
            print(na)

        for nb, bb in enumerate(axion_gay):
            ebl_class.change_axion_contribution(aa, bb)
            working_model = UnivariateSpline(
                waves_ebl,
                list_working_models[
                    working_model_name]['callable_func'](waves_ebl)
                + host_function_std(
                    waves_ebl, mass_eV=aa, gay_GeV=bb),
                s=0, k=1)
            # if na % 25 == 0 and nb % 25 == 0:
            #     plt.figure()
            #     plt.title(str(aa) + '  ' + str(bb))
            #     plt.loglog(waves_ebl, working_model(waves_ebl))
            #     plt.show()
            # host_function(waves_ebl, mass_eV=aa, gay_GeV=bb, sigma=4e-3)
            # host_decay = (D_factor * gamma_def(mass_eV=aa, gay_GeV=bb)).to(
            #     u.nW * u.m ** -2 * u.sr ** -1).value
            # print(host_decay, working_model(upper_lims_all['lambda']),
            #       working_model(upper_lims_all['lambda'])
            #              + host_decay)

            values_gay_array[na, nb] += 2. * chi2_upperlims(
                x_model=(10 ** ebl_class.ebl_axion_spline(
                    np.log10(c.value / (upper_lims_all['lambda'] * 1e-6)),
                    0.,
                    grid=False)
                         + working_model(upper_lims_all['lambda'])),
                x_obs=upper_lims_all['nuInu'],
                err_obs=upper_lims_all['1 sigma'])

            # values_gay_array_NH[na, nb] += 2. * chi2_upperlims(
            #     x_model=(10 ** ebl_class.ebl_axion_spline(
            #         np.log10(c.value
            #                  / (upper_lims_all_woNH['lambda'] * 1e-6)),
            #         0.,
            #         grid=False)
            #              + working_model(upper_lims_all_woNH['lambda'])),
            #     x_obs=upper_lims_all_woNH['nuInu'],
            #     err_obs=upper_lims_all_woNH['1 sigma'])

            values_gay_array_NH[na, nb] += (
                    ((16.7 - (10 ** ebl_class.ebl_axion_spline(
                        np.log10(c.value / (0.608 * 1e-6)), 0.,
                        grid=False)
                              + working_model(0.608))
                      ) / 1.47) ** 2.)

            values_nuInu[na, nb] += (10 ** ebl_class.ebl_axion_spline(
                np.log10(c.value / (0.608 * 1e-6)), 0., grid=False)
                                     + working_model(0.608))

    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_nuInu', values_nuInu)
    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_UL', values_gay_array)
    np.save('outputs/' + direct_name + '/'
            + str(working_model_name) + '_params_measur',
            values_gay_array_NH)
    aaa.append([working_model_name, list_working_models[
        working_model_name]['label']])
    print()

np.save('outputs/' + direct_name + '/axion_params',
        np.column_stack((axion_mac2, axion_gay)))

np.save('outputs/' + direct_name + '/list_models', aaa)
print(direct_name)
