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

# Calculations of emissivity and EBL ----------------#

config_data = read_config_file('data.yml')

lambdas = np.array([1, 0.35, 0.2, 0.15])
freqs = np.log10(3e8/(lambdas * 1e-6))
zz_plotarray = np.linspace(0, 6)
emiss = np.zeros([3, len(zz_plotarray)])
models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'k', 'g']
fig = plt.figure()
axes = fig.gca()
j=0

for key in config_data.keys():
    print('EBL model: ', key)
    z_array = np.linspace(float(config_data[key]['zmin']), float(config_data[key]['zmax']), config_data[key]['zsteps'])
    freq_array = np.logspace(np.log10(float(config_data[key]['lmin'])), np.log10(float(config_data[key]['lmax'])),
                             config_data[key]['lfsteps'])
    sfr = config_data[key]['sfr']
    sfr_params = config_data[key]['sfr_params']
    path_SSP = config_data[key]['path_SSP']
    dust_abs = config_data[key]['dust_abs_models']

    test_stuff = EBL_model(z_array, freq_array, sfr, sfr_params, path_SSP, dust_abs_models=dust_abs)
    spline_emi = test_stuff.calc_emissivity()
    plt.plot(zz_plotarray, spline_emi(freqs[0], zz_plotarray)[0], linestyle=models[j],
             color=colors[0], label=config_data[key]['name'])
    for i in range(1, len(freqs)):
        plt.plot(zz_plotarray, spline_emi(freqs[i], zz_plotarray)[0], color=colors[i], linestyle=models[j])
    #test_stuff.calc_ebl()
    j+=1

lines = axes.get_lines()
legend1 = plt.legend(loc=1)
legend2 = plt.legend([lines[i] for i in range(len(lambdas))], lambdas, loc=4, title=r'$\lambda$ [um]')
axes.add_artist(legend1)
axes.add_artist(legend2)
plt.show()
