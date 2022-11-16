# IMPORTS --------------------------------------------#
import yaml
import numpy as np

from EBL_class import EBL_model


# Configuration file reading and data input ---------#

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


# Calculations of emissivity and EBL ----------------#

config_data = read_config_file('data.yml')

for key in config_data.keys():
    print('EBL model: ', key)
    z_array = np.linspace(float(config_data[key]['zmin']), float(config_data[key]['zmax']), config_data[key]['zsteps'])
    freq_array = np.logspace(np.log10(float(config_data[key]['lmin'])), np.log10(float(config_data[key]['lmax'])),
                             config_data[key]['lfsteps'])
    sfr = config_data[key]['sfr']
    sfr_params = config_data[key]['sfr_params']
    path_SSP = config_data[key]['path_SSP']
    dust_abs = config_data[key]['dust_abs_model']

    test_stuff = EBL_model(z_array, freq_array, sfr, sfr_params, path_SSP, dust_abs_model=dust_abs)
    test_stuff.calc_emissivity()
    #test_stuff.calc_ebl()
