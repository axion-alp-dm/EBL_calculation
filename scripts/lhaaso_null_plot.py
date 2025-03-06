import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


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
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


# aaa = 'logParabola_BOSA'

# aaa = 'PWandEBL_BOSA'
# bbb = '_vs_bosa'
# vert_lines_against = 'BOSA.txt'

# aaa = 'PWandEBL_vs_chary'
# bbb = '_vs_chary'
# vert_lines_against = 'Chary.txt'

aaa = 'PWandEBL_vs_3body'
bbb = '_vs_3body'
vert_lines_against = '3_grey_bodies.txt'


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso/asimov_dict_' + aaa + '.yaml')

number_of_params = len(output_data['param_names'])

fig_params, ax_params = plt.subplots(
    1, number_of_params, figsize=(12, 5))

plt.subplot(1, number_of_params, 1)
plt.suptitle(r'$\phi (E) = \phi_0  E^{-\Gamma} e^{-\tau}$')

for param in range(number_of_params):
    plt.subplot(1, number_of_params, param+1)
    plt.xlabel(output_data['param_names'][param])

fig, ax = plt.subplots(figsize=(8, 6))
plt.suptitle(r'$\phi (E) = \phi_0  E^{-\Gamma} e^{-\tau}$')

hatches = ['', '', '']

my_ebl = ['3_grey_bodies.txt',
          'Chary.txt',
          'BOSA.txt']

for ni, i in enumerate(my_ebl):
    print(i)
    color = next(ax._get_lines.prop_cycler)['color']

    plt.figure(fig_params)

    param_array = np.array(output_data[i]['params_values' + bbb])

    for param in range(number_of_params):
        plt.subplot(1, number_of_params, param + 1)
        if param == number_of_params-1:
            plt.hist(param_array[:, param], color=color,
             bins=30, label=i, hatch=hatches[ni], alpha=0.4)
        else:
            plt.hist(param_array[:, param], color=color,
             bins=30, hatch=hatches[ni], alpha=0.4)

    # --------------------------------------------------------
    plt.figure(fig)
    plt.hist(2 * (np.array(output_data[i]['logL' + bbb])
                  - np.array(output_data['logL_poisson_with_itself'])),
             alpha=0.4, color=color,
             bins=30, label=i, hatch=hatches[ni])

    plt.axvline(2 * (output_data[i]['logL_asimov' + bbb]
                    - output_data[vert_lines_against]['logL_asimov' + bbb]),
                    ls='-',
                     color=color)


plt.legend()

plt.xlabel(r'- 2 $\Delta$log $L$')

plt.savefig('outputs/lhaaso/hist_3_eblmodels_' + aaa + '.png',
            bbox_inches='tight')

plt.figure(fig_params)
plt.subplot(1, number_of_params, param+1)
plt.legend(loc=2, bbox_to_anchor=(1.02, 0.99))
plt.savefig('outputs/lhaaso/params_3_eblmodels_' + aaa + '.png',
            bbox_inches='tight')


plt.show()
