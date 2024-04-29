import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data
from scipy.interpolate import UnivariateSpline

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

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

waves_ebl = np.geomspace(0.1, 10, num=200)
freq_array_ebl = np.log10(3e8 / (waves_ebl * 1e-6))


input_file_dir = ('outputs/final_outputs_Zevol_fixezZsolar '
                  '2024-04-11 13:41:34/')
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

config_data = read_config_file(input_file_dir + 'input_data.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=True)
ebl_class.ebl_ssp_calculation(config_data['ssp_models']['SB99_dustFinke'])
data_y_SB99 = 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0., grid=False)

model_ssp_spline = UnivariateSpline(waves_ebl, data_y_SB99, k=1, s=0)

ihl_data = np.loadtxt('outputs/ihl_spline.txt')
ihl_spline = UnivariateSpline(ihl_data[:, 0], ihl_data[:, 1], k=1, s=0)

config_data = read_config_file(
    'scripts/input_files/input_data_stripped.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=True)
ebl_class.ebl_ssp_calculation(
    config_data['ssp_models']['SB99_dustFinkestrippedonly'])
data_y_stripped = 10 ** ebl_class.ebl_ssp_spline(
        freq_array_ebl, 0., grid=False)

model_stripped_spline = UnivariateSpline(
    waves_ebl, data_y_stripped, k=1, s=0)

fig, ax = plt.subplots(figsize=(10, 8))
plt.loglog(waves_ebl, data_y_SB99, label='Our model', color='b', lw=2)
plt.loglog(ihl_data[:, 0], ihl_data[:, 1], label='IHL', color='r', lw=2)
plt.loglog(waves_ebl, data_y_stripped, label='Stripped', color='orange',
           lw=2)
plt.loglog(waves_ebl,
           model_ssp_spline(waves_ebl)
           + ihl_spline(waves_ebl)
           + model_stripped_spline(waves_ebl),
           label='Total', color='fuchsia', lw=2, ls='--')

plt.legend()

_, _ = import_cb_data(
    lambda_min_total=0.,
    lambda_max_total=5.,
    ax1=ax, plot_measurs=True)

plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

plt.xlim(0.1, 6)
plt.ylim(0.0015, 130)

plt.xscale('log')
plt.yscale('log')

plt.savefig('outputs/figures_paper/metall_comparison.png',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/metall_comparison.pdf',
            bbox_inches='tight')
plt.show()
