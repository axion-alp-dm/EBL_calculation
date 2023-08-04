# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

from jacobi import propagate

from ebl_codes.EBL_class import EBL_model
from ebl_measurements.read_ebl_biteau import dictionary_datatype

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=24)
plt.rc('axes', titlesize=30)
plt.rc('axes', labelsize=30)
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=24)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

init_time = time.process_time()
# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


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


def input_yaml_data_into_class(yaml_data, log_prints=False):
    z_array = np.linspace(float(yaml_data['redshift_array']['zmin']),
                          float(yaml_data['redshift_array']['zmax']),
                          yaml_data['redshift_array']['zsteps'])

    lamb_array = np.logspace(np.log10(float(
        yaml_data['wavelenght_array']['lmin'])),
        np.log10(float(
            yaml_data['wavelenght_array']['lmax'])),
        yaml_data['wavelenght_array']['lfsteps'])

    return EBL_model(z_array, lamb_array,
                     h=float(yaml_data['cosmology_params']['cosmo'][0]),
                     omegaM=float(
                         yaml_data['cosmology_params']['cosmo'][1]),
                     omegaBar=float(
                         yaml_data['cosmology_params']['omegaBar']),
                     t_intsteps=yaml_data['t_intsteps'],
                     z_max=yaml_data['z_intmax'],
                     log_prints=log_prints)


# FIGURE: AXION MASS-GAYY AND MASS-GAMMA PARAMETER SPACES -------

config_data = read_config_file('scripts/input_data_iminuit_test.yml')
ebl_class = input_yaml_data_into_class(config_data)

for key in config_data['ssp_models']:
    print(key)
    ebl_class.logging_prints = True
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

# our line model, unicode parameter names are supported
plt.subplots(12)
plt.subplot(121)
plt.title(config_data['ssp_models'][key]['name'])
ebl_class.logging_prints = False


def spline_attempt(x, params):
    config_data['ssp_models'][key]['sfr_params'] = params
    return ebl_class.ebl_ssp_individualData(
        yaml_data=config_data['ssp_models'][key],
        x_data=x)


table_measurs = dictionary_datatype(
    'ebl_measurements/optical_data_2023', 'IGL', plot_measurs=True)

data_x = table_measurs['lambda']
data_x_freq = data_x.to(u.Hz, equivalencies=u.spectral())

data_y = table_measurs['nuInu']

data_yerr = table_measurs['nuInu_errp']

least_squares = LeastSquares(data_x, data_y, data_yerr, spline_attempt)

print('%.2fs' % (time.process_time() - init_time))
init_time = time.process_time()
# m = Minuit(least_squares, ([0.02, 2.05, 3.5, 5.02]))  # starting
m = Minuit(least_squares, ([0.0092, 2.79, 3.10, 6.97]))  # starting

# values
# m.limits = [[0.005, 0.020], [2., 2.9], [2.8, 3.5], [5., 5.7]]
m.limits = [[0.005, 0.019], [2., 3.5], [1., 3.], [6., 7.]]
print(m.params)

# m.migrad()  # finds minimum of least_squares function
# m.hesse()  # accurately computes uncertainties

# draw data and fitted line
xx_plot = np.logspace(-1, 1, num=100)
xx_plot_freq = np.log10(c.value / np.array(xx_plot) * 1e6)

# plt.errorbar(data_x, data_y, data_yerr,
#              color='k', fmt="o",
#              label="lower limits")
plt.plot(xx_plot, 10 ** ebl_class.ebl_ssp_spline(xx_plot_freq, 0., grid=False),
         'b',
         label="MF17")
# plt.plot(data_x, spline_attempt(x=data_x, a=0.015, b=2.7, c=2.9, d=5.6),
#          label="MD14")

config_data['ssp_models'][key]['sfr_params'] = [m.params[0].value,
                                                m.params[1].value,
                                                m.params[2].value,
                                                m.params[3].value]
ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])
plt.plot(xx_plot, 10 ** ebl_class.ebl_ssp_spline(xx_plot_freq, 0., grid=False),
         'r',
         label="fit")

# y, y_cov = propagate(lambda pars:
#                      spline_attempt(xx_plot, pars),
#                      m.values, m.covariance)
# yerr_prop = np.diag(y_cov) ** 0.5
# plt.fill_between(xx_plot, y - yerr_prop, y + yerr_prop,
#                  facecolor="C1", alpha=0.5)

# display legend with some fit info
# fit_info = [
#     f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}",
# ]
# for p, v, e in zip(m.parameters, m.values, m.errors):
#     fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

print(m.params)
print(m.values)
# plt.legend(title="\n".join(fit_info))
plt.yscale('log')
plt.xscale('log')
plt.ylim(1, 20)
plt.xlim(0.1, 10.)
# plt.xlabel(r'Frequency log10(Hz)')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ sr)')
print('%.2fs' % (time.process_time() - init_time))

plt.subplot(122)
# plt.title(config_data['ssp_models'][key]['name'])
x_sfr = np.linspace(0, 10)

sfr_md14 = ebl_class.sfr_function(
    function_input=config_data['ssp_models'][key]['sfr'],
    xx_array=x_sfr, params=[0.015, 2.7, 2.9, 5.6])
plt.plot(x_sfr, sfr_md14, color='b', label='MD14')

sfr_mdf17 = ebl_class.sfr_function(
    function_input=config_data['ssp_models'][key]['sfr'],
    xx_array=x_sfr, params=[0.0092, 2.79, 3.10, 6.97])
plt.plot(x_sfr, sfr_mdf17, color='g', label='MF17')

sfr_fit = ebl_class.sfr_function(
    function_input=config_data['ssp_models'][key]['sfr'],
    xx_array=x_sfr,
    params=[m.params[0].value, m.params[1].value,
            m.params[2].value, m.params[3].value])
plt.plot(x_sfr, sfr_fit, '-r', label='fit')

# y, y_cov = propagate(lambda pars:
#                      sfr(pars, x_sfr),
#                      m.values, m.covariance)
# yerr_prop = np.diag(y_cov) ** 0.5
# plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
#                  facecolor="C1", alpha=0.5)

plt.yscale('log')
plt.xlabel('redshift z')
plt.ylabel(r'SFR [M$_{\odot}$ yr$^{-1}$Mpc$^{-3}$]')
plt.legend()

plt.show()
