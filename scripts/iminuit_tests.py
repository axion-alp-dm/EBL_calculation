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
def gamma_from_rest(mass, gay):
    return (((mass * u.eV) ** 3. * (gay * u.GeV ** -1) ** 2. / 32. /
             h_plank.to(u.eV * u.s)).to(u.s ** -1).value)


axion_mac2 = np.logspace(np.log10(3), 1, num=20)
axion_gay = np.logspace(np.log10(5e-11), -9, num=20)

axion_gamma = np.logspace(np.log10(gamma_from_rest(axion_mac2[0],
                                                   axion_gay[0])),
                          np.log10(gamma_from_rest(axion_mac2[-1],
                                                   axion_gay[-1])),
                          num=len(axion_gay))

values_gamma_array = np.zeros((len(axion_mac2), len(axion_gamma)))
values_gay_array = np.zeros((len(axion_mac2), len(axion_gay)))

config_data = read_config_file('scripts/input_data_iminuit_test.yml')
ebl_class = input_yaml_data_into_class(config_data)

for key in config_data['ssp_models']:
    print(key)
    ebl_class.logging_prints = True
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

# our line model, unicode parameter names are supported
plt.subplots(12)
plt.subplot(121)
# plt.title(config_data['ssp_models'][key]['name'])
ebl_class.logging_prints = False


def spline_attempt(x, params):
    config_data['ssp_models'][key]['sfr_params'] = params
    return ebl_class.ebl_ssp_individualData(
        yaml_data=config_data['ssp_models'][key],
        x_data=x)


# data_x = [0.153, 0.225, 0.356, 0.470, 0.618, 0.749, 0.895, 1.021, 1.252,
#           1.643, 2.150, 3.544, 4.487, 7.841]#, 23.675,
#           #70.890, 101.000, 161.000, 161.000, 249.000, 357.000, 504.000]
# data_x_freq = np.log10(c.value / np.array(data_x)[::-1] * 1e6)
#
# data_y = [1.45, 3.15, 4.03, 5.36, 7.47, 9.55, 10.15, 10.44, 10.38,
#           10.12, 8.72, 5.17, 3.60, 2.45]#, 3.01,
#           #6.90, 10.22, 16.47, 13.14, 10.00, 5.83, 2.46]
#
# data_yerr = [0.27, 0.67, 0.78, 0.93, 1.12, 1.41, 1.51, 1.60, 1.52,
#              1.51, 1.22, 0.76, 0.52, 1.11]#, 0.32,
#              # 1.36, 2.01, 6.13, 2.82, 2.10, 1.87, 2.81]

data_x = [0.356, 0.470, 0.618, 0.749, 0.895, 1.021, 1.252,
          1.643, 2.150, 3.544, 4.487]
data_x_freq = np.log10(c.value / np.array(data_x) * 1e6)

data_y = [4.03, 5.36, 7.47, 9.55, 10.15, 10.44, 10.38,
          10.12, 8.72, 5.17, 3.60]

data_yerr = [0.78, 0.93, 1.12, 1.41, 1.51, 1.60, 1.52,
             1.51, 1.22, 0.76, 0.52]

least_squares = LeastSquares(data_x, data_y, data_yerr, spline_attempt)

print('%.2fs' % (time.process_time() - init_time))
init_time = time.process_time()
m = Minuit(least_squares, ([0.02, 2.05, 3.5, 5.02]))  # starting
# values
m.limits = [[0.005, 0.020], [2., 2.9], [2.8, 3.5], [5., 5.7]]
print(m.params)

m.migrad()  # finds minimum of least_squares function
m.hesse()  # accurately computes uncertainties

# draw data and fitted line
xx_plot = np.logspace(-1, 1, num=100)
xx_plot_freq = np.log10(c.value / np.array(xx_plot) * 1e6)

plt.errorbar(data_x, data_y, data_yerr,
             markerfacecolor='k', fmt="o",
             label="lower limits")
plt.plot(xx_plot, 10 ** ebl_class.ebl_ssp_spline(xx_plot_freq, 0., grid=False),
         'b',
         label="MD14")
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

y, y_cov = propagate(lambda pars:
                     spline_attempt(xx_plot, pars),
                     m.values, m.covariance)
yerr_prop = np.diag(y_cov) ** 0.5
plt.fill_between(xx_plot, y - yerr_prop, y + yerr_prop,
                 facecolor="C1", alpha=0.5)

# display legend with some fit info
fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(data_x) - m.nfit}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

print(m.values)
plt.legend(title="\n".join(fit_info))
plt.yscale('log')
plt.xscale('log')
plt.ylim(1, 20)
plt.xlim(0.3, 5.5)
# plt.xlabel(r'Frequency log10(Hz)')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ sr)')
print('%.2fs' % (time.process_time() - init_time))


plt.subplot(122)
# plt.title(config_data['ssp_models'][key]['name'])
x_sfr = np.linspace(0, 10)
m1 = [0.015, 2.7, 2.9, 5.6]
sfr = (lambda mi, x: eval(config_data['ssp_models'][key]['sfr'])(mi, x))
plt.plot(x_sfr, sfr(m1, x_sfr), color='b', label='MD14')
m2 = [m.params[0].value, m.params[1].value,
      m.params[2].value, m.params[3].value]
plt.plot(x_sfr, sfr(m2, x_sfr), '-r', label='fit')

y, y_cov = propagate(lambda pars:
                     sfr(pars, x_sfr),
                     m.values, m.covariance)
yerr_prop = np.diag(y_cov) ** 0.5
plt.fill_between(x_sfr, y - yerr_prop, y + yerr_prop,
                 facecolor="C1", alpha=0.5)

plt.yscale('log')
plt.xlabel('z')
plt.ylabel('sfr(z)')
plt.legend()

plt.show()
