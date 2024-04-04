# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np

from ebl_codes.EBL_class import EBL_model

from data.emissivity_measurs.emissivity_read_data import emissivity_data
from data.cb_measurs.import_cb_measurs import import_cb_data
from data.sfr_measurs.sfr_read import *

from astropy import units as u
from astropy.constants import h as h_plank
from astropy.constants import c

from iminuit import Minuit
from iminuit.cost import LeastSquares

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")
direct_name = str('final_outputs_Zevol_fixezZsolar'
                  + time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime())
                  )
print(direct_name)

# If the directory for outputs is not present, create it.
if not os.path.exists("outputs/"):
    os.makedirs("outputs/")
if not os.path.exists('outputs/' + direct_name):
    os.makedirs('outputs/' + direct_name)


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


def chi2_upperlims(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2. * (x_obs < x_model))


def chi2_measurs(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2.)


config_data = read_config_file(
    'scripts/input_files/input_data_Finke.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data)

# COB measurements that we are going to use
upper_lims_ebldata, igl_ebldata = import_cb_data(
    lambda_min_total=0.1, lambda_max_total=5.,
    plot_measurs=False)

print(np.shape(igl_ebldata))

# FIGURE: sfr fit ------------------------------------------------
sfr_data = sfr_data_dict()
print(np.shape(sfr_data))
# FIGURE: EMISSIVITIES IN DIFFERENT REDSHIFTS ------------------

emiss_data = emissivity_data()
freq_emiss = c.value / (emiss_data['lambda'] * 1e-6)
print(np.shape(emiss_data))
# MINIMIZATION OF CHI2 OF SSPs
for nkey, key in enumerate(config_data['ssp_models']):
    print()
    print('SSP model: ', config_data['ssp_models'][key]['name'])


    def fit_igl(lambda_igl, params):
        config_data['ssp_models'][key]['sfr_params'] = params[0:4].copy()
        config_data['ssp_models'][key]['args_metall'] = params[4:].copy()
        return ebl_class.ebl_ssp_individualData(
            yaml_data=config_data['ssp_models'][key],
            x_data=lambda_igl)


    def fit_emiss(x_all, params):
        lambda_emiss, z_emiss = x_all
        freq_emissions = np.log10(c.value / lambda_emiss * 1e6)

        config_data['ssp_models'][key]['sfr_params'] = params[0:4].copy()
        config_data['ssp_models'][key]['args_metall'] = params[4:].copy()

        ebl_class.emiss_ssp_calculation(config_data['ssp_models'][key])

        return 10 ** (freq_emissions
                      + ebl_class.emiss_ssp_spline(freq_emissions,
                                                   z_emiss)
                      - 7)


    def sfr(x, params):
        return ebl_class.sfr_function(
            config_data['ssp_models'][key]['sfr'], x, params[0:4])


    combined_likelihood = (LeastSquares(igl_ebldata['lambda'],
                                        igl_ebldata['nuInu'],
                                        igl_ebldata['1 sigma'],
                                        fit_igl)
                           + LeastSquares((emiss_data['lambda'],
                                           emiss_data['z']),
                                          emiss_data['eje'],
                                          (emiss_data['eje_n']
                                           + emiss_data['eje_p']) / 2.,
                                          fit_emiss)
                           + LeastSquares(sfr_data[:, 0],
                                          sfr_data[:, 3],
                                          (sfr_data[:, 4]
                                           + sfr_data[:, 5]) / 2.,
                                          sfr)
                           )

    init_time = time.process_time()


    aaa = np.concatenate((config_data['ssp_models'][key]['sfr_params'],
                          config_data['ssp_models'][key]['args_metall']))
    print(aaa)

    m = Minuit(combined_likelihood, aaa)
    m.limits = [[None, None], [None, None], [None, None], [None, None],
                [-3., 0.2], [0., 2.], [0.5, 5.], [0.1, 0.25]]
    m.fixed[7] = True
    m.values[7] = 0.02
    print(m.params)

    m.migrad()  # finds minimum of least_squares function
    m.hesse()  # accurately computes uncertainties

    outputs = open('outputs/' + direct_name + '/z_fits_info.txt', 'a+')
    outputs.write(str(key) + '\n')
    outputs.write('SSP model: '
                       + str(config_data['ssp_models'][key]['name'])
                       + '\n')
    outputs.write(str(m.params) + '\n')
    outputs.write(str(m.values) + '\n')
    outputs.write(str(m.covariance) + '\n')
    outputs.write(f"$\\chi^2$/$n_\\mathrm{{dof}}$ "
                       f"= {m.fval:.1f} / {m.ndof:.0f} "
                  f"= {m.fmin.reduced_chi2:.1f}" + '\n')

    outputs.write('Individual chi2 values:\n')
    aaa = np.array(np.array(m.params.to_table()[0])[:, 2], dtype=float)
    outputs.write(
        'cob data: ' + str(chi2_measurs(
            fit_igl(igl_ebldata['lambda'], aaa),
            igl_ebldata['nuInu'], igl_ebldata['1 sigma'])) + '\n')
    outputs.write(
        'emissivities data: ' + str(chi2_measurs(
            fit_emiss((emiss_data['lambda'], emiss_data['z']), aaa),
            emiss_data['eje'],
            (emiss_data['eje_n'] + emiss_data['eje_p']) / 2.))
        + '\n')
    outputs.write(
        'sfr data: ' + str(chi2_measurs(
            sfr(sfr_data[:, 0], aaa),
            sfr_data[:, 3], (sfr_data[:, 4] + sfr_data[:, 5]) / 2.))
        + '\n')
    outputs.write('\n\n\n')
    outputs.close()

    print(m.params)
    print(m.values)
    print(m.covariance)
    print(f"$\\chi^2$/$n_\\mathrm{{dof}}$ "
          f"= {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")

    print('Fit: %.2fs' % (time.process_time() - init_time))
    init_time = time.process_time()

    print(config_data['ssp_models'][key]['sfr_params'])

    config_data['ssp_models'][key]['sfr_params'] = [m.params[0].value,
                                                    m.params[1].value,
                                                    m.params[2].value,
                                                    m.params[3].value]

    config_data['ssp_models'][key]['args_metall'] = [m.params[4].value,
                                                    m.params[5].value,
                                                    m.params[6].value,
                                                    m.params[7].value]
    ebl_class.ebl_ssp_calculation(config_data['ssp_models'][key])

    np.save('outputs/' + direct_name + '/' + key + 'spline',
            ebl_class.ebl_ssp_spline)

    ccc = []
    for i in range(len(aaa) ** 2):
        ccc.append(float(m.covariance.flatten()[i]))
    config_data['ssp_models'][key]['cov_matrix'] = ccc
    print('\n\n')

outputs = open('outputs/' + direct_name + '/input_data.yml', 'w')
yaml.dump(config_data, outputs,
          default_flow_style=False, allow_unicode=True)
outputs.close()
