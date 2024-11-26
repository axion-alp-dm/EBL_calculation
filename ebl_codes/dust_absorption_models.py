import numpy as np


def calculate_dust(wv_array, z_array=0.,
                   models=None,
                   dust_params=None,
                   verbose=False):
    """
    Function to calculate the fraction of photons which will escape
    absorption by dust. Dependencies both with wavelength and redshift,
    but can decide to be redshift independent as well.
    The result is given in units of log10(absorption) because of
    its use in the EBL_class.

    Parameters:
    :param wv_array: float or array  [microns]
        Wavelength values at which to calculate the dust absorption.

    :param z_array: float or array
        Redshift values at which to calculate the dust absorption.

    :param models: list of strings or None
        Models of dust fraction as a function of wavelength
        and redshift.

        -> If 2 strings are given, the first model is applied for
            wavelength and the second for redshift dependence.
            The models will usually assume that these two dependencies
            are multiplied.
            If either of them does not correspond to any listed models,
            no dust absorption will be calculated.
            - Wavelength accepted values: kneiske2002, razzaque2009
            - Redshift accepted values: abdollahi2018

        -> If 1 string is given, a combined wavelength and redshift
            model will be applied.
            - Accepted values: finke2022

        -> If the number of strings is neither 1 nor 2, no dust
            absorption model will be applied.

    :param dust_params: dictionary
        Desired parameters for any of the listed functions of dust
        absorption. Be careful when adding new functions,
        not to overlap two different parameters under the same name.

    :param verbose: bool
        Choose whether to print all the outputs of the calculations.


    Outputs:
    :return: 2D array with shape (wv_array x z_array)
        Result of the calculation in log10(absorption) since it is the
        variable output for EBL_class.
        For now, the function returns a 1D array. Redshift variability
        is implemented just as a float in the EBL class.
    """
    if np.shape(z_array) == ():
        dust_att = np.zeros([np.shape(wv_array)[0], 1])
    else:
        dust_att = np.zeros([np.shape(wv_array)[0], np.shape(z_array)[0]])

    # The absorption models are defined in one definition
    if len(models) == 1:
        if models[0] == 'finke2022':
            dust_att = finke2022(
                wv_array, z_array, dust_params, verbose=verbose)

        elif models[0] == 'finke2022_2':
            dust_att = finke2022_2(
                wv_array, z_array, dust_params, verbose=verbose)

        else:
            print('   -> No dust absorption dependency with either'
                  ' wavelength or redshift.')

    # The absorption models for wavelength and redshift are not defined
    # together
    elif len(models) == 2:
        # Wavelength dependency
        if models[0] == 'kneiske2002':
            dust_att += kneiske2002(
                wv_array[:, np.newaxis], dust_params, verbose=verbose)

        elif models[0] == 'razzaque2009':
            dust_att += razzaque2009(
                wv_array[:, np.newaxis], dust_params, verbose=verbose)

        else:
            print('   -> No dust absorption dependency with wavelength.')

        # Redshift dependency
        if models[1] == 'abdollahi2018':
            dust_att += abdollahi2018(
                z_array[np.newaxis, :], dust_params, verbose=verbose)

        else:
            print('   -> No dust absorption dependency with redshift.')
            # continue

    else:
        print('   -> No dust absorption model chosen.')
        print('   -> CAREFUL: has the model been correctly chosen?')
        print('   -> Number of name inputs in the array: ', len(models))

    dust_att[np.isnan(dust_att)] = -43.
    dust_att[np.invert(np.isfinite(dust_att))] = -43.
    return dust_att


def kneiske2002(wv, dust_params, verbose=True):
    """
    Dust attenuation as a function of wavelength following
    Kneiske02 or 0202104

    :param wv: float or array  [microns]
        Wavelength values to compute dust absorption.
    :param Ebv_Kn02: float
        E(B-V) or color index
    :param R_Kn02: float
        Random index
    """
    try:
        Ebv_Kn02 = dust_params['Ebv_Kn02']
    except:
        Ebv_Kn02 = 0.15
        if verbose:
            print('   -> Default parameter for Ebv_Kn02 chosen: ',
                  Ebv_Kn02)

    try:
        R_Kn02 = dust_params['R_Kn02']
    except:
        R_Kn02 = 3.2
        if verbose:
            print('   -> Default parameter for R_Kn02 chosen: ',
                  R_Kn02)

    return (np.minimum(-.4 * Ebv_Kn02 * .68 * R_Kn02
                       * (1. / wv - .35), 0.))


def razzaque2009(lambda_array, dust_params, verbose=True):
    """
    Dust attenuation as a function of wavelength following
    Razzaque09 or 0807.4294

    :param lambda_array: float or array  [microns]
        Wavelength values to compute dust absorption.
    """
    try:
        lambda_cuts_rz09 = dust_params['lambda_cuts_rz09']
    except:
        lambda_cuts_rz09 = [0.165, 0.220, 0.422]
        if verbose:
            print('   -> Default parameters for '
                  'lambda_cuts_rz09 chosen: ', lambda_cuts_rz09)

    try:
        initial_value_rz09 = dust_params['initial_value_rz09']
    except:
        initial_value_rz09 = [0.688, 0.151, 1.0, 0.728]
        if verbose:
            print('   -> Default parameters for '
                  'initial_value_rz09 chosen: ', initial_value_rz09)

    try:
        multipl_factor_rz09 = dust_params['multipl_factor_rz09']
    except:
        multipl_factor_rz09 = [0.556, -0.136, 1.148, 0.422]
        if verbose:
            print('   -> Default parameters for '
                  'multipl_factor_rz09 chosen: ', multipl_factor_rz09)

    yy = np.zeros(np.shape(lambda_array))
    yy += ((initial_value_rz09[0]
            + multipl_factor_rz09[0] * np.log10(lambda_array))
           * (lambda_array < lambda_cuts_rz09[0]))
    yy += ((initial_value_rz09[1]
            + multipl_factor_rz09[1] * np.log10(lambda_array))
           * (lambda_array < lambda_cuts_rz09[1])
           * (lambda_array > lambda_cuts_rz09[0]))
    yy += ((initial_value_rz09[2]
            + multipl_factor_rz09[2] * np.log10(lambda_array))
           * (lambda_array < lambda_cuts_rz09[2])
           * (lambda_array > lambda_cuts_rz09[1]))
    yy += ((initial_value_rz09[3]
            + multipl_factor_rz09[3] * np.log10(lambda_array))
           * (lambda_array > lambda_cuts_rz09[2]))
    return np.log10(yy)


def abdollahi2018(z_array, params_dust=None, verbose=True):
    """
    Dust attenuation as a function of redshift following Abdollahi18
    or 1812.01031, in the supplementary material.
    (supplement in https://pubmed.ncbi.nlm.nih.gov/30498122/).
    Result in log10(dust_att).

    :param z_array: float or array
        Redshift values to compute dust absorption.
    :param params_ab18: array or None
        Parameters following the fitting of the paper.
        Order: [m_d, n_d, p_d, q_d]
    """
    try:
        params_ab18 = params_dust['params_ab18']
    except:
        params_ab18 = [1.49, 0.64, 3.4, 3.54]
        if verbose:
            print('   -> Default parameters for params_ab18 chosen: ',
                  params_ab18)

    return (-0.4 * params_ab18[0] * (1. + z_array) ** params_ab18[1]
            / (1. + ((1. + z_array) / params_ab18[2]) ** params_ab18[3]))


def dust_att_finke2(lambda_array, params_dust=None, verbose=True):
    """

    :param lambda_array:
    :param lambda_steps_fn22:
    :param fesc_steps_fn22:
    :return:
    """
    try:
        lambda_steps_fn22 = params_dust['lambda_steps_fn22']
    except:
        lambda_steps_fn22 = [0.15, 0.167, 0.218, 0.422, 2.]
        if verbose:
            print('   -> Default parameters for '
                  'lambda_steps_fn22 chosen: ', lambda_steps_fn22)

    try:
        fesc_steps_fn22 = params_dust['fesc_steps_fn22']
    except:
        fesc_steps_fn22 = np.array([1.88, 2.18, 2.93, 3.93, 8.57]) * 0.1
        if verbose:
            print('   -> Default parameters for '
                  'fesc_steps_fn22 chosen: ', fesc_steps_fn22)

    yy = np.zeros(np.shape(lambda_array))
    yy += ((fesc_steps_fn22[1]
            + (fesc_steps_fn22[1] - fesc_steps_fn22[0])
            / (np.log10(lambda_steps_fn22[1] / lambda_steps_fn22[0]))
            * (np.log10(lambda_array) - np.log10(lambda_steps_fn22[1])))
           * (lambda_array <= lambda_steps_fn22[1]))
    yy += ((fesc_steps_fn22[2]
            + (fesc_steps_fn22[2] - fesc_steps_fn22[1])
            / (np.log10(lambda_steps_fn22[2] / lambda_steps_fn22[1]))
            * (np.log10(lambda_array) - np.log10(lambda_steps_fn22[2])))
           * (lambda_array <= lambda_steps_fn22[2])
           * (lambda_array > lambda_steps_fn22[1]))
    yy += ((fesc_steps_fn22[3]
            + (fesc_steps_fn22[3] - fesc_steps_fn22[2])
            / (np.log10(lambda_steps_fn22[3] / lambda_steps_fn22[2]))
            * (np.log10(lambda_array) - np.log10(lambda_steps_fn22[3])))
           * (lambda_array <= lambda_steps_fn22[3])
           * (lambda_array > lambda_steps_fn22[2]))
    yy += ((fesc_steps_fn22[4]
            + (fesc_steps_fn22[4] - fesc_steps_fn22[3])
            / (np.log10(lambda_steps_fn22[4] / lambda_steps_fn22[3]))
            * (np.log10(lambda_array) - np.log10(lambda_steps_fn22[4])))
           * (lambda_array > lambda_steps_fn22[3]))
    return np.log10(yy)


def finke2022(lambda_array, z_array, dust_params, verbose=True):
    """
    Dust attenuation as a function of wavelength and redshift
    following Finke22 or 2210.01157

    :param lambda_array:  float or array  [microns]
        Wavelength values to compute dust absorption.
    :param z_array:  float or array
        Redshift values to compute dust absorption.
    :return:
    """
    if np.shape(z_array) == ():
        yy = np.zeros([np.shape(lambda_array)[0], 1])
    else:
        yy = np.zeros([np.shape(lambda_array)[0], np.shape(z_array)[0]])

    yy += abdollahi2018(z_array, dust_params, verbose=verbose)
    yy += (razzaque2009(
        lambda_array, dust_params, verbose=verbose)[:, np.newaxis]
           - razzaque2009(0.15, dust_params, verbose=verbose))
    return np.minimum(yy, 0)


def finke2022_2(lambda_array, z_array, dust_params, verbose=True):
    """
    Dust attenuation as a function of wavelength and redshift
    following Finke22 or 2210.01157.
    Following the second definition, formula 13.

    :param lambda_array:  float or array  [microns]
        Wavelength values to compute dust absorption.
    :param z_array:  float or array
        Redshift values to compute dust absorption.
    :return:
    """
    if np.shape(z_array) == ():
        yy = np.zeros([np.shape(lambda_array)[0], 1])
    else:
        yy = np.zeros([np.shape(lambda_array)[0], np.shape(z_array)[0]])

    yy += abdollahi2018(z_array, dust_params, verbose=verbose)
    yy += (dust_att_finke2(
        lambda_array, dust_params, verbose=verbose)[:, np.newaxis]
           - dust_att_finke2(0.15, dust_params))

    return np.minimum(yy, 0)
