def madau14(zz_array, params=None, verbose=True):
    """
    Eq. 15](https://www.annualreviews.org/content/journals/10.1146/annurev-astro-081811-125615#f9
    :param zz_array:
    :param params:
    :param verbose:
    :return:
    """
    if params is None:
        params = [0.015, 2.7, 2.9, 5.6]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return (params[0] * (1 + zz_array) ** params[1]
            / (1 + ((1 + zz_array) / params[2]) ** params[3]))


def sfr_finke22a(zz_array, params=None, verbose=True):
    """

    :param zz_array:
    :param params:
    :param verbose:
    :return:
    """
    if params is None:
        params = [-2.04, 2.81, 1.25, -1.25, -1.84, -4.40, 1., 2., 3., 4.]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return (10 ** params[0] * (
            ((1 + zz_array) ** params[1] * (zz_array < params[-4]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (1 + zz_array) **
               params[2] * (zz_array >= params[-4]) * (zz_array < params[-3]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (
            1 + params[-3]) ** (params[2] - params[3]) * (1 + zz_array) **
               params[3] * (zz_array >= params[-3]) * (zz_array < params[-2]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (
            1 + params[-3]) ** (params[2] - params[3]) * (
                       1 + params[-2]) ** (params[3] - params[4]) * (
                       1 + zz_array) ** params[4] * (
                       zz_array >= params[-2]) * (zz_array < params[-1]))
            + ((1 + params[-4]) ** (params[1] - params[2]) * (
            1 + params[-3]) ** (params[2] - params[3]) * (
                       1 + params[-2]) ** (params[3] - params[4]) * (
                       1 + params[-1]) ** (params[4] - params[5]) * (
                       1 + zz_array) ** params[5] * (
                       zz_array >= params[-1]))))


def sfr_cuba(zz_array, params=None, verbose=True):
    """
    Eq. 53 from https://iopscience.iop.org/article/10.1088/0004-637X/746/2/125
    :param zz_array:
    :param params:
    :param verbose:
    :return:
    """
    if params is None:
        params = [6.9e-3, 0.14, 2.2, 1.5, 2.7, 4.1]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return ((params[0] + params[1] * (zz_array / params[2]) ** params[3])
            / (1. + (zz_array / params[4]) ** params[5]))


# -----------------------------------------------------------------------------
model_list = {
    'madau14': madau14,
    'sfr_finke22a': sfr_finke22a,
    'sfr_cuba': sfr_cuba
}


def sfr_model(zz_array, sfr_model=None, sfr_params=None, verbose=True):
    if sfr_model in model_list.keys():
        return model_list[sfr_model](
            zz_array=zz_array, params=sfr_params, verbose=verbose)
    else:
        ValueError('No sfr model chosen.\n'
                   + 'Accepted models: ' + str(model_list.keys()) + '\n'
                   + 'Model not recognized: ' + sfr_model)


sfr_model(0., sfr_model='mada14')
