def sfr_model(zz_array, sfr_model=None, sfr_params=None, verbose=True):
    if sfr_model == 'abdollahi09':
        return abdollahi09(zz_array=zz_array, params=sfr_params,
                           verbose=verbose)

    if sfr_model == 'sfr_finke22a':
        return sfr_finke22a(zz_array=zz_array, params=sfr_params,
                            verbose=verbose)

    if sfr_model == 'sfr_cuba':
        return sfr_cuba(zz_array=zz_array, params=sfr_params,
                        verbose=verbose)
    else:
        print('No sfr model chosen.')
        return 0


def abdollahi09(zz_array, params=None, verbose=True):
    if params is None:
        params = [0.0150, 2.254, 3.347, 6.56]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return (params[0] * (1 + zz_array) ** params[1]
            / (1 + ((1 + zz_array) / params[2]) ** params[3]))


def sfr_finke22a(zz_array, params=None, verbose=True):
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
    if params is None:
        params = [6.9e-3, 0.14, 2.2, 1.5, 2.7, 4.1]
        if verbose:
            print('   -> SFR: default parameters chosen: ',
                  params)
    return ((params[0] + params[1] * (zz_array / params[2]) ** params[3])
            / (1. + (zz_array / params[4]) ** params[5]))
