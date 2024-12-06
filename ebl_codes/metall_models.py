import numpy as np

def metall_model(zz_array, metall_model=None, metall_params=None,
                 verbose=True):
    if metall_model == 'tanikawa22':
        return tanikawa22(zz_array=zz_array, params=metall_params,
                           verbose=verbose)

    if metall_model == 'constant':
        return constant(zz_array=zz_array, params=metall_params,
                        verbose=verbose)

    else:
        print('No metallicity model chosen.')
        return 0


def tanikawa22(zz_array, params=None, verbose=True):
    if params is None:
        params = [ 0.153, 0.074, 1.34, 0.02]
        if verbose:
            print('   -> Metallicity: default parameters chosen: ',
                  params)
    return 10 ** (params[0] - params[1] * zz_array ** params[2]) * params[3]


def constant(zz_array, params=None, verbose=True):
    if params is None:
        params = [0.02]
        if verbose:
            print('   -> Metallicity: default parameters chosen: ',
                  params)
    return np.ones(np.shape(zz_array)) * params

