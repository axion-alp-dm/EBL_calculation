import numpy as np


def calculate_dust(model_lambda, model_z, wv_array, z_array, **kwargs):
    """
    Function to calculate the fraction of photons which will escape absorption by dust. Dependencies both with
    wavelength and redshift, but can decide to be redshift independent as well.

    Parameters:
    :param model_lambda: string
        Model of dust fraction as a function of wavelength. If it does not correspond to any listed model,
        no dust absorption will be calculated.
        Accepted values: kneiste2002, razzaque2009

    :param model_z: string
        Model of dust absorption as a function of redshift. If it does not correspond to any listed model,
        dust absorption will not have redshift dependency.
        Accepted values: abdollahi2018

    :param wv_array: float or array
        Wavelength values at which to calculate the dust absorption.

    :param z_array: float or array
        Redshift values at which to calculate the dust absorption.

    :param kwargs: individual floats
        Desired parameters for any of the listed functions of dust absorption. Be careful when adding new functions,
        not to overlap two different parameters under the same name.


    Outputs:
    :return: 1D array with len(wv_array)
        For now, the function returns a 1D array. Redshift variability is implemented just as a float.
    """

    dust_att = np.zeros(np.shape(wv_array))

    # Wavelength dependency
    if model_lambda == 'kneiste2002':
        dust_att = kneiste2002(wv_array, **kwargs)

    if model_lambda == 'razzaque2009':
        dust_att = razzaque2009(wv_array)

    else:
        print('   -> No dust absorption definition dependency with lambda.')

    # Redshift dependency
    if model_z == 'absA_finke2022':
        dust_att *= abdollahi2018(z_array, **kwargs)

    else:
        print('   -> No dust absorption definition dependency with redshift.')

    return dust_att


def kneiste2002(wv, Ebv=0.15, R=3.2):
    """
    Dust attenuation as a function of wavelength following Kneiste02 or 0202104

    :param wv: float or array
        Wavelength values to compute dust absorption.
    :param Ebv: float
        E(B-V) or color index
    :param R: float
        Random index
    """
    return -.4 * Ebv * .68 * R * (1. / wv - .35)


def razzaque2009(lambda_array):
    """
    Dust attenuation as a function of wavelength following Razzaque09 or 0807.4294

    :param lambda_array: float or array
        Wavelength values to compute dust absorption.
    """
    # lambda has to be input in microns.
    yy  = np.zeros(np.shape(lambda_array))
    yy += (0.688 + 0.556 * np.log10(lambda_array)) * (lambda_array < 0.165)
    yy += (0.151 - 0.136 * np.log10(lambda_array)) * (lambda_array < 0.220) * (lambda_array > 0.165)
    yy += (1.000 + 1.148 * np.log10(lambda_array)) * (lambda_array < 0.422) * (lambda_array > 0.220)
    yy += (0.728 + 0.422 * np.log10(lambda_array)) * (lambda_array > 0.422)
    return yy


def abdollahi2018(z_array, md=1.49, nd=0.64, pd=3.4, qd=3.54):
    """
    Dust attenuation as a function of redshift following Abdollahi18 or 1812.01031, in the supplementary material.
    (supplement in https://pubmed.ncbi.nlm.nih.gov/30498122/)

    :param z_array:float or array
        Redshift values to compute dust absorption.
    :param md: float
        Parameter following the fitting of the paper.
    :param nd: float
        Parameter following the fitting of the paper.
    :param pd: float
        Parameter following the fitting of the paper.
    :param qd: float
        Parameter following the fitting of the paper.
    """
    return md * (1. + z_array)**nd / (1. + ((1. + z_array) / pd)**qd)
