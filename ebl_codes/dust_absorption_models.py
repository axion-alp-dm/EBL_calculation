import numpy as np


def calculate_dust(wv_array, z_array=0., models=None, **kwargs):
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

    :param kwargs: individual floats
        Desired parameters for any of the listed functions of dust
        absorption. Be careful when adding new functions,
        not to overlap two different parameters under the same name.


    Outputs:
    :return: 1D array with len(wv_array)
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
            dust_att = finke2022(wv_array, z_array)

        elif models[0] == 'finke2022_2':
            dust_att = finke2022_2(wv_array, z_array)

        else:
            print('No dust absorption dependency with either'
                  ' wavelength or redshift.')

    # The absorption models for wavelength and redshift are not defined
    # together
    elif len(models) == 2:
        # Wavelength dependency
        if models[0] == 'kneiske2002':
            dust_att += kneiske2002(wv_array[:, np.newaxis], **kwargs)

        elif models[0] == 'razzaque2009':
            dust_att += razzaque2009(wv_array[:, np.newaxis])

        else:
            print('   -> No dust absorption dependency with wavelength.')

        # Redshift dependency
        if models[1] == 'abdollahi2018':
            dust_att += abdollahi2018(z_array[np.newaxis, :], **kwargs)

        # else:
        #     print('   -> No dust absorption dependency with redshift.')

    else:
        print('   -> No dust absorption model chosen.')

    dust_att[np.isnan(dust_att)] = -43.
    dust_att[np.invert(np.isfinite(dust_att))] = -43.
    return dust_att


def kneiske2002(wv, Ebv=0.15, R=3.2):
    """
    Dust attenuation as a function of wavelength following
    Kneiske02 or 0202104

    :param wv: float or array  [microns]
        Wavelength values to compute dust absorption.
    :param Ebv: float
        E(B-V) or color index
    :param R: float
        Random index
    """
    return np.minimum(-.4 * Ebv * .68 * R * (1. / wv - .35), 0)


def razzaque2009(lambda_array):
    """
    Dust attenuation as a function of wavelength following
    Razzaque09 or 0807.4294

    :param lambda_array: float or array  [microns]
        Wavelength values to compute dust absorption.
    """
    yy = np.zeros(np.shape(lambda_array))
    yy += ((0.688 + 0.556 * np.log10(lambda_array))
           * (lambda_array < 0.165))
    yy += ((0.151 - 0.136 * np.log10(lambda_array))
           * (lambda_array < 0.220) * (lambda_array > 0.165))
    yy += ((1.000 + 1.148 * np.log10(lambda_array))
           * (lambda_array < 0.422) * (lambda_array > 0.220))
    yy += ((0.728 + 0.422 * np.log10(lambda_array))
           * (lambda_array > 0.422))
    return np.log10(yy)


def abdollahi2018(z_array, md=1.49, nd=0.64, pd=3.4, qd=3.54):
    """
    Dust attenuation as a function of redshift following Abdollahi18
    or 1812.01031, in the supplementary material.
    (supplement in https://pubmed.ncbi.nlm.nih.gov/30498122/).
    Result in log10(dust_att).

    :param z_array: float or array
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
    return -0.4 * md * (1. + z_array) ** nd / (
            1. + ((1. + z_array) / pd) ** qd)


def dust_att_finke2(lambda_array,
                    lambda_steps=None,
                    fesc_steps=None):
    """

    :param lambda_array:
    :param lambda_steps:
    :param fesc_steps:
    :return:
    """
    if lambda_steps is None:
        lambda_steps = [0.15, 0.167, 0.218, 0.422, 2.]
    if fesc_steps is None:
        fesc_steps = np.array([1.88, 2.18, 2.93, 3.93, 8.57])*0.1

    yy = np.zeros(np.shape(lambda_array))
    yy += ((fesc_steps[1] + (fesc_steps[1] - fesc_steps[0])
            / (np.log10(lambda_steps[1]) - np.log10(lambda_steps[0]))
            * (np.log10(lambda_array) - np.log10(lambda_steps[1])))
           * (lambda_array <= lambda_steps[1]))
    yy += ((fesc_steps[2] + (fesc_steps[2] - fesc_steps[1])
            / (np.log10(lambda_steps[2]) - np.log10(lambda_steps[1]))
            * (np.log10(lambda_array) - np.log10(lambda_steps[2])))
           * (lambda_array <= lambda_steps[2])
           * (lambda_array > lambda_steps[1]))
    yy += ((fesc_steps[3] + (fesc_steps[3] - fesc_steps[2])
            / (np.log10(lambda_steps[3]) - np.log10(lambda_steps[2]))
            * (np.log10(lambda_array) - np.log10(lambda_steps[3])))
           * (lambda_array <= lambda_steps[3])
           * (lambda_array > lambda_steps[2]))
    yy += ((fesc_steps[4] + (fesc_steps[4] - fesc_steps[3])
            / (np.log10(lambda_steps[4]) - np.log10(lambda_steps[3]))
            * (np.log10(lambda_array) - np.log10(lambda_steps[4])))
           * (lambda_array > lambda_steps[3]))
    return np.log10(yy)


def finke2022(lambda_array, z_array):
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
    yy += abdollahi2018(z_array)
    yy += razzaque2009(lambda_array)[:, np.newaxis] - razzaque2009(0.15)
    return np.minimum(yy, 0)


def finke2022_2(lambda_array, z_array, fesc_steps=None):
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
    yy += abdollahi2018(z_array)
    yy += (dust_att_finke2(lambda_array,
                           fesc_steps=fesc_steps)[:, np.newaxis]
           - dust_att_finke2(0.15,
                             fesc_steps=fesc_steps))

    return np.minimum(yy, 0)


# TESTS FOR DIFFERENT DUST MODELS
'''
import matplotlib.pyplot as plt

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

plt.figure()
x_lambda = np.logspace(-2, 1, num=5000)
# x_zetas = np.linspace(1e-6, 6, num=7)
x_zetas = np.array([0, 2, 4, 6])

print(dust_att_finke2(lambda_array=0.15,
                      lambda_steps=[1.88, 2.18, 2.93, 3.93, 8.57]))
print(dust_att_finke2(lambda_array=0.15,
                      lambda_steps=np.array(
                          [1.88, 2.18, 2.93, 3.93, 8.57]) * 0.1))
print(dust_att_finke2(lambda_array=0.15,
                      lambda_steps=[0.257, 0.287, 0.271, 0.628, 0.959]))

plt.plot(x_lambda,
         10 ** dust_att_finke2(x_lambda,
                           fesc_steps=np.array(
                               [1.88, 2.18, 2.93, 3.93, 8.57])),
         '.', label='model A only 13')
print(10 ** dust_att_finke2(x_lambda))
plt.plot(x_lambda,
         10 ** finke2022_2(x_lambda, z_array=0),
         '+', label='params as model A A*0.1')
plt.plot(x_lambda,
         10 ** finke2022_2(x_lambda, z_array=0.,
                           fesc_steps=np.array(
                               [1.88, 2.18, 2.93, 3.93, 8.57])),
         '.', label='params model A')
plt.plot(x_lambda,
         10 ** finke2022_2(x_lambda, z_array=0.,
                           fesc_steps=np.array(
                               [0.257, 0.287, 0.271, 0.628, 0.959])),
         '.', label='params as model B (fixed)')

x_finke = np.sort([0.08013941241504934, 0.16275654480267543,
                   0.2202666354, 0.41705401612006, 1.4732255113])
y_finke = np.sort([0.013964242228906798, 0.2719143926371579,
                   0.3792727582008306, 0.5045838705056581,
                   0.9982318077639305])

plt.plot(x_finke, y_finke, '-', label='rough Fig 10 z=0')

plt.ylabel('Escape fraction of photons')
plt.xlabel('lambda (microns)')
plt.legend()
plt.xscale('log')
# plt.show()

plt.figure()

alpha = 1.
plt.plot(x_lambda, 10 ** calculate_dust(
    x_lambda, z_array=x_zetas[0], models=['finke2022'], model_combined=True),
         'k', alpha=alpha, label=r'Finke2022 z=%.2f' % x_zetas[0])

plt.plot(x_lambda, 10**calculate_dust(
    x_lambda, z_array=x_zetas[0], models=['finke2022_2'], model_combined=True),
         'b', alpha=alpha, label=r'Finke2 z=%.2f' % x_zetas[0])

for i in range(1, len(x_zetas) - 1):
    alpha -= 0.15
    plt.plot(x_lambda, 10 ** calculate_dust(
        x_lambda, z_array=x_zetas[i], models=['finke2022'],
        model_combined=True),
             'k', alpha=alpha)
    plt.plot(x_lambda, 10**calculate_dust(
        x_lambda, z_array=x_zetas[i], models=['finke2022_2'],
        model_combined=True),
             'b', alpha=alpha)

alpha -= 0.15
plt.plot(x_lambda, 10 ** calculate_dust(
    x_lambda, z_array=x_zetas[-1], models=['finke2022'], model_combined=True),
         'k', alpha=alpha, label=r'Finke2022 z=%.2f' % x_zetas[-1])
plt.plot(x_lambda, 10**calculate_dust(
    x_lambda, z_array=x_zetas[-1], models=['finke2022_2'],
    model_combined=True),
         'b', alpha=alpha, label=r'Finke2022 z=%.2f' % x_zetas[-1])

plt.plot(x_lambda, 10 ** calculate_dust(
    x_lambda, models=['razzaque2009', 'aaa']), 'r', label='Razzaque2009')
plt.plot(x_lambda, 10 ** kneiske2002(x_lambda), 'limegreen',
         label='Kneiske2002')

plt.plot(x_lambda, 10**dust_att_finke2(x_lambda), 'orange')
plt.plot(x_finke, y_finke, '-', label='rough Fig 10 z=0')

plt.ylabel('Escape fraction of photons')
plt.xlabel('lambda (microns)')
plt.legend()
plt.xscale('log')
plt.ylim(0., 1.2)
plt.xlim(0.05, 10)

plt.figure()
aaa = np.linspace(0, 6)
plt.plot(aaa, 10 ** abdollahi2018(aaa), label='g(z) in Finke22')
plt.plot(x_zetas, 10 ** abdollahi2018(x_zetas), 'or', label='Chosen redshifts')
plt.legend()
plt.ylabel('g(z) = 10**(-0.4 A(z))')
plt.xlabel('z')

yyy = calculate_dust(x_lambda, 0, models=['aaa', 'abdollahi2018', 'aaa'])
# plt.show()
'''
