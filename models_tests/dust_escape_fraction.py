import time
import numpy as np
import matplotlib.pyplot as plt


def razzaque2009(lambda_array):
    # lambda has to be input in microns.
    yy  = np.zeros(np.shape(lambda_array))
    yy += (0.688 + 0.556 * np.log10(lambda_array)) * (lambda_array < 0.167)
    yy += (0.151 - 0.136 * np.log10(lambda_array)) * (lambda_array < 0.218) * (lambda_array > 0.167)
    yy += (1.000 + 1.148 * np.log10(lambda_array)) * (lambda_array < 0.422) * (lambda_array > 0.218) # The paper here fails?
    yy += (0.728 + 0.422 * np.log10(lambda_array)) * (lambda_array > 0.422)
    return yy


def absA_finke2022(z_array, md=1.49, nd=0.64, pd=3.4, qd=3.54):
    return md * (1. + z_array)**nd / (1. + ((1. + z_array) / pd)**qd)


def finke2022(lambda_array, z_array):
    yy  = 10**(-0.4 * absA_finke2022(z_array))
    yy *= razzaque2009(lambda_array)
    yy /= razzaque2009(0.15)
    return yy


def abs_A_kneiske2002(lambda_array, E_bv=0.15, R=3.2):
    return 0.68 * E_bv * R * (lambda_array**-1. - 0.35)


def kneiske2002(lambda_array):
    return 10**(-0.4*abs_A_kneiske2002(lambda_array))


plt.figure()
x_lambda = np.logspace(-2, 1, num=5000)
x_zetas  = np.linspace(1e-6, 6, num=7)

alpha = 1.
plt.plot(x_lambda, finke2022(x_lambda, x_zetas[0]), 'k',
         alpha=alpha, label=r'Finke2022 z=%.2f' % x_zetas[0])
for i in range(1, len(x_zetas)-1):
    alpha -= 0.15
    plt.plot(x_lambda, finke2022(x_lambda, x_zetas[i]), 'k', alpha=alpha)

alpha -= 0.15
plt.plot(x_lambda, finke2022(x_lambda, x_zetas[-1]), 'k',
         alpha=alpha, label=r'Finke2022 z=%.2f' % x_zetas[-1])

plt.plot(x_lambda, razzaque2009(x_lambda), 'r', label='Razzaque2009')
plt.plot(x_lambda, kneiske2002(x_lambda), 'limegreen', label='Kneiske2002')

plt.ylabel('Escape fraction of photons')
plt.xlabel('lambda (microns)')
plt.legend()
plt.xscale('log')
plt.ylim(0., 1.2)
plt.xlim(0.05, 10)

plt.figure()
aaa = np.linspace(0, 6)
plt.plot(aaa, absA_finke2022(aaa), label='A(z) in Finke2022')
plt.plot(x_zetas, absA_finke2022(x_zetas), 'or', label='Chosen redshifts')
plt.legend()
plt.ylabel('A(z)')
plt.xlabel('z')
plt.show()
