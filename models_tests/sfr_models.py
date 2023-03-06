import numpy as np
import matplotlib.pyplot as plt

sfr_kneiske02 = 'lambda ci, x : ci[0]*((x+1)/(ci[1]+1))**(ci[2]*(x<=ci[1]) - ci[3]*(x>ci[1]))'
sfr_params_kneiske02 = [0.15, 1.1, 3.4, 0.]
sfr_kneiske = lambda x: eval(sfr_kneiske02)(sfr_params_kneiske02, x)


sfr_finke22 = 'lambda ci, x : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[2])**ci[3])'
sfr_params_finke22 = [9.2e-3, 2.79, 3.10, 6.97]
sfr_finke = lambda x: eval(sfr_finke22)(sfr_params_finke22, x)


def sfr_finke22a(x, ci):
    return (10**ci[0] * (
      ((1+x)**ci[1] * (x<ci[-4]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+     x)** ci[2]                                                     * (x>=ci[-4])*(x<ci[-3]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+ci[-3])**(ci[2]-ci[3])*(1+     x)** ci[3]                           * (x>=ci[-3])*(x<ci[-2]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+ci[-3])**(ci[2]-ci[3])*(1+ci[-2])**(ci[3]-ci[4])*(1+     x)** ci[4] * (x>=ci[-2])*(x<ci[-1]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+ci[-3])**(ci[2]-ci[3])*(1+ci[-2])**(ci[3]-ci[4])*(1+ci[-1])**(ci[4]-ci[5])*(1+x)**ci[5] * (x>=ci[-1]))))


sfr_params_finke22a = [-2.04, 2.81, 1.25, -1.25, -1.84, -4.40, 1., 2., 3., 4.]


plt.figure()
x_array = np.linspace(0, 10, num=500)

plt.plot(x_array, sfr_kneiske(x_array), label='Kneiske02')
plt.plot(x_array, sfr_finke(x_array), label='finke22: MD14')
plt.plot(x_array, sfr_finke22a(x_array, sfr_params_finke22a), label='finke22: piece')

plt.yscale('log')
plt.ylabel(r'sfr(z) [M$_{\odot}$year$^{-1}$ Mpc$^{-3}$]')
plt.xlabel('z')
plt.legend()

plt.show()
