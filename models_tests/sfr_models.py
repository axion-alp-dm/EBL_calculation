import numpy as np
import matplotlib.pyplot as plt

sfr_kneiske02 = 'lambda ci, x : ci[0]*((x+1)/(ci[1]+1))**(ci[2]*(x<=ci[1]) - ci[3]*(x>ci[1]))'
sfr_params_kneiske02 = [0.15, 1.1, 3.4, 0.]
sfr_kneiske = lambda x: eval(sfr_kneiske02)(sfr_params_kneiske02, x)


sfr_madau = 'lambda ci, x : ci[0] * (1 + x)**ci[1] / (1 + ((1+x)/ci[' \
             '2])**ci[3])'

sfr_params_finke22 = [9.2e-3, 2.79, 3.10, 6.97]
sfr_finke22 = lambda x: eval(sfr_madau)(sfr_params_finke22, x)

sfr_params_madau14 = [0.015, 2.7, 2.9, 5.6]
sfr_madau14 = lambda x: eval(sfr_madau)(sfr_params_madau14, x)

sfr_params_madau17 = [0.01, 2.6, 3.2, 6.2]
sfr_madau17 = lambda x: eval(sfr_madau)(sfr_params_madau17, x)


def sfr_finke22a(x, ci):
    return (10**ci[0] * (
      ((1+x)**ci[1] * (x<ci[-4]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+     x)** ci[2]                                                     * (x>=ci[-4])*(x<ci[-3]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+ci[-3])**(ci[2]-ci[3])*(1+     x)** ci[3]                           * (x>=ci[-3])*(x<ci[-2]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+ci[-3])**(ci[2]-ci[3])*(1+ci[-2])**(ci[3]-ci[4])*(1+     x)** ci[4] * (x>=ci[-2])*(x<ci[-1]))
    + ((1+ci[-4])**(ci[1]-ci[2])*(1+ci[-3])**(ci[2]-ci[3])*(1+ci[-2])**(ci[3]-ci[4])*(1+ci[-1])**(ci[4]-ci[5])*(1+x)**ci[5] * (x>=ci[-1]))))


sfr_params_finke22a = [-2.04, 2.81, 1.25, -1.25, -1.84, -4.40, 1., 2., 3., 4.]

def sfr_cuba(zz_axis):
    return (6.9e-3 + 0.14*(zz_axis/2.2)**1.5) / (1. + (zz_axis/2.7)**4.1)



plt.figure()
x_array = np.linspace(0, 10, num=500)

plt.plot(x_array, sfr_kneiske(x_array), label='Kneiske02')
plt.plot(x_array, sfr_madau14(x_array), label='MD14')
plt.plot(x_array, sfr_madau17(x_array), label='MF17')
plt.plot(x_array, sfr_finke22(x_array), label='finke22: MD14')
plt.plot(x_array, sfr_finke22a(x_array, sfr_params_finke22a),
         label='finke22: piece')
plt.plot(x_array, sfr_cuba(zz_axis=x_array), label='CUBA')

plt.yscale('log')
plt.ylabel(r'sfr(z) [M$_{\odot}$year$^{-1}$ Mpc$^{-3}$]')
plt.xlabel('z')
plt.legend()

def metall_mean(zz, args=[0.153, 0.074, 1.34, 0.02]):
    return 10 ** (args[0] - args[1] * zz ** args[2]) * args[3]
plt.figure()
plt.plot(x_array, metall_mean(x_array))
plt.plot(x_array, metall_mean(x_array, args=[-1.,0.01, 2., 0.02]))
plt.yscale('log')

plt.show()
