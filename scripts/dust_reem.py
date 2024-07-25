import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
# from astropy.constants import c
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from ebl_codes import dust_absorption_models as dust_abs


print(8.69 + np.log10(np.array([0.05, 0.02, 0.008, 0.004, 0.0004])/0.02))

data = fits.open('../outputs/dust_reem/Z.fits')
aaa = data[1].data

for key in aaa.dtype.names:
    print(key)
    if key != 'wavelength':
        plt.loglog(
            aaa['wavelength'],
            aaa[key]
            , label=key)
        plt.loglog(
            aaa['wavelength'],
            10**10 * aaa[key], ls='--', label=key)

        lumin = (aaa[key] * (const.L_sun.to(u.erg/u.s)).value
            / (aaa['wavelength'] * 10.))
        plt.loglog(aaa['wavelength'], lumin)
        yyy = lumin * np.log(10.) * (aaa['wavelength'] * 10.)
        print(simpson(y=yyy, x=np.log10((aaa['wavelength'] * 10.))))
plt.legend()
yyy = np.column_stack((
    aaa['nuLnu[Z=6.99103]'], aaa['nuLnu[Z=7.99103]'],
    aaa['nuLnu[Z=8.29205999]'], aaa['nuLnu[Z=8.69]'],
    aaa['nuLnu[Z=9.08794001]']))
yyy = (yyy * (const.L_sun.to(u.erg/u.s)).value
            / (aaa['wavelength'] * 10.)[:, np.newaxis])
yyy *= (1E10 * const.c.value
                    / ((const.c.value/aaa['wavelength'] * 1e9)**2.)[:,
                                                               np.newaxis])

dust_reem_spline = RectBivariateSpline(
    x=np.log10(aaa['wavelength'] * 10.),
    y=np.log10([0.0004, 0.004, 0.008, 0.02, 0.05]),
    z=yyy,
    kx=1, ky=1, s=0
)
# dust_reem_spline = RegularGridInterpolator(
#         points=(,
#                 sb99_log_time,
#                 np.log10(ssp_metall)),
#         values=ssp_log_emis,
#         method='linear',
#         bounds_error=False, fill_value=-1.
#     )

data2 = fits.open('../outputs/dust_reem/LTIR.fits')
aaa2 = data2[1].data

for key in aaa2.dtype.names:
    print(key)
    if key != 'wavelength':
        plt.loglog(aaa2['wavelength'], aaa2[key], label=key)
plt.legend()


# -------------------------------------------------------------------
# T WITH METALLICITY EVOLUTION
def load_sb99(path_ssp, pop_filename):
    ssp_metall = np.sort(np.array(os.listdir(path_ssp),
                                  dtype=float))
    print(ssp_metall)
    d = np.loadtxt(path_ssp + '/0.004/kroupa_004.spectrum1', skiprows=6)

    # Get unique time steps and frequencies, and spectral data
    t_total = np.unique(d[:, 0])
    l_total = np.unique(d[:, 1])
    print(len(t_total), np.log10(t_total[0]), np.log10(t_total[-1]))
    print(len(l_total), np.log10(l_total[0]), np.log10(l_total[-1]))

    dd_total = np.zeros((l_total.shape[0],
                         t_total.shape[0],
                         len(ssp_metall) + 1))

    for n_met, met in enumerate(ssp_metall):
        data = np.loadtxt(
            path_ssp + str(met) + '/' + pop_filename
            + str(met).replace('0.', '')
            + '.spectrum1',
            skiprows=6)

        dd_total[:, :, n_met + 1] = data[:, 2].reshape(
            t_total.shape[0],
            l_total.shape[0]).T

    ssp_metall = np.insert(ssp_metall, 0, 1e-43)
    print(ssp_metall)
    dd_total[:, :, 0] = dd_total[:, :, 1]

    # Define the quantities we will work with
    sb99_log_time = np.log10(t_total)  # log(time/yrs)
    sb99_log_freq = np.log10(  # log(frequency/Hz)
        const.c.value / l_total[::-1] / 1E-10)
    ssp_log_emis = (dd_total[::-1]  # log(em[erg/s/Hz/M_solar])
                    - 6.
                    + np.log10(1E10 * const.c.value)
                    - 2. * sb99_log_freq[:, np.newaxis,
                           np.newaxis]
                    )

    ssp_log_emis[np.isnan(ssp_log_emis)] = -43.
    ssp_log_emis[
        np.invert(np.isfinite(ssp_log_emis))] = -43.

    ssp_lumin_spline = RegularGridInterpolator(
        points=(sb99_log_freq,
                sb99_log_time,
                np.log10(ssp_metall)),
        values=ssp_log_emis,
        method='linear',
        bounds_error=False, fill_value=-1.
    )
    return ssp_lumin_spline
sb99_spline = load_sb99('../data/ssp_synthetic_spectra'
                        '/starburst99/kroupa_padova/',
                        'kroupa_')

fig = plt.figure(figsize=(10, 10))
axes = fig.gca()

lambda_array = np.logspace(2., 8., num=500)
freq = np.log10(const.c.value / lambda_array * 1e10)

frac_notDust = 1 - 10**dust_abs.calculate_dust(
    wv_array=lambda_array*1e-4,
    z_array=np.array([6., 4., 2., 1., 0.]),
    models=['finke2022_2']
)
print('aaa')
print(np.nanmin(frac_notDust))
print(np.nanmax(frac_notDust))
print(frac_notDust)

for ni, age in enumerate(np.log10([5., 10, 20, 100, 500,
                                   900]) + 6):
    color = next(axes._get_lines.prop_cycler)['color']

    for nmetall, metall in enumerate([0.0004, 0.004, 0.008, 0.02, 0.05]):
        if nmetall == 4:
            plt.plot(lambda_array,
                     sb99_spline((freq, age, np.log10(metall))),
                     linestyle='-', lw=2,
                     label='%.0f Myr' % ((10 ** age) * 1e-6),
                     color=color,
                     alpha=0.3 + nmetall / 6.)
            plt.plot(lambda_array,
                     sb99_spline((freq, age, np.log10(metall)))
                     * frac_notDust[:, nmetall],
                     linestyle='--', lw=2,
                     # label='%.0f Myr' % ((10 ** age) * 1e-6),
                     color=color,
                     alpha=0.3 + nmetall / 6.)
            print()
            lumin = 10**sb99_spline((freq, age, np.log10(metall)))\
                    * frac_notDust[:, nmetall]
            yyy = lumin * np.log(10.) * lambda_array
            dust_emitted = simpson(y=yyy, x=np.log10(lambda_array))
            print(metall, age, dust_emitted)

            # key = aaa.dtype.names[5 - nmetall]
            # print(key)
            # lumin = (aaa[key] * (const.L_sun.to(u.erg / u.s)).value
            #          / (aaa['wavelength'] * 10.))
            # plt.loglog(aaa['wavelength'], lumin)
            yyy = (dust_reem_spline(
                x=np.log10(lambda_array),
                y=np.log10(metall)
            ) * np.log(10.) * lambda_array[:, np.newaxis])
            dust_reem = simpson(y=yyy,
                                x=np.log10(lambda_array[:, np.newaxis]),
                                axis=0)

            f_tir = dust_emitted / dust_reem*1e5
            plt.plot(lambda_array,
                     # sb99_spline((freq, age, np.log10(metall)))
                     # * (1 - frac_notDust[:, nmetall])
                     np.log10(f_tir * (dust_reem_spline(
                         x=np.log10(lambda_array), y=np.log10(metall)))),
                     linestyle='dotted', lw=2,
                     color=color,
                     alpha=0.3 + nmetall / 6.)


            print()

        else:
            plt.plot(lambda_array,
                     sb99_spline((freq, age, np.log10(metall))),
                     linestyle='-', lw=2,
                     color=color,
                     alpha=0.3 + nmetall / 6.)
            plt.plot(lambda_array,
                     sb99_spline((freq, age, np.log10(metall)))
                     * frac_notDust[:, nmetall],
                     linestyle='--', lw=2,
                     # label='%.0f Myr' % ((10 ** age) * 1e-6),
                     color=color,
                     alpha=0.3 + nmetall / 6.)
            print()
            lumin = 10**sb99_spline((freq, age, np.log10(metall)))\
                    * frac_notDust[:, nmetall]
            yyy = lumin * np.log(10.) * lambda_array
            dust_emitted = simpson(y=yyy, x=np.log10(lambda_array))
            print(metall, age, dust_emitted)

            key = aaa.dtype.names[5 - nmetall]
            print(key)
            lumin = (aaa[key] * (const.L_sun.to(u.erg / u.s)).value
                     / (aaa['wavelength'] * 10.))
            plt.loglog(aaa['wavelength'], lumin)
            yyy = lumin * np.log(10.) * (aaa['wavelength'] * 10.)
            dust_reem = simpson(y=yyy, x=np.log10((aaa['wavelength'] * 10.)))
            print(dust_reem/dust_emitted)

plt.xscale('log')

plt.xlim(1e2, 1e6)
plt.ylim(24, 34)
models = ['dotted', '-']
legend22 = plt.legend(
    [plt.Line2D([], [], linewidth=2, linestyle='-',
                color=color, alpha=0.3 + i / 6.) for i in range(5)],
    [0.0004, 0.004, 0.008, 0.02, 0.05],
    loc=8, bbox_to_anchor=(0.47, 0.005),
    fontsize=20,
    title='Metallicity', title_fontsize=22)

axes.add_artist(legend22)

plt.legend(fontsize=18, title='Ages', title_fontsize=22)

plt.xlabel(r'Wavelength ($\AA$)')
plt.ylabel(r'log$_{10}$(L$_{\lambda}$ '  # /Lsun '
           r'[erg s$^{-1}$ $\mathrm{\AA}^{-1}$ M$_{\odot}^{-1}$])')
plt.show()
