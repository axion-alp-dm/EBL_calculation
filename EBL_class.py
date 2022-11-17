# IMPORTS -----------------------------------#
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.integrate   import simpson

from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy           import units as u

import dust_absorption_models as dust_abs


# EBL class definition --------------------------#

# Class to calculate EBL from SSP data.
#   Calculation of emissivities and EBL afterwards using a forward model using integration over time.
class EBL_model(object):
    def __init__(self, z_array, lambda_array,
                 sfr, sfr_params,
                 path_SSP, ssp_type='SB99',
                 t_intsteps=201,
                 z_max=35, h=0.7, omegaM=0.3, omegaBar=0.0222/0.7**2.,
                 dust_abs_model='att_kn2002',
                 axion_decay=True
                 ):

        self._t2z = None
        self._z_cube = None
        self._cube = None
        self._log_t_SSP  = None
        self._log_fr_SSP = None
        self._log_em_SSP = None
        self._wv_SSP = None
        self._LookbackTime_cube = None
        self._log_freq_cube = None
        self._log_integr_t_cube = None
        self._emi_spline = None
        self._ebl_spline = None

        self._z_array = z_array
        self._z_max = z_max
        self._lambda_array = lambda_array[::-1]
        self._freq_array = np.log10(c.value / lambda_array[::-1] * 1e6)
        self._t_intsteps = t_intsteps

        self.read_SSP_file(path_SSP, ssp_type)
        self.dust_abs_model = dust_abs_model
        self.dust_att()

        self._h = h
        self._omegaM = omegaM
        self._omegaB0 = omegaBar
        self._cosmo = FlatLambdaCDM(H0=h*100., Om0=omegaM, Ob0=omegaBar)
        self._sfr = lambda x: eval(sfr)(sfr_params, x)
        self._sfr_params = sfr_params
        self.intcubes()

        self._axion_decay = axion_decay

    @property
    def z_array(self):
        return self._z_array

    @z_array.setter
    def z_array(self, z):
        self._z_array = z
        self.intcubes()
        return

    @property
    def lambda_array(self):
        return self._lambda_array

    @lambda_array.setter
    def lambda_array(self, mu):
        self.lambda_array = mu
        self.intcubes()
        return

    @property
    def t_intsteps(self):
        return self._t_intsteps

    @t_intsteps.setter
    def t_intsteps(self, t):
        self._t_intsteps = t
        self.intcubes()
        return

    @property
    def sfr(self):
        return self._sfr

    @sfr.setter
    def sfr(self, sfr):
        self._sfr = lambda x: eval(sfr)(self._sfr_params, x)
        return

    @property
    def sfr_params(self):
        return self._sfr_params

    @sfr_params.setter
    def sfr_params(self, sfr_params):
        self._sfr_params = sfr_params
        self._sfr = lambda x: eval(self._sfr)(sfr_params, x)
        return

    #@property
    #def


    def read_SSP_file(self, datfile, ssp_type):
        """
        Read simple stellar population model spectra from starburst 99 output:
        http://www.stsci.edu/science/starburst99/

        [erg s^-1 A^-1], 1E6 M_solar
        """
        init_time = time.time()
        d = np.loadtxt(datfile, skiprows=6)

        # Get unique time steps and frequencies, and spectral data
        t = np.unique(d[:, 0])
        l = np.unique(d[:, 1])
        dd = d[:, 3].reshape(t.shape[0], l.shape[0]).T

        # Define the quantities we will work with
        self._log_t_SSP  = np.log10(t)  # log(time/yrs)
        self._wv_SSP     = l[::-1] * 1e-4
        self._log_fr_SSP = np.log10(c.value / l[::-1] / 1E-10)  # log(frequency/Hz)
        self._log_em_SSP = (dd[::-1] - 6. + np.log10(1E10 * c.value) - 2. * self._log_fr_SSP[:, np.newaxis])  # log(em [erg/s/Hz/M_solar])
        # Sanity check
        self._log_em_SSP[np.invert(np.isfinite(self._log_em_SSP))] = -43.

        end_time = time.time()
        print('   Reading SSP file: %.2fs' % (end_time - init_time))
        return

    def dust_att(self):
        self._log_em_SSP += 0.15 * dust_abs.calculate_dust(self.dust_abs_model, self._wv_SSP)[:, np.newaxis]

    def intcubes(self):
        self._cube = np.ones([self._lambda_array.shape[0], self._z_array.shape[0], self._t_intsteps])
        self._log_integr_t_cube     = self._cube * np.linspace(0., 1., self._t_intsteps)
        self._log_freq_cube         = self._cube * self._freq_array[:, np.newaxis, np.newaxis]
        self._z_cube                = self._cube * self._z_array[np.newaxis, :, np.newaxis]
        self._LookbackTime_cube     = self._cube * self._cosmo.lookback_time(
                                                                    self._z_array).to(u.yr)[np.newaxis, :, np.newaxis]
        return

    def plot_sfr(self):
        plt.figure()
        plt.plot(self._z_array, self._sfr(self._z_array))
        plt.yscale('log')
        plt.title('sfr(z)')
        plt.savefig('sfr.png')
        return

    def calc_emissivity(self):
        print('Emissivity')
        init_time = time.time()
        log_t_ssp_intcube = np.log10((self._cosmo.lookback_time(self._z_max) - self._LookbackTime_cube).to(u.yr).value)
        log_t_ssp_intcube[log_t_ssp_intcube > self._log_t_SSP[-1]] = self._log_t_SSP[-1]

        # Array of time values that we are going to integrate over (in log10)
        log_t_ssp_intcube = (log_t_ssp_intcube - self._log_t_SSP[0]) * self._log_integr_t_cube + self._log_t_SSP[0]

        set_tssp_cube = time.time()
        print('   Set log_t_ssp_intcube: %.2fs' % (set_tssp_cube - init_time))

        # Two interpolations, transforming t->z (using log10 for both of them) and a spline with the SSP data
        self._t2z = UnivariateSpline(
            np.log10(self._cosmo.lookback_time(self._z_array).to(u.yr).value), np.log10(self._z_array), s=0, k=1)
        ssp_spline = RectBivariateSpline(x=self._log_fr_SSP, y=self._log_t_SSP, z=self._log_em_SSP, kx=1, ky=1)

        calc_splines = time.time()
        print('   Set splines: %.2fs' % (calc_splines - set_tssp_cube))

        # Initialise mask to limit integration range to SSP data
        s = (self._log_freq_cube >= self._log_fr_SSP[0]) * (self._log_freq_cube <= self._log_fr_SSP[-1])

        calc_s = time.time()
        print('   Set frequency mask: %.2fs' % (-calc_splines + calc_s))

        # Interior of emissivity integral: L{t(z)-t(z')} * dens(z') * |dt'/dz'|
        kernel_emiss = self._cube * 1E-43
        kernel_emiss[s] = (10. ** ssp_spline.ev(self._log_freq_cube[s], log_t_ssp_intcube[s])  # L(t)
                      * 10. ** log_t_ssp_intcube[s] * np.log(10.)  # Integration over y=log(x), so this is the variable change
                      * self._sfr(10. ** self._t2z(np.log10(self._LookbackTime_cube[s].value + 10. ** log_t_ssp_intcube[s])))) # sfr(z)

        calc_kernel = time.time()
        print('   Set kernel: %.2fs' % (-calc_s + calc_kernel))

        # Calculate emissivity
        em = simpson(kernel_emiss, x=log_t_ssp_intcube, axis=-1)  # [erg s^-1 Hz^-1 Mpc^-3]
        lem = np.log10(em)
        lem[np.invert(np.isfinite(lem))] = -43.
        self._emi_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lem, kx=1, ky=1)

        calc_emissivity = time.time()
        print('   Calculation time for emissivity: %.2fs' % (calc_emissivity - calc_kernel))

        plt.figure()
        wv1 = np.where(abs(self._wv_SSP-1)<0.002)
        plt.plot(self._log_t_SSP, self._log_em_SSP[wv1[0][0], :])
        plt.savefig('outputs/ssp_age.png')

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        wv = 125
        aaa = [1, 18, 54, -1]
        mark = ['^', 'v', '+', 'x']
        z_axis = 10. ** self._t2z(np.log10(self._LookbackTime_cube.value + 10. ** log_t_ssp_intcube))
        for i in range(4):
            plt.subplot(311)
            plt.plot(z_axis[wv, aaa[i], :],
                     10. ** ssp_spline.ev(self._log_freq_cube[wv, aaa[i], :],
                                          log_t_ssp_intcube)[wv, aaa[i], :],
                     marker=mark[i], label=r'%.2f' % self._z_array[aaa[i]])
            plt.subplot(312)
            plt.plot(z_axis[wv, aaa[i], :],
                     (log_t_ssp_intcube[wv, aaa[i], :]),
                     marker=mark[i])
            plt.subplot(313)
            plt.plot(z_axis[wv, aaa[i], :],
                     self._sfr(10. ** self._t2z(np.log10(self._LookbackTime_cube.value + 10. ** log_t_ssp_intcube)))[wv, aaa[i], :],
                     marker=mark[i])

        all_xlims = [-0.2, 18]

        plt.subplot(311)
        plt.yscale('log')
        plt.ylabel('L(t(z) - t(z^))')
        plt.legend(title='z')
        plt.xlim(all_xlims)

        plt.subplot(312)
        plt.ylabel('log10(t(z) - t(z^))')
        plt.xlim(all_xlims)

        plt.subplot(313)
        plt.ylabel('sfr(z)')
        plt.xlabel('z')
        plt.yscale('log')
        plt.xlim(all_xlims)

        ax_lookback = ax1.twiny()
        lookback_labels = [2, 4, 6, 8, 10, 12, 13]
        ax_lookback.set_xticks(10**self._t2z(9+np.log10(lookback_labels)))
        ax_lookback.set_xlim(all_xlims)
        ax_lookback.set_xticklabels(lookback_labels)
        ax_lookback.set_xlabel("Lookback time (Gyr)")

        plt.savefig('outputs/test2.png')

        plt.figure()
        for i in range(4):
            plt.plot(self._z_array, em[125 + i * 11, :], '.', label=r'%.2f $\mu$m' % (self._lambda_array[125 + i * 11]))
        plt.legend()
        plt.yscale('log')
        plt.ylabel(r'$\epsilon_{\nu}$ [erg/s/Hz/Mpc3]')
        plt.xlabel('z')
        plt.title('Emissivity')
        #plt.ylim(1e26, 6.7e27)
        plt.savefig('outputs/Emissivity.png')

        del log_t_ssp_intcube, kernel_emiss, s, self._t2z, ssp_spline, em, lem

        end_time = time.time()
        print('   Calculation time figures: %.2fs' % (end_time - init_time))
        return

    def calc_ebl(self):
        print('EBL')
        init_time = time.time()

        def cosmo_term(zz):
            return 1. / (1.022699E-1 * self._h) / (1. + zz) / np.sqrt(1-self._omegaM + self._omegaM * (1. + zz)**3.)

        def cosmo_term2(zz):
            return np.sqrt(1-self._omegaM + self._omegaM * (1. + zz)**3.)

        eblzintcube = self._z_cube + self._log_integr_t_cube * (np.max(self._z_array) - self._z_cube)

        end_z = time.time()
        print('   Calculation time for z cube: %.2fs' % (end_z - init_time))

        # Calculate integration values
        eblintcube = 10. ** self._emi_spline.ev(
                                (self._log_freq_cube + np.log10((1. + eblzintcube) / (1. + self._z_cube))).flatten(),
                                eblzintcube.flatten()).reshape(self._cube.shape)

        eblintcube *= cosmo_term(eblzintcube) * 1E9

        # s -> yr, Mpc^-3 -> m^-3, erg -> nJ,
        eblintcube *= (u.erg * u.Mpc**-3 * u.s**-1).to(u.nJ * u.m**-3 * u.year**-1)
        eblintcube *= 10. ** self._log_freq_cube * c.value / 4. / np.pi

        end_eblcube = time.time()
        print('   Calculation time for ebl ssp cube: %.2fs' % (end_eblcube - end_z))


        # EBL from SSP
        ebl_SSP = simpson(eblintcube, x=eblzintcube, axis=-1)

        end_ebl = time.time()
        print('   Calculation time for ebl ssp: %.2fs' % (end_ebl - end_eblcube))

        if self._axion_decay:
            version2004 = False
            if version2004:
                integration_cube = self._cube * 1E-43
                Lh = 95/0.7
                wv_a = 2.48 * 1e-6
                s = abs(np.log10(c.value / 10**self._log_freq_cube / (1 + eblzintcube) / wv_a)) < np.log10(1.5)

                integration_cube[s] = Lh / (1 + eblzintcube[s])**3. / cosmo_term2(eblzintcube[s])
                integration_cube *= c.value / 4 / np.pi / (self._h * 100) * (0.010 * self._h**3.)

                ebl_axion = simpson(integration_cube, x=eblzintcube, axis=-1)

            else:
                ff = 1. # Fraction of axions that decay into photons
                tau = 8e-24 * u.s**-1
                massc2_axion = 12 * u.eV
                #print(self._cosmo.Odm(0.))
                #print(self._cosmo.critical_density0)
                #print(self._cosmo.H(eblzintcube[0, 0, 0]).to(u.s**-1))
                I_wv = (c / (4.*np.pi) * ff * self._cosmo.Odm(0.)
                        * self._cosmo.critical_density0.to(u.kg * u.m**-3) * c**2. * tau
                        / ((c/10**self._log_freq_cube[:, :, 0]/u.s**-1) * (1 + self._z_cube[:, :, 0])
                        * (self._cosmo.H(self._z_cube[:, :, 0]).to(u.s**-1)))).to(u.nW*u.m**-3)#2*u.micron**-1)
                #print(massc2_axion.to(u.J))
                #print(h)
                #I_wv *= c/10**self._log_freq_cube[:, :, 0]# * (10**self._log_freq_cube[:, :, -1] < (massc2_axion.to(u.J) / 2./ h_plank).to(u.s**-1).value)
                #print(I_wv[0,0])

                ebl_axion = I_wv.value

        else:
            ebl_axion = 0.

        end_ebl_axion = time.time()
        print('   Calculation time for ebl axions: %.2fs' % (end_ebl_axion - end_ebl))

        # Calculation of the whole EBL
        lebl = np.log10(ebl_SSP + ebl_axion)
        lebl[np.isnan(lebl)] = -43.
        lebl[np.invert(np.isfinite(lebl))] = -43.
        self._ebl_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

        end_ebl_total = time.time()
        print('   Calculation time for ebl total: %.2fs' % (end_ebl_total - end_ebl_axion))

        plt.figure()
        for i in range(1):
            plt.plot(self._lambda_array, ebl_SSP[:, i * 10], '.', label=r'SSP %.2f' % self._z_array[i * 10])
            if self._axion_decay:
                plt.plot(self._lambda_array, ebl_axion[:, i * 10], '.', label=r'Ax %.2f' % self._z_array[i * 10])
                plt.plot(self._lambda_array, ebl_axion[:, i * 10] + ebl_SSP[:, i * 10], '.',
                         label=r'Tot %.2f' % self._z_array[i * 10])

        plt.legend(title='z')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(r'EBL SED (nW / m$^2$ sr)')
        plt.xlim([.1, 1E3])
        plt.ylim([1E-6, 1.5*np.max(ebl_axion[:, i * 10] + ebl_SSP[:, i * 10])])
        plt.savefig('outputs/ebl.png')

        # Free memory
        del eblzintcube, eblintcube, ebl_SSP, ebl_axion, lebl

        end_time = time.time()
        print('   Calculation time for figures: %.2fs' % (end_time - end_ebl_total))
        return
