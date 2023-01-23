# IMPORTS -----------------------------------#
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.integrate import simpson

from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy import units as u

import dust_absorption_models as dust_abs

from hmf import MassFunction

# EBL class definition --------------------------#

# Class to calculate EBL from SSP data.
#   Calculation of emissivities and EBL afterwards using a forward model using integration over time.
class EBL_model(object):
    def __init__(self, z_array, lambda_array,
                 sfr, sfr_params,
                 path_SSP, ssp_type='SB99',
                 t_intsteps=201,
                 z_max=35, h=0.7, omegaM=0.3, omegaBar=0.0222 / 0.7 ** 2.,
                 dust_abs_models='att_kn2002',
                 axion_decay=True, axion_gamma=1e-24, massc2_axion=1.,
                 log_prints=True
                 ):

        self._ebl_intra_spline = None
        self._t2z = None
        self._z_cube = None
        self._cube = None
        self._log_t_SSP = None
        self._log_fr_SSP = None
        self._log_em_SSP = None
        self._wv_SSP = None
        self._LookbackTime_cube = None
        self._log_freq_cube = None
        self._log_integr_t_cube = None
        self._emi_spline = None
        self._ebl_tot_spline = None
        self._ebl_ssp_spline = None
        self._ebl_axion_spline = None
        self._ebl_SSP = None
        self._ebl_intrahalo = None

        self._process_time = time.process_time()
        logging.basicConfig(level='INFO',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

        self._z_array = z_array
        self._z_max = z_max
        self._lambda_array = lambda_array[::-1]
        self._freq_array = np.log10(c.value / lambda_array[::-1] * 1e6)
        self._t_intsteps = t_intsteps

        self.read_SSP_file(path_SSP, ssp_type)
        self._dust_abs_models = dust_abs_models
        # self.dust_att()

        self._h = h
        self._omegaM = omegaM
        self._omegaB0 = omegaBar
        self._cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM, Ob0=omegaBar, Tcmb0=2.7255)

        self._sfr = lambda x: eval(sfr)(sfr_params, x)
        self._sfr_params = sfr_params

        self.intcubes()

        self._axion_decay = axion_decay
        self._axion_gamma = axion_gamma * u.s ** -1
        self._axion_massc2 = massc2_axion * u.eV

        self._intrahalo_light = True

        self._log_prints = True

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
        self._sfr = self._sfr(sfr_params)
        return

    @property
    def emi_spline(self):
        return self._emi_spline

    @property
    def ebl_spline(self):
        return self._ebl_tot_spline

    @property
    def ebl_axion_spline(self):
        return self._ebl_axion_spline

    @property
    def ebl_ssp_spline(self):
        return self._ebl_ssp_spline

    @property
    def ebl_intra_spline(self):
        return self._ebl_intra_spline

    @property
    def ebl_total_spline(self):
        return self._ebl_tot_spline

    @property
    def dust_abs_model(self):
        return self._dust_abs_models

    @dust_abs_model.setter
    def dust_abs_model(self, new_dust_model):
        self._dust_abs_models = new_dust_model
        self.calc_emissivity_ssp()
        self.calc_ebl()
        return

    @property
    def axion_mass(self):
        return self._axion_massc2

    @axion_mass.setter
    def axion_mass(self, new_mass):
        self._axion_massc2 = new_mass * u.eV
        self.calc_ebl()
        return

    @property
    def axion_gamma(self):
        return self._axion_gamma

    @axion_gamma.setter
    def axion_gamma(self, new_gamma):
        self._axion_gamma = new_gamma * u.s ** -1
        self.calc_ebl()
        return

    @property
    def logging_prints(self):
        return self._log_prints

    @logging_prints.setter
    def logging_prints(self, new_print):
        self._log_prints = new_print
        return

    def logging_info(self, text):
        if self._log_prints:
            logging.info('%.2fs: %r' % (time.process_time() - self._process_time, text))
            self._process_time = time.process_time()
    
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
        self._log_t_SSP = np.log10(t)  # log(time/yrs)
        self._wv_SSP = l[::-1] * 1e-4
        self._log_fr_SSP = np.log10(c.value / l[::-1] / 1E-10)  # log(frequency/Hz)
        self._log_em_SSP = (dd[::-1] - 6. + np.log10(1E10 * c.value)
                            - 2. * self._log_fr_SSP[:, np.newaxis])  # log(em[erg/s/Hz/M_solar])
        # Sanity check
        self._log_em_SSP[np.invert(np.isfinite(self._log_em_SSP))] = -43.

        #self.logging_info('Reading SSP file')
        return

    def dust_att(self):
        self.logging_info('Calculating dust absorption')
        print('Calculating dust absorption')
        init_time = time.time()

        self._log_em_SSP += dust_abs.calculate_dust(
            self._wv_SSP, models=self._dust_abs_models, z_array=0.)[:, np.newaxis]

        dust_time = time.time()
        self.logging_info('   Set dust absorption')
        print('   Set dust absorption: %.2fs' % (dust_time - init_time))

    def intcubes(self):
        #self.logging_info('Calculating integration cubes')

        self._cube = np.ones([self._lambda_array.shape[0], self._z_array.shape[0], self._t_intsteps])
        self._log_integr_t_cube = self._cube * np.linspace(0., 1., self._t_intsteps)
        self._log_freq_cube = self._cube * self._freq_array[:, np.newaxis, np.newaxis]
        self._z_cube = self._cube * self._z_array[np.newaxis, :, np.newaxis]
        self._LookbackTime_cube = self._cube * self._cosmo.lookback_time(
            self._z_array).to(u.yr)[np.newaxis, :, np.newaxis]

        #self.logging_info('Calculate the cubes')
        return

    def plot_sfr(self):
        plt.figure()
        plt.plot(self._z_array, self._sfr(self._z_array))
        plt.yscale('log')
        plt.title('sfr(z)')
        plt.savefig('sfr.png')
        return

    def calc_emissivity_ssp(self):
        print('Emissivity')

        log_t_ssp_intcube = np.log10((self._cosmo.lookback_time(self._z_max) - self._LookbackTime_cube).to(u.yr).value)
        log_t_ssp_intcube[log_t_ssp_intcube > self._log_t_SSP[-1]] = self._log_t_SSP[-1]

        # Array of time values that we are going to integrate over (in log10)
        log_t_ssp_intcube = (log_t_ssp_intcube - self._log_t_SSP[0]) * self._log_integr_t_cube + self._log_t_SSP[0]

        self.logging_info('Set log_t_ssp_intcube')

        # Two interpolations, transforming t->z (using log10 for both of them) and a spline with the SSP data
        self._t2z = UnivariateSpline(
            np.log10(self._cosmo.lookback_time(self._z_array).to(u.yr).value), np.log10(self._z_array), s=0, k=1)
        ssp_spline = RectBivariateSpline(x=self._log_fr_SSP, y=self._log_t_SSP, z=self._log_em_SSP, kx=1, ky=1)

        self.logging_info('Set splines')

        # Initialise mask to limit integration range to SSP data (in wavelength/frequency)
        s = (self._log_freq_cube >= self._log_fr_SSP[0]) * (self._log_freq_cube <= self._log_fr_SSP[-1])

        self.logging_info('Set frequency mask')

        # Interior of emissivity integral: L{t(z)-t(z')} * dens(z') * |dt'/dz'|
        kernel_emiss = self._cube * 1E-43
        kernel_emiss[s] = (10. ** log_t_ssp_intcube[s] * np.log(10.)  # Variable change, integration over y=log10(x)
                           * 10. ** ssp_spline.ev(self._log_freq_cube[s], log_t_ssp_intcube[s])  # L(t)
                           * self._sfr(10. ** self._t2z(                                         # sfr(z(t))
                    np.log10(self._LookbackTime_cube[s].value + 10. ** log_t_ssp_intcube[s]))))

        self.logging_info('Set kernel')

        # Calculate emissivity
        em = simpson(kernel_emiss, x=log_t_ssp_intcube, axis=-1)  # [erg s^-1 Hz^-1 Mpc^-3]

        print('Calculating dust absorption')

        lem = np.log10(em)
        lem += dust_abs.calculate_dust(self._lambda_array, models=self._dust_abs_models, z_array=self._z_array)

        self.logging_info('Set dust absorption')

        lem[np.invert(np.isfinite(lem))] = -43.
        self._emi_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lem, kx=1, ky=1)

        del log_t_ssp_intcube, kernel_emiss, s, self._t2z, ssp_spline, em, lem

        self.logging_info('Calculation time for emissivity')

        return

    def calc_ebl(self):

        if self._ebl_SSP is None:
            self.calc_emissivity_ssp()

            ebl_z_intcube = self._z_cube + self._log_integr_t_cube * (np.max(self._z_array) - self._z_cube)

            self.logging_info('Calculation time for z cube')

            # Calculate integration values
            ebl_intcube = 10. ** self._emi_spline.ev(
                (self._log_freq_cube + np.log10((1. + ebl_z_intcube) / (1. + self._z_cube))).flatten(),
                ebl_z_intcube.flatten()).reshape(self._cube.shape)

            ebl_intcube /= ((1. + ebl_z_intcube) * self._cosmo.H(ebl_z_intcube).to(u.s ** -1).value)

            # Mpc^-3 -> m^-3, erg/s -> nW
            ebl_intcube *= (u.erg * u.Mpc ** -3 * u.s ** -1).to(u.nW * u.m ** -3)
            ebl_intcube *= 10. ** self._log_freq_cube * c.value / 4. / np.pi

            self.logging_info('Calculation time for ebl ssp cube')

            # EBL from SSP
            self._ebl_SSP = simpson(ebl_intcube, x=ebl_z_intcube, axis=-1)

            self.logging_info('Calculation time for ebl ssp')

            lebl = np.log10(self._ebl_SSP)
            lebl[np.isnan(lebl)] = -43.
            lebl[np.invert(np.isfinite(lebl))] = -43.
            self._ebl_ssp_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

            del ebl_z_intcube, ebl_intcube, lebl

        if self._intrahalo_light:
            if self._ebl_intra_spline is None:

                m_min = np.log10(1e9)# / self._h)
                m_max = np.log10(1e13)# / self._h)
                alpha = 1.

                Aihl = 10**-3.23
                f_ihl = lambda x: Aihl * (x / 1e12) ** 0.1

                old_spectrum = np.loadtxt('Swire_library/Ell13_template_norm.sed')
                old_spline = UnivariateSpline(old_spectrum[:, 0], old_spectrum[:, 1], s=0, k=1, ext=1)
                print(old_spectrum[abs(old_spectrum[:,0]-22000).argmin(), 1])
                old_spectrum[:, 1] *= old_spectrum[:, 0] / old_spline(22000) / 22000.
                #old_spectrum[:, 1] *= 1. / old_spline(22000) 
                print(old_spectrum[abs(old_spectrum[:,0]-22000).argmin(), 1])
		
                plt.figure()
                plt.plot(old_spectrum[:,0], old_spectrum[:,1])
                plt.xscale('log')
                plt.yscale('log')

                plt.axhline(1, color='grey')
                plt.axvline(5500)
                plt.axvline(22000)

                plt.xlabel(r'Wavelength [A]')
                plt.ylabel(r'lambda * F_lambda [erg cm-2 s-1]')

                plt.ylim(0.01, 10000)


                old_spectrum[:, 0] *= 1e-4
               
                old_spectrum_spline = UnivariateSpline(np.log10(old_spectrum[:, 0]), np.log10(old_spectrum[:, 1]), s=0, k=1)

                mf = MassFunction(cosmo_model=self._cosmo, Mmin=m_min, Mmax=m_max)

                L22 = 5.64e12 * (self._h / 0.7)**(-2) * (mf.m / 2.7e14 * self._h / 0.7)**0.72 / 2.2e-6
            
                kernel_intrahalo = np.zeros((len(self._freq_array), len(self._z_array)))

                for nzi, zi in enumerate(self._z_array):

                    mf.update(z=zi)
                
                    lambda_luminosity = ((f_ihl(mf.m)* L22 * (1 + zi) ** alpha)[:, np.newaxis] 
                    * 10**old_spectrum_spline(np.log10(self._lambda_array[np.newaxis, :]))
                          )
                    
                   
                    kernel = ( mf.m[:, np.newaxis] * np.log(10.)  # Variable change, integration over log10(M)
                      * lambda_luminosity
                      * mf.dndm[:, np.newaxis]
                      )
                    
                    kernel_intrahalo[:, nzi] = simpson(kernel, x=np.log10(mf.m), axis=0)



                plt.plot(self._lambda_array*1e4, 10**old_spectrum_spline(np.log10(self._lambda_array)))
                nu_luminosity = kernel_intrahalo * c / (10 ** self._freq_array[:, np.newaxis])**2. * u.s**2
                
                nu_luminosity *= (u.solLum.to(u.W)*u.W / (u.Mpc*self._h)**3 / u.m).to(u.erg /u.s / u.Mpc**3 / u.m)

                print('nu luminosity')
                print('%.2e %r' %(nu_luminosity[0,0].value, str(nu_luminosity[0,0].unit)))
                aaa = np.log10(nu_luminosity.value)
                aaa[np.isnan(aaa)] = -43.
                aaa[np.invert(np.isfinite(aaa))] = -43.
                nu_lumin_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=aaa, kx=1, ky=1)
                plt.figure()
                plt.plot(self._lambda_array, nu_luminosity[:,0])
                plt.xscale('log')
                plt.yscale('log')

                z_integr = self._cube * np.linspace(0., 1., self._t_intsteps)
                z_integr = self._z_array[np.newaxis, :, np.newaxis] + (self._z_max - self._z_array[np.newaxis, :, np.newaxis]) * z_integr

                kernel_ebl_intra = 10**((nu_lumin_spline.ev((self._log_freq_cube + np.log10((1. + z_integr)/ (1. + self._z_cube))).flatten(),
                        z_integr.flatten())).reshape(self._cube.shape))

                kernel_ebl_intra /= ((1. + z_integr) *  self._cosmo.H(z_integr).to(u.s ** -1).value)


                plt.figure()
                plt.plot(z_integr[0, 0, :], kernel_ebl_intra[0, 0, :])
                plt.yscale('log')

                plt.title(simpson(kernel_ebl_intra[0,0,:], x=z_integr[0,0,:]))


                self._ebl_intrahalo = simpson(kernel_ebl_intra, x=z_integr, axis=-1)

                self._ebl_intrahalo *= u.erg * u.s / u.Mpc**3
                self._ebl_intrahalo *= c**2 / (self._lambda_array[:, np.newaxis] * 1e-6 *u.m * 4. * np.pi)
                print('self.ebl intrahalo')
                print(self._ebl_intrahalo[0,0])
                self._ebl_intrahalo = self._ebl_intrahalo.to(u.nW / u.m**2).value
                print('final intrahalo')
                print(self._ebl_intrahalo[0,0])
                lebl = np.log10(self._ebl_intrahalo)
                lebl[np.isnan(lebl)] = -43.
                lebl[np.invert(np.isfinite(lebl))] = -43.
                self._ebl_intra_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)
        else:
            self._ebl_intrahalo = 0.

        if self._axion_decay:

            z_star = self._axion_massc2 \
                     / (2. * h_plank.to(u.eV * u.s) * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1) - 1.

            I_v = ((c / (4. * np.pi * u.sr)
                    * self._cosmo.Odm(0.) * self._cosmo.critical_density0.to(u.kg * u.m ** -3)
                    * c ** 2. * self._axion_gamma / self._axion_massc2.to(u.J)
                    * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1 * h_plank * (1 + self._z_cube[:, :, 0])
                    / self._cosmo.H(z_star).to(u.s ** -1)).to(u.nW * u.m ** -2 * u.sr ** -1)
                   * (z_star > self._z_cube[:, :, 0]))

            ebl_axion = I_v.value

            lebl = np.log10(ebl_axion)
            lebl[np.isnan(lebl)] = -43.
            lebl[np.invert(np.isfinite(lebl))] = -43.
            self._ebl_axion_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

            del z_star, I_v, lebl

            self.logging_info('Calculation time for ebl axions')

        else:
            ebl_axion = 0.

        # Calculation of the whole EBL
        lebl = np.log10(self._ebl_SSP + ebl_axion + self._ebl_intrahalo)
        lebl[np.isnan(lebl)] = -43.
        lebl[np.invert(np.isfinite(lebl))] = -43.
        self._ebl_tot_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

        # Free memory
        #del ebl_z_intcube, ebl_intcube, lebl

        self.logging_info('Calculation time for ebl total')

        return
