# IMPORTS -----------------------------------#
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
                 axion_decay=True, axion_gamma=1e-24, massc2_axion=1.
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

        self._axion_decay = True
        self._axion_gamma = axion_gamma * u.s ** -1
        self._massc2_axion = massc2_axion * u.eV

        self._intrahalo_light = True

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
        return self._massc2_axion

    @axion_mass.setter
    def axion_mass(self, new_mass):
        self._massc2_axion = new_mass * u.eV
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
        self._log_em_SSP = (dd[::-1] - 6. + np.log10(1E10 * c.value) - 2. * self._log_fr_SSP[:,
                                                                            np.newaxis])  # log(em [erg/s/Hz/M_solar])
        # Sanity check
        self._log_em_SSP[np.invert(np.isfinite(self._log_em_SSP))] = -43.

        end_time = time.time()
        print('   Reading SSP file: %.2fs' % (end_time - init_time))
        return

    def dust_att(self):
        print('Calculating dust absorption')
        init_time = time.time()

        self._log_em_SSP += dust_abs.calculate_dust(self._wv_SSP, models=self._dust_abs_models, z_array=0.)[:,
                            np.newaxis]

        dust_time = time.time()
        print('   Set dust absorption: %.2fs' % (dust_time - init_time))

    def intcubes(self):
        print('Calculating integration cubes')
        init_time = time.time()

        self._cube = np.ones([self._lambda_array.shape[0], self._z_array.shape[0], self._t_intsteps])
        self._log_integr_t_cube = self._cube * np.linspace(0., 1., self._t_intsteps)
        self._log_freq_cube = self._cube * self._freq_array[:, np.newaxis, np.newaxis]
        self._z_cube = self._cube * self._z_array[np.newaxis, :, np.newaxis]
        self._LookbackTime_cube = self._cube * self._cosmo.lookback_time(
            self._z_array).to(u.yr)[np.newaxis, :, np.newaxis]
        int_time = time.time()
        print('   Calculate the cubes: %.2fs' % (int_time - init_time))
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
                           * 10. ** log_t_ssp_intcube[s] * np.log(10.)  # Variable change, integration over y=log10(x)
                           * self._sfr(10. ** self._t2z(
                    np.log10(self._LookbackTime_cube[s].value + 10. ** log_t_ssp_intcube[s]))))  # sfr(z)

        calc_kernel = time.time()
        print('   Set kernel: %.2fs' % (-calc_s + calc_kernel))

        # Calculate emissivity
        em = simpson(kernel_emiss, x=log_t_ssp_intcube, axis=-1)  # [erg s^-1 Hz^-1 Mpc^-3]

        print('Calculating dust absorption')
        init_time = time.time()

        lem = np.log10(em)
        lem += dust_abs.calculate_dust(self._lambda_array, models=self._dust_abs_models, z_array=self._z_array)

        dust_time = time.time()
        print('   Set dust absorption: %.2fs' % (dust_time - init_time))

        lem[np.invert(np.isfinite(lem))] = -43.
        self._emi_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lem, kx=1, ky=1)

        del log_t_ssp_intcube, kernel_emiss, s, self._t2z, ssp_spline, em, lem

        calc_emissivity = time.time()
        print('   Calculation time for emissivity: %.2fs' % (calc_emissivity - calc_kernel))

        return

    def calc_ebl(self):
        # print('EBL')
        init_time = time.time()

        def cosmo_term(zz):
            return 1. / (1.022699E-1 * self._h) / (1. + zz) / np.sqrt(1 - self._omegaM + self._omegaM * (1. + zz) ** 3.)

        if self._ebl_SSP is None:
            self.calc_emissivity_ssp()

            eblzintcube = self._z_cube + self._log_integr_t_cube * (np.max(self._z_array) - self._z_cube)

            end_z = time.time()
            print('   Calculation time for z cube: %.2fs' % (end_z - init_time))

            # Calculate integration values
            eblintcube = 10. ** self._emi_spline.ev(
                (self._log_freq_cube + np.log10((1. + eblzintcube) / (1. + self._z_cube))).flatten(),
                eblzintcube.flatten()).reshape(self._cube.shape)

            eblintcube *= cosmo_term(eblzintcube) * 1E9

            # s -> yr, Mpc^-3 -> m^-3, erg -> nJ,
            eblintcube *= (u.erg * u.Mpc ** -3 * u.s ** -1).to(u.nJ * u.m ** -3 * u.year ** -1)
            eblintcube *= 10. ** self._log_freq_cube * c.value / 4. / np.pi

            end_eblcube = time.time()
            print('   Calculation time for ebl ssp cube: %.2fs' % (end_eblcube - end_z))

            # EBL from SSP
            self._ebl_SSP = simpson(eblintcube, x=eblzintcube, axis=-1)

            end_ebl = time.time()
            print('   Calculation time for ebl ssp: %.2fs' % (end_ebl - end_eblcube))

            lebl = np.log10(self._ebl_SSP)
            lebl[np.isnan(lebl)] = -43.
            lebl[np.invert(np.isfinite(lebl))] = -43.
            self._ebl_ssp_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

            del eblzintcube, eblintcube, lebl

        if self._intrahalo_light:
            if self._ebl_intra_spline is None:

                m_min = np.log10(1e9)# / self._h)
                m_max = np.log10(1e13)# / self._h)
                alpha = 1.

                Aihl = 10**-3.23
                f_ihl = lambda x: Aihl * (x / 1e12) ** 0.1

                old_spectrum = np.loadtxt('Swire_library/Ell13_template_norm.sed')
                old_spline = UnivariateSpline(old_spectrum[:, 0], old_spectrum[:, 1], s=0, k=1, ext=1)
                old_spectrum[:, 0] *= 1e-4
                old_spectrum[:, 1] *= old_spline(5500) / old_spline(2200)
                old_spectrum_spline = UnivariateSpline(old_spectrum[:, 0], old_spectrum[:, 1], s=0, k=1, ext=1)

                mf = MassFunction(cosmo_model=self._cosmo, Mmin=m_min, Mmax=m_max)

                L22 = (5.64e12 * (self._h / 0.7) ** (-2) * (mf.m / (2.7e14 / (self._h / 0.7))) ** 0.72
                   / 2.2 * 1e-6)
            
                kernel_intrahalo = np.zeros((len(self._freq_array), len(self._z_array)))

                for nzi, zi in enumerate(self._z_array):

                    mf.update(z=zi)
                
                    lambda_luminosity = ((f_ihl(mf.m)* L22 * (1 + zi) ** alpha)[:, np.newaxis] 
                    * old_spectrum_spline(self._lambda_array[np.newaxis, :])
                          )
                    
                   
                    kernel = ( mf.m[:, np.newaxis] * np.log(10.)  # Variable change, integration over log10(M)
                      * lambda_luminosity
                      * mf.dndm[:, np.newaxis]
                      )
                    
                    kernel_intrahalo[:, nzi] = simpson(kernel, x=np.log10(mf.m), axis=0)

                print(mf.m)
                print( (u.solLum.to(u.nW)*u.nW / (u.Mpc*self._h)**3).to(u.nW / u.m**3))

                nu_luminosity = kernel_intrahalo * c.value / (10** self._freq_array[:, np.newaxis])**2.
                print('aaaa')
                print(nu_luminosity)
                aaa = np.log10(nu_luminosity)
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
                print('kernel ebl intrahalo')
                print(kernel_ebl_intra[0, 0, :])
                kernel_ebl_intra /= ((1. + z_integr) *  self._cosmo.H(z_integr).to(u.s ** -1).value)
                print()
                print('kernel ebl intrahalo after putting kernel stuff')
                print(kernel_ebl_intra[0, 0, :])
                print()
                print('z integration')
                print(z_integr[0, 0, :])
                #print()
                plt.figure()
                plt.plot(z_integr[0, 0, :], kernel_ebl_intra[0, 0, :])
                plt.yscale('log')

                plt.title(simpson(kernel_ebl_intra[0,0,:], x=z_integr[0,0,:]))

               # print(z_integr[::-1])
                self._ebl_intrahalo = simpson(kernel_ebl_intra, x=z_integr, axis=-1)
                print('self.ebl intrahalo')
                print(self._ebl_intrahalo)

                self._ebl_intrahalo *= (u.solLum.to(u.nW)*u.nW / (u.Mpc*self._h)**3).to(u.nW / u.m**3).value
                self._ebl_intrahalo *= c.value**2 / (self._lambda_array[:, np.newaxis] * 1e-6 * 4. * np.pi)
                self._ebl_intrahalo *= 1e10
                print('self.ebl intrahalo')
                print(self._ebl_intrahalo)
                lebl = np.log10(self._ebl_intrahalo)
                lebl[np.isnan(lebl)] = -43.
                lebl[np.invert(np.isfinite(lebl))] = -43.
                self._ebl_intra_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)
        else:
            self._ebl_intrahalo = 0.

        if self._axion_decay:
            start_ebl_axion = time.time()

            z_star = self._massc2_axion \
                     / (2. * h_plank.to(u.eV * u.s) * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1) - 1.

            I_v = ((c / (4. * np.pi * u.sr)
                    * self._cosmo.Odm(0.) * self._cosmo.critical_density0.to(u.kg * u.m ** -3)
                    * c ** 2. * self._axion_gamma / self._massc2_axion.to(u.J)
                    * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1 * h_plank * (1 + self._z_cube[:, :, 0])
                    / self._cosmo.H(z_star).to(u.s ** -1)).to(u.nW * u.m ** -2 * u.sr ** -1)
                   * (z_star > self._z_cube[:, :, 0]))

            ebl_axion = I_v.value

            lebl = np.log10(ebl_axion)
            lebl[np.isnan(lebl)] = -43.
            lebl[np.invert(np.isfinite(lebl))] = -43.
            self._ebl_axion_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

            del z_star, I_v, lebl

            end_ebl_axion = time.time()
            #print('   Calculation time for ebl axions: %.2fs' % (end_ebl_axion - start_ebl_axion))

        else:
            ebl_axion = 0.

        # Calculation of the whole EBL
        start_ebl_total = time.time()
        lebl = np.log10(self._ebl_SSP + ebl_axion + self._ebl_intrahalo)
        lebl[np.isnan(lebl)] = -43.
        lebl[np.invert(np.isfinite(lebl))] = -43.
        self._ebl_tot_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=lebl, kx=1, ky=1)

        # Free memory
        #del eblzintcube, eblintcube, lebl

        end_ebl_total = time.time()
        #print('   Calculation time for ebl total: %.2fs' % (end_ebl_total - start_ebl_total))

        return
