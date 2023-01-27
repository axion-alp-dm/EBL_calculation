# IMPORTS -----------------------------------#
import logging
import time
import numpy as np

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebl_codes import dust_absorption_models as dust_abs

# from hmf import MassFunction


class EBL_model(object):
    """
    Class that computes the EBL contribution coming from three sources:
    Single Stellar Populations (SSP), Intra-Halo Light (IHL) and axion decay.

    Units of returns
    -----------------
    EBL:
      -> Cubes: nW m**-2 sr**-1
      -> Splines: log10(nW m**-2 sr**-1)
    Emissivity spline: log10(erg s^-1 Hz^-1 Mpc^-3)
    """

    def logging_info(self, text):
        if self._log_prints:
            logging.info('%.2fs: %s' % (time.process_time() - self._process_time, text))
            self._process_time = time.process_time()

    def __init__(self, z_array, lambda_array,
                 h=0.7, omegaM=0.3, omegaBar=0.0222 / 0.7 ** 2.,
                 log_prints=False,
                 t_intsteps=201,
                 z_max=35
                 ):
        """
        Initialize the source class.

        Parameters
        ----------
        z_array: array-like
            Array of redshifts at which to calculate the EBL.
        lambda_array: array-like [microns]
            Array of wavelengths at which to calculate the EBL.
        h: float
            Little H0, or h == H0 / 100 km/s/Mpc
        omegaM: float
            Fraction of matter in our current Universe.
        omegaBar: float
            Fraction of baryons in our current Universe.
        log_prints: Bool
            Whether to print the loggins of our procedures or not.
        t_intsteps: int
            Number of integration steps to compute with (Simpson's method)
        z_max: float
            Maximum redshift at which we form SSPs.
        """

        self._process_time = time.process_time()
        logging.basicConfig(level='INFO',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self._log_prints = log_prints

        self._cube = None
        self._z_cube = None
        self._log_freq_cube = None
        self._steps_integration_cube = None

        self._ssp_log_time = None
        self._ssp_log_freq = None
        self._ssp_log_emis = None

        self._emi_spline = None
        self._ebl_tot_spline = None
        self._ebl_ssp_spline = None
        self._ebl_axion_spline = None
        self._ebl_intra_spline = None

        self._ebl_SSP = 0.
        self._ebl_axion = 0.
        self._ebl_intrahalo = 0.

        self._h = h
        self._omegaM = omegaM
        self._omegaB0 = omegaBar
        self._cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM, Ob0=omegaBar, Tcmb0=2.7255)

        self._z_array = z_array
        self._z_max = z_max
        self._lambda_array = lambda_array[::-1]
        self._freq_array = np.log10(c.value / lambda_array[::-1] * 1e6)
        self._t_intsteps = t_intsteps

        self.intcubes()

        return

    @property
    def emi_spline(self):
        return self._emi_spline

    @property
    def ebl_total_spline(self):
        return self._ebl_tot_spline

    @property
    def ebl_ssp_spline(self):
        return self._ebl_ssp_spline

    @property
    def ebl_intra_spline(self):
        return self._ebl_intra_spline

    @property
    def ebl_axion_spline(self):
        return self._ebl_axion_spline

    @property
    def logging_prints(self):
        return self._log_prints

    @logging_prints.setter
    def logging_prints(self, new_print):
        self._log_prints = new_print
        return

    def change_axion_contribution(self, mass, gamma):
        """
        Recalculate EBL contribution from axion decay, returned in units: nW m**-2 sr**-1.
        Based on the formula and expressions given by: http://arxiv.org/abs/2208.13794

        EBL units:
        ----------
        -> Cubes: nW m**-2 sr**-1
        -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        mass: float [eV]
            Value of (m_a * c**2) of the decaying axion.
        gamma: float [s**-1]
            Decay rate of the axion.
        """
        self.ebl_axion_calculation(axion_mass=mass, axion_gamma=gamma)
        self.ebl_sum_contributions()
        return

    def change_ssp_contribution(self, yaml_file):
        """
        Recalculate the EBL SSP contribution.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        yaml_file: dictionary
            Data necessary to reconstruct the EBL component from a SSP.
        """
        self.ebl_ssp_calculation(yaml_data=yaml_file)
        self.ebl_sum_contributions()
        return

    def change_ihl_contribution(self, a_ihl, alpha):
        """
        Recalculate EBL contribution from Intra-Halo Light (IHL), returned in units: nW m**-2 sr**-1.
        Based on the formula and expressions given by: http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter alpha, as (1 + z)**alpha.
        
         EBL units:
        ----------
        -> Cubes: nW m**-2 sr**-1
        -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        a_ihl: float
            Exponential of the IHL intensity. Default: -3.23.
        alpha: float
            Index of the redshift dependency of the IHL. Default: 1.
        """
        self.ebl_intrahalo_calculation(log10_Aihl=a_ihl, alpha=alpha)
        self.ebl_sum_contributions()
        return

    def change_H0(self, new_H0):
        """
        Reinitialize the class with a new value of H0.
        new_H0: float
            New value for H0.
        """
        self._cosmo = FlatLambdaCDM(H0=new_H0, Om0=self._omegaM, Ob0=self._omegaB0, Tcmb0=2.7255)

        self._ebl_ssp_spline = None
        self._ebl_axion_spline = None
        self._ebl_intra_spline = None

        return

    def read_SSP_file(self, data_file, ssp_type):
        """
        Read Simple Stellar Population model spectra.

        Spectra units:
        --------------
        log10(erg s^-1 A^-1).

        Starburst99 output:
        http://www.stsci.edu/science/starburst99/
        The calculations assume that the value of 'Total stellar mass' in the simulation has been left as default,
        1e6 Msun. If not, change the calculation of emissivity of -6. to log10(new total mass).

        Popstar output:
        on progress
        """
        if ssp_type == 'SB99':
            d = np.loadtxt(data_file, skiprows=6)

            # Get unique time steps and frequencies, and spectral data
            t = np.unique(d[:, 0])
            l = np.unique(d[:, 1])
            dd = d[:, 3].reshape(t.shape[0], l.shape[0]).T

            # Define the quantities we will work with
            self._ssp_log_time = np.log10(t)  # log(time/yrs)
            self._ssp_log_freq = np.log10(c.value / l[::-1] / 1E-10)  # log(frequency/Hz)
            self._ssp_log_emis = (dd[::-1] - 6. + np.log10(1E10 * c.value)
                                  - 2. * self._ssp_log_freq[:, np.newaxis])  # log(em[erg/s/Hz/M_solar])

        # if ssp_type == 'Popstar':

        # Sanity check and log info
        self._ssp_log_emis[np.isnan(self._ssp_log_emis)] = -43.
        self._ssp_log_emis[np.invert(np.isfinite(self._ssp_log_emis))] = -43.
        self.logging_info('Reading of SSP file')
        return

    def intcubes(self):
        """
        Calculation of cubes that will be globally used.

        Their shapes are:
        (len(frequency array), len(z array), integration steps)
        """

        self._cube = np.ones([self._lambda_array.shape[0], self._z_array.shape[0], self._t_intsteps])
        self._steps_integration_cube = self._cube * np.linspace(0., 1., self._t_intsteps)
        self._log_freq_cube = self._cube * self._freq_array[:, np.newaxis, np.newaxis]
        self._z_cube = self._cube * self._z_array[np.newaxis, :, np.newaxis]

        self.logging_info('Calculate the cubes')
        return

    def emissivity_ssp_calculation(self, yaml_data):
        """
        Calculation of SSP emissivity from the parameters given in the dictionary.

        Emissivity spline units:
        ----------------------------
        log10(erg s^-1 Hz^-1 Mpc^-3)

        Parameters
        ----------
        yaml_data: dict
            Data necessary to reconstruct the EBL component from a SSP.
        """
        self.logging_info('SSP parameters: %s' % yaml_data['name'])

        sfr = lambda x: eval(yaml_data['sfr'])(yaml_data['sfr_params'], x)

        self.read_SSP_file(yaml_data['path_SSP'], yaml_data['ssp_type'])

        lookback_time_cube = self._cube * self._cosmo.lookback_time(self._z_array).to(u.yr)[np.newaxis, :, np.newaxis]

        # Array of time values that we are going to integrate over (in log10)
        log_t_ssp_intcube = np.log10((self._cosmo.lookback_time(self._z_max) - lookback_time_cube).to(u.yr).value)
        log_t_ssp_intcube[log_t_ssp_intcube > self._ssp_log_time[-1]] = self._ssp_log_time[-1]

        log_t_ssp_intcube = ((log_t_ssp_intcube - self._ssp_log_time[0]) * self._steps_integration_cube
                             + self._ssp_log_time[0])

        self.logging_info('SSP emissivity: set time integration cube')

        # Two interpolations, transforming t->z (using log10 for both of them) and a bi spline with the SSP data
        t2z = UnivariateSpline(
            np.log10(self._cosmo.lookback_time(self._z_array).to(u.yr).value), np.log10(self._z_array), s=0, k=1)

        ssp_spline = RectBivariateSpline(x=self._ssp_log_freq, y=self._ssp_log_time, z=self._ssp_log_emis, kx=1, ky=1)

        self.logging_info('SSP emissivity: set splines')

        # Initialise mask to limit integration range to SSP data (in wavelength/frequency)
        s = (self._log_freq_cube >= self._ssp_log_freq[0]) * (self._log_freq_cube <= self._ssp_log_freq[-1])

        self.logging_info('SSP emissivity: set frequency mask')

        # Interior of emissivity integral: L{t(z)-t(z')} * dens(z') * |dt'/dz'|
        kernel_emiss = self._cube * 1E-43
        kernel_emiss[s] = (10. ** log_t_ssp_intcube[s] * np.log(10.)  # Variable change, integration over y=log10(x)
                           * 10. ** ssp_spline.ev(self._log_freq_cube[s], log_t_ssp_intcube[s])  # L(t)
                           * sfr(10. ** t2z(                                                     # sfr(z(t))
                                np.log10(lookback_time_cube[s].value + 10. ** log_t_ssp_intcube[s]))))

        self.logging_info('SSP emissivity: set kernel')

        # Calculate emissivity in units [erg s^-1 Hz^-1 Mpc^-3] == [erg Mpc^-3]
        emissivity = simpson(kernel_emiss, x=log_t_ssp_intcube, axis=-1)

        self.logging_info('SSP emissivity: integrate emissivity')

        # Dust absorption (applied in log10)
        log10_emiss = np.log10(emissivity)
        log10_emiss += dust_abs.calculate_dust(
            self._lambda_array, models=yaml_data['dust_abs_models'], z_array=self._z_array)

        self.logging_info('SSP emissivity: set dust absorption')

        # Spline of the emissivity
        log10_emiss[np.isnan(log10_emiss)] = -43.
        log10_emiss[np.invert(np.isfinite(log10_emiss))] = -43.
        self._emi_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=log10_emiss, kx=1, ky=1)

        # Free memory and log the time
        del log_t_ssp_intcube, kernel_emiss, s, t2z, ssp_spline, emissivity, log10_emiss
        self.logging_info('SSP emissivity: end')
        return

    def ebl_ssp_calculation(self, yaml_data):
        """
        Calculate the EBL SSP contribution.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        yaml_data: dictionary
            Data necessary to reconstruct the EBL component from a SSP.
        """

        self.emissivity_ssp_calculation(yaml_data)

        ebl_z_intcube = self._z_cube + self._steps_integration_cube * (np.max(self._z_array) - self._z_cube)

        self.logging_info('SSP EBL: calculation of z cube')

        # Calculate integration values
        ebl_intcube = 10. ** self._emi_spline.ev(
            (self._log_freq_cube + np.log10((1. + ebl_z_intcube) / (1. + self._z_cube))).flatten(),
            ebl_z_intcube.flatten()).reshape(self._cube.shape)

        ebl_intcube /= ((1. + ebl_z_intcube) * self._cosmo.H(ebl_z_intcube).to(u.s ** -1).value)

        # Mpc^-3 -> m^-3, erg/s -> nW
        ebl_intcube *= (u.erg * u.Mpc ** -3 * u.s ** -1).to(u.nW * u.m ** -3)
        ebl_intcube *= 10. ** self._log_freq_cube * c.value / 4. / np.pi

        self.logging_info('SSP EBL: calculation of kernel')

        # Integration of EBL from SSP
        self._ebl_SSP = simpson(ebl_intcube, x=ebl_z_intcube, axis=-1)

        self.logging_info('SSP EBL: integration')

        # Spline of the IHL EBL intensity
        log10_ebl = np.log10(self._ebl_SSP)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_ssp_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del ebl_z_intcube, ebl_intcube, log10_ebl
        self.logging_info('SSP EBL: done')
        return

    def ebl_intrahalo_calculation(self, log10_Aihl, alpha):
        """
        EBL contribution from Intra-Halo Light (IHL), returned in units: nW m**-2 sr**-1.
        Based on the formula and expressions given by: http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter alpha, as (1 + z)**alpha.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        log10_Aihl: float
            Exponential of the IHL intensity. Default: -3.23.
        alpha: float
            Index of the redshift dependency of the IHL. Default: 1.
        """
        Aihl = 10 ** log10_Aihl
        f_ihl = lambda x: Aihl * (x / 1e12) ** 0.1

        # log10 of masses over which we integrate the IHL contribution
        m_min = np.log10(1e9)
        m_max = np.log10(1e13)

        # Use the spectrum of a synthetic old galaxy (taken from the SWIRE library),
        # and normalize it at a wavelength of 2.2 microns
        old_spectrum = np.loadtxt('Swire_library/Ell13_template_norm.sed')
        old_spline = UnivariateSpline(old_spectrum[:, 0], old_spectrum[:, 1], s=0, k=1, ext=1)

        old_spectrum[:, 1] *= old_spectrum[:, 0] / old_spline(22000) / 22000.  # S_lambda = F_lambda * lambda
        old_spectrum[:, 0] *= 1e-4
        old_spectrum_spline = UnivariateSpline(np.log10(old_spectrum[:, 0]), np.log10(old_spectrum[:, 1]), s=0, k=1)

        # Initialize an object to calculate dn/dM
        mf = MassFunction(cosmo_model=self._cosmo, Mmin=m_min, Mmax=m_max)

        # Total luminosity of a galaxy at 2.2 microns
        L22 = 5.64e12 * (self._h / 0.7) ** (-2) * (mf.m / 2.7e14 * self._h / 0.7) ** 0.72 / 2.2e-6

        # The object to calculate dn/dM returns this quantity for a specific redshift
        kernel_intrahalo = np.zeros((len(self._freq_array), len(self._z_array)))
        for nzi, zi in enumerate(self._z_array):
            mf.update(z=zi)
            lambda_luminosity = ((f_ihl(mf.m) * L22 * (1 + zi) ** alpha)[:, np.newaxis]
                                 * 10 ** old_spectrum_spline(np.log10(self._lambda_array[np.newaxis, :]))
                                 )
            kernel = (mf.m[:, np.newaxis] * np.log(10.)  # Variable change, integration over log10(M)
                      * lambda_luminosity
                      * mf.dndm[:, np.newaxis]
                      )
            kernel_intrahalo[:, nzi] = simpson(kernel, x=np.log10(mf.m), axis=0)

        nu_luminosity = kernel_intrahalo * c / (10 ** self._freq_array[:, np.newaxis]) ** 2. * u.s ** 2
        nu_luminosity *= (u.solLum.to(u.W) * u.W / (u.Mpc * self._h) ** 3 / u.m).to(u.erg / u.s / u.Mpc ** 3 / u.m)

        # Spline of the luminosity
        log10_lumin = np.log10(nu_luminosity.value)
        log10_lumin[np.isnan(log10_lumin)] = -43.
        log10_lumin[np.invert(np.isfinite(log10_lumin))] = -43.
        nu_lumin_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=log10_lumin, kx=1, ky=1)

        # Integrate the EBL intensity kernel over values of z
        z_integr = self._z_array[np.newaxis, :, np.newaxis] + (
                self._z_max - self._z_array[np.newaxis, :, np.newaxis]) * self._steps_integration_cube

        kernel_ebl_intra = (10 ** (
            (nu_lumin_spline.ev(
                (self._log_freq_cube + np.log10((1. + z_integr) / (1. + self._z_cube))).flatten(),
                z_integr.flatten())).reshape(self._cube.shape))
                            / ((1. + z_integr) * self._cosmo.H(z_integr).to(u.s ** -1).value))

        self._ebl_intrahalo = simpson(kernel_ebl_intra, x=z_integr, axis=-1)

        self._ebl_intrahalo *= u.erg * u.s / u.Mpc ** 3
        self._ebl_intrahalo *= c ** 2 / (self._lambda_array[:, np.newaxis] * 1e-6 * u.m * 4. * np.pi)

        self._ebl_intrahalo = self._ebl_intrahalo.to(u.nW / u.m ** 2).value

        # Spline of the IHL EBL intensity
        log10_ebl = np.log10(self._ebl_intrahalo)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_intra_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del old_spectrum, old_spectrum_spline, old_spline, mf, L22, kernel_intrahalo
        del lambda_luminosity, nu_luminosity, kernel, kernel_ebl_intra, nu_lumin_spline
        del z_integr, log10_lumin, log10_ebl
        self.logging_info('Calculation time for ebl ihl')
        return

    def ebl_axion_calculation(self, axion_mass, axion_gamma):
        """
        EBL contribution from axion decay, returned in units: nW m**-2 sr**-1.
        Based on the formula and expressions given by: http://arxiv.org/abs/2208.13794

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        axion_mass: float [eV]
            Value of (m_a * c**2) of the decaying axion.
        axion_gamma: float [s**-1]
            Decay rate of the axion.
        """
        axion_mass = axion_mass * u.eV
        axion_gamma = axion_gamma * u.s ** -1

        z_star = axion_mass / (2. * h_plank.to(u.eV * u.s) * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1) - 1.

        self._ebl_axion = ((c / (4. * np.pi * u.sr)
                            * self._cosmo.Odm(0.) * self._cosmo.critical_density0
                            * c ** 2. * axion_gamma / axion_mass
                            * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1 * h_plank * (1 + self._z_cube[:, :, 0])
                            / self._cosmo.H(z_star)).to(u.nW * u.m ** -2 * u.sr ** -1)
                           * (z_star > self._z_cube[:, :, 0])).value

        # Spline of the axion EBL intensity
        log10_ebl = np.log10(self._ebl_axion)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_axion_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del z_star, log10_ebl
        self.logging_info('Calculation time for ebl axions')
        return

    def ebl_sum_contributions(self):
        """
        Sum of the contributions to the EBL which have been previously calculated.
        If any of the components has not been calculated, its contribution will be 0.
        Components are Single Stellar Populations (SSP), Intra-Halo Light (IHL) and axion decay.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)
        """
        # Spline of the total EBL intensity
        log10_ebl = np.log10(self._ebl_SSP + self._ebl_axion + self._ebl_intrahalo)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_tot_spline = RectBivariateSpline(x=self._freq_array, y=self._z_array, z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del log10_ebl
        self.logging_info('Calculation time for ebl total')
        return

    def ebl_all_calculations(self, ssp_yaml=None,
                             log10_Aihl=-3.23, alpha=1.,
                             axion_mass=1., axion_gamma=5e-23):
        """
        Calculate the EBL total contribution from our three components:
        Single Stellar Populations (SSP), Intra-Halo Light (IHL) and axion decay.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        ssp_yaml: dictionary
            Data necessary to reconstruct the EBL component from a SSP. Default: model from Kneiske02.
        log10_Aihl: float
            Exponential of the IHL intensity. Default: -3.23.
        alpha: float
            Index of the redshift dependency of the IHL. Default: 1.
        axion_mass: float [eV]
            Value of (m_a * c**2) of the decaying axion. Default: 1 eV.
        axion_gamma: float [s**-1]
            Decay rate of the axion. Default: 5e-23 s**-1.
        """
        if ssp_yaml is None:
            ssp_yaml = {'name': 'Kneiske02',
                        'sfr': 'lambda ci, x : ci[0]*((x+1)/(ci[1]+1))**(ci[2]*(x<=ci[1]) - ci[3]*(x>ci[1]))',
                        'sfr_params': [0.15, 1.1, 3.4, 0.0],
                        'ssp_type': 'SB99', 'path_SSP': 'ssp/final_run_spectrum',
                        'dust_abs_models': ['kneiske2002', 'aaaa']}

        self.ebl_ssp_calculation(ssp_yaml)
        self.ebl_intrahalo_calculation(log10_Aihl, alpha)
        self.ebl_axion_calculation(axion_mass, axion_gamma)

        self.ebl_sum_contributions()

        return
