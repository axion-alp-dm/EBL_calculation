# IMPORTS -----------------------------------#
import logging
import os
import time
import numpy as np

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from fast_interp import interp2d as fast_interp2d

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebl_codes import dust_absorption_models as dust_abs


# from hmf import MassFunction


class EBL_model(object):
    """
    Class that computes the EBL contribution coming from three sources:
    Single Stellar Populations (SSP), Intra-Halo Light (IHL)
    and axion decay.

    Units of returns
    -----------------
    EBL:
      -> Cubes: nW m**-2 sr**-1
      -> Splines: log10(nW m**-2 sr**-1)
    Emissivity spline: log10(erg s^-1 Hz^-1 Mpc^-3)
    """

    def logging_info(self, text):
        if self._log_prints:
            logging.info(
                '%.2fs: %s' % (time.process_time()
                               - self._process_time, text))
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
            Number of integration steps to compute with
            (uses Simpson's integration method)
        z_max: float
            Maximum redshift at which we form SSPs.
        """

        self._shifted_times_emiss = None
        self._s = None
        self._log_t_ssp_intcube = None
        self._process_time = time.process_time()
        logging.basicConfig(level='INFO',
                            format='%(asctime)s - %(levelname)s'
                                   ' - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self._log_prints = log_prints

        self._cube = None
        self._z_cube = None
        self._log_freq_cube = None
        self._steps_integration_cube = None

        self._ssp_log_time = None
        self._ssp_log_freq = None
        self._ssp_log_emis = None
        self._kernel_emiss = None

        self._emiss_ssp_cube = 0.
        self._emiss_ihl_cube = 0.
        self._emiss_axion_cube = 0.
        self._emiss_total_cube = 0.

        self._emiss_ssp_spline = None
        self._emiss_ihl_spline = None
        self._emiss_axion_spline = None
        self._emiss_total_spline = None

        self._ebl_ssp_cube = 0.
        self._ebl_ihl_cube = 0.
        self._ebl_axion_cube = 0.

        self._ebl_ssp_spline = None
        self._ebl_ihl_spline = None
        self._ebl_axion_spline = None
        self._ebl_total_spline = None

        self._h = h
        self._omegaM = omegaM
        self._omegaB0 = omegaBar
        self._cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                                    Ob0=omegaBar, Tcmb0=2.7255)

        self._z_array = z_array
        self._z_max = z_max
        self._lambda_array = lambda_array[::-1]
        self._freq_array = np.log10(c.value / lambda_array[::-1] * 1e6)
        self._t_intsteps = t_intsteps

        self._last_ssp = None
        self._ebl_intcube = None
        self._shifted_freq = None
        self._ebl_z_intcube = None

        self.intcubes()

        return

    @property
    def emiss_ssp_spline(self):
        return self._emiss_ssp_spline

    @property
    def emiss_ihl_spline(self):
        return self._emiss_ihl_spline

    @property
    def emiss_axion_spline(self):
        return self._emiss_axion_spline

    @property
    def emiss_total_spline(self):
        return self._emiss_total_spline

    @property
    def ebl_total_spline(self):
        return self._ebl_total_spline

    @property
    def ebl_ssp_spline(self):
        return self._ebl_ssp_spline

    @property
    def ebl_ihl_spline(self):
        return self._ebl_ihl_spline

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
        Recalculate EBL contribution from axion decay.
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

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
        Recalculate EBL contribution from Intra-Halo Light (IHL).
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be
        emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the
        galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter
        alpha, as (1 + z)**alpha.
        
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
        self._cosmo = FlatLambdaCDM(H0=new_H0, Om0=self._omegaM,
                                    Ob0=self._omegaB0, Tcmb0=2.7255)

        self._ebl_ssp_spline = None
        self._ebl_axion_spline = None
        self._ebl_ihl_spline = None

        return

    def read_SSP_file(self, data_file, ssp_type,
                      pop_filename='', cut_popstar=False):
        """
        Read Simple Stellar Population model spectra.

        Spectra units:
        --------------
        log10(erg s^-1 A^-1).

        Starburst99 output:
        http://www.stsci.edu/science/starburst99/
        The calculations assume that the value of 'Total stellar mass'
        in the simulation has been left as default, 1e6 Msun. If not,
        change the calculation of emissivity of '-6.' to
        log10(new total mass).

        Popstar output:
        on progress
        """
        if ssp_type == 'SB99':
            if cut_popstar is False:
                d = np.loadtxt(data_file, skiprows=6)

                # Get unique time steps and frequencies, and spectral data
                t_total = np.unique(d[:, 0])
                l_total = np.unique(d[:, 1])
                dd_total = d[:, 3].reshape(t_total.shape[0],
                                           l_total.shape[0]).T

            else:
                data_starburst_old = np.loadtxt(
                    'ssp/final_run_spectrum', skiprows=6)
                t_old = np.unique(data_starburst_old[:, 0])
                l_old = np.unique(data_starburst_old[:, 1])
                dd_old = data_starburst_old[:, 3].reshape(t_old.shape[0],
                                                          l_old.shape[0]).T

                data_starburst = np.loadtxt(
                    'ssp/low_res_for_real.spectrum1', skiprows=6)
                t = np.unique(data_starburst[:, 0])
                l_total = np.unique(data_starburst[:, 1])
                dd = data_starburst[:, 3].reshape(t.shape[0],
                                                  l_total.shape[0]).T

                dd_total = np.zeros((len(l_total),
                                     len(t) + sum(t_old > t[-1])))
                t_total = np.zeros(len(t) + sum(t_old > t[-1]))
                aaa = np.where((t_old - t[-1]) > 0)[0][0]

                t_total[:len(t)] = t
                t_total[len(t):] = t_old[aaa:]

                dd_total[:, :len(t)] = dd
                dd_total[:, len(t):] = dd_old[:, aaa:]

            # Define the quantities we will work with
            self._ssp_log_time = np.log10(t_total)  # log(time/yrs)
            self._ssp_log_freq = np.log10(  # log(frequency/Hz)
                c.value / l_total[::-1] / 1E-10)
            self._ssp_log_emis = (dd_total[::-1]  # log(em[erg/s/Hz/M_solar])
                                  - 6.
                                  + np.log10(1E10 * c.value)
                                  - 2. * self._ssp_log_freq[:, np.newaxis])

        elif ssp_type == 'Popstar09':
            list_files = os.listdir(data_file)

            numbers = []
            for listt in list_files:
                numbers.append(
                    float(listt.replace(pop_filename, '')))

            indexes = np.argsort(numbers)
            self._ssp_log_time = np.sort(numbers)
            ssp_wavelenghts = np.loadtxt(data_file + list_files[0])[:, 0]
            pop09_lumin_cube = np.zeros((len(ssp_wavelenghts),
                                         len(list_files)))

            self._ssp_log_freq = np.log10(  # log(frequency/Hz)
                c.value / ssp_wavelenghts[::-1] / 1E-10)

            x_is_1e4 = np.argmin(np.abs(ssp_wavelenghts - 1e4))
            cut = 5e27 / 3.828e33

            for nind, ind in enumerate(indexes):
                yyy = np.loadtxt(
                    data_file
                    + pop_filename
                    + str('%.2f' % numbers[ind])
                )[:, 1]

                if np.shape(np.where(yyy[:x_is_1e4] < cut))[1] == 0:
                    min_x = 0
                else:
                    min_x = np.where(yyy[:x_is_1e4] < cut)[0][-1]

                if cut_popstar:
                    pop09_lumin_cube[min_x:, nind] = yyy[min_x:]
                else:
                    pop09_lumin_cube[:, nind] = yyy

            # log(em[erg/s/Hz/M_solar])
            self._ssp_log_emis = np.log10(pop09_lumin_cube[::-1]
                                          * 3.828e33)
            self._ssp_log_emis[np.isnan(self._ssp_log_emis)] = -43.
            self._ssp_log_emis[
                np.invert(np.isfinite(self._ssp_log_emis))] = -43.
            self._ssp_log_emis += (np.log10(1E10 * c.value)
                                   - 2. * self._ssp_log_freq[:, np.newaxis])

            self._ssp_log_freq = (self._ssp_log_freq[1:]
                                  + self._ssp_log_freq[:-1]) / 2.
            self._ssp_log_emis = (self._ssp_log_emis[1:, :]
                                  + self._ssp_log_emis[:-1, :]) / 2.

            del pop09_lumin_cube

        elif ssp_type == 'pegase3':
            # pegase_metall = [0.1, 0.05, 0.02, 0.008,
            #                  0.004, 0.0004, 0.0001]
            pegase_metall = [pop_filename]  # [0.02]

            data_pegase = np.loadtxt(
                'ssp/pegase3/spectral_resultsZ0.02.txt')
            t_pegase = np.unique(data_pegase[:, 0])
            l_pegase = np.unique(data_pegase[:, 1])

            dd_pegase = np.zeros((l_pegase.shape[0],
                                  t_pegase.shape[0],
                                  len(pegase_metall)))

            for n_met, met in enumerate(pegase_metall):
                data_pegase = np.loadtxt(
                    'ssp/pegase3/spectral_resultsZ' + str(met) + '.txt')
                dd_pegase[:, :, n_met] = data_pegase[:, 2].reshape(
                    t_pegase.shape[0],
                    l_pegase.shape[0]).T[::-1]

            self._ssp_log_time = np.log10(t_pegase * 1e6)  # log(time/yrs)
            self._ssp_log_time[np.isnan(self._ssp_log_time)] = -43.
            self._ssp_log_time[
                np.invert(np.isfinite(self._ssp_log_time))] = -43.

            self._ssp_log_freq = np.log10(  # log(frequency/Hz)
                c.value / l_pegase[::-1] / 1E-10)

            self._ssp_log_emis = np.log10(dd_pegase)
            self._ssp_log_emis[np.isnan(self._ssp_log_emis)] = -43.
            self._ssp_log_emis[
                np.invert(np.isfinite(self._ssp_log_emis))] = -43.

            self._ssp_log_emis += (np.log10(1E10 * c.value)
                                   - 2. * self._ssp_log_freq
                                   [:, np.newaxis, np.newaxis])

        # import matplotlib.pyplot as plt
        # plt.figure(20, figsize=(10, 8))
        # plt.title('ssp: %s , %s' % (ssp_type, pop_filename))
        #
        # if ssp_type == 'Popstar09':
        #     label = 'dotted'
        # elif ssp_type == 'pegase3':
        #     label = '--'
        # else:
        #     label = 'solid'
        #
        # color = ['b', 'orange', 'k', 'r', 'green', 'grey', 'limegreen',
        #          'purple', 'brown']
        #
        # for i, age in enumerate([6.0, 6.5, 7.5, 8., 8.5, 9., 10.]):
        #     aaa = np.abs(self._ssp_log_time - age).argmin()
        #     plt.plot(c.value * 1e10 / 10 ** self._ssp_log_freq,
        #              self._ssp_log_emis[:, aaa],
        #              color=color[i], linestyle=label,
        #              label='log(t) = %.2f'
        #                    % self._ssp_log_time[aaa])
        #
        # plt.xscale('log')
        #
        # if ssp_type == 'SB99':
        #     plt.legend()
        #
        # plt.xlim(1e2, 1e6)
        # # plt.ylim(10, 22)
        #
        # plt.xlabel('wavelenght [A]')

        # Sanity check and log info
        self._ssp_log_emis[np.isnan(self._ssp_log_emis)] = -43.
        self._ssp_log_emis[
            np.invert(np.isfinite(self._ssp_log_emis))] = -43.
        self.logging_info('Reading of SSP file')
        return

    def intcubes(self):
        """
        Calculation of cubes that will be globally used.

        Their shapes are:
        (len(frequency array), len(z array), integration steps)
        """
        # Cubes to initialize the general quantities needed
        self._cube = np.ones(
            [self._lambda_array.shape[0], self._z_array.shape[0],
             self._t_intsteps])
        self._steps_integration_cube = (self._cube * np.linspace(
            0., 1., self._t_intsteps))
        self._log_freq_cube = (self._cube
                               * self._freq_array[:, np.newaxis, np.newaxis])
        self._z_cube = self._cube * self._z_array[np.newaxis, :, np.newaxis]

        self.logging_info('Initialize cubes: end')
        return

    def sfr_function(self, function_input, xx_array, params=None):
        """
        Stellar Formation Rate (SFR) is the density of stars that are born
        as a function of time/redshift.

        function_input: string or callable
            Formula (analytical or numerical) of the SFR.
        xx_array: 1D array
            Redshift values to input in the formula.
        params: 1D array or list
            Optional parameters that can enter the sfr formula.

        :return: 1D array
            SFR values with the same length of the xx-array.
        """
        if params is None:
            params = []

        if type(function_input) == str:
            return (lambda x: eval(function_input)(x, params))(xx_array)
        elif callable(function_input):
            return function_input(xx_array)
        else:
            print(
                'Unrecognized type of sfr '
                '(required string or UnivariateSpline')
            return 0.

    def emiss_ssp_calculation(self, yaml_data, sfr=None):
        """
        Calculation of SSP emissivity from the parameters given in
        the dictionary.

        Emissivity units:
        ----------
            -> Cubes: erg / s / Mpc**3 == 1e-7 W / Mpc**3
            -> Splines: log10(erg / s / Mpc**3 == 1e-7 W / Mpc**3)

        Parameters
        ----------
        yaml_data: dict
            Data necessary to reconstruct the EBL component from a SSP.
        sfr: string or callable (spline, function...)
            Formula of the sfr used to calculate the emissivity.
        """
        self.logging_info('SSP parameters: %s' % yaml_data['name'])

        if sfr is None:
            sfr_formula = yaml_data['sfr']
            sfr_params = yaml_data['sfr_params']
        else:
            sfr_formula = sfr
            sfr_params = None

        if (self._ssp_log_emis is None
                or (self._last_ssp != [
                    yaml_data['path_SSP'], yaml_data['ssp_type'],
                    yaml_data['file_name'], yaml_data['cut_popstar']])):
            self.read_SSP_file(yaml_data['path_SSP'],
                               yaml_data['ssp_type'],
                               pop_filename=yaml_data['file_name'],
                               cut_popstar=yaml_data['cut_popstar'])

            ssp_spline = RectBivariateSpline(x=self._ssp_log_freq,
                                             y=self._ssp_log_time,
                                             z=self._ssp_log_emis,
                                             kx=1, ky=1)

            lookback_time_cube = self._cube * self._cosmo.lookback_time(
                self._z_array).to(u.yr)[np.newaxis, :, np.newaxis]

            # Array of time values we integrate the emissivity over
            # (in log10)
            self._log_t_ssp_intcube = np.log10(
                (self._cosmo.lookback_time(self._z_max)
                 - lookback_time_cube).to(u.yr).value)
            self._log_t_ssp_intcube[self._log_t_ssp_intcube
                                    > self._ssp_log_time[-1]] = (
                self._ssp_log_time[-1])

            self._log_t_ssp_intcube = (
                    (self._log_t_ssp_intcube - self._ssp_log_time[0])
                    * self._steps_integration_cube
                    + self._ssp_log_time[0])

            self.logging_info('SSP emissivity: set time integration cube')

            # Initialise mask to limit integration range to SSP data (in
            # wavelength/frequency)
            self._s = ((self._log_freq_cube >= self._ssp_log_freq[0])
                       * (self._log_freq_cube <= self._ssp_log_freq[-1]))

            self.logging_info('SSP emissivity: set frequency mask')

            # Two interpolations, transforming t->z (using log10 for both of
            # them) and a bi spline with the SSP data
            t2z = UnivariateSpline(
                np.log10(self._cosmo.lookback_time(
                    self._z_array).to(u.yr).value),
                np.log10(self._z_array),
                s=0, k=1)

            self._shifted_times_emiss = 10. ** t2z(np.log10(
                lookback_time_cube[self._s].value
                + 10. ** self._log_t_ssp_intcube[self._s]))

            self.logging_info('SSP emissivity: set splines')

            # Interior of emissivity integral:
            # L{t(z)-t(z')} * dens(z') * |dt'/dz'|

            self._kernel_emiss = self._cube * 1E-43
            self._kernel_emiss = (10. ** self._log_t_ssp_intcube  # Variable
                                  # change,
                                  * np.log(10.)  # integration over y=log10(x)
                                  * 10. ** ssp_spline.ev(  # L(t)
                        self._log_freq_cube, self._log_t_ssp_intcube))
            self.logging_info('SSP emissivity: set the initial kernel')

        kernel_emiss = self._cube * 1e-43
        kernel_emiss[self._s] = (
                self._kernel_emiss[self._s]  # sfr(z(t))
                * (self.sfr_function(sfr_formula,
                                     self._shifted_times_emiss,
                                     sfr_params)))


        self.logging_info('SSP emissivity: calculate ssp kernel')

        # Calculate emissivity in units
        # [erg s^-1 Hz^-1 Mpc^-3] == [erg Mpc^-3]
        self._emiss_ssp_cube = simpson(kernel_emiss,
                                       x=self._log_t_ssp_intcube,
                                       axis=-1)

        self.logging_info('SSP emissivity: integrate emissivity')

        # Dust absorption (applied in log10)
        self._emiss_ssp_cube *= 10 ** dust_abs.calculate_dust(
            self._lambda_array, models=yaml_data['dust_abs_models'],
            z_array=self._z_array)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.title(yaml_data['name'])
        # yyy = 10 ** dust_abs.calculate_dust(
        #     self._lambda_array, models=yaml_data['dust_abs_models'],
        #     z_array=self._z_array)
        # print('Shape of dust abs', np.shape(yyy))
        #
        # alpha = 1.
        # plt.plot(self._lambda_array, yyy[:, 0],
        #          'k', alpha=alpha, label=r'z=%.2f' %
        #                                  self._z_array[0])
        # for i in [2, 4, 6]:
        #     alpha -= 0.15
        #     aaa = (np.abs(self._z_array - i)).argmin()
        #     plt.plot(self._lambda_array, yyy[:, aaa],
        #              'k', alpha=alpha, label='%.2f' % i)
        #
        # plt.ylabel('Escape fraction of photons')
        # plt.xlabel('lambda (microns)')
        # plt.legend()
        # plt.xscale('log')
        # plt.ylim(0., 1.2)
        # plt.xlim(0.05, 10)

        self.logging_info('SSP emissivity: set dust absorption')

        # Spline of the emissivity
        log10_emiss = np.log10(self._emiss_ssp_cube)
        log10_emiss[np.isnan(log10_emiss)] = -43.
        log10_emiss[np.invert(np.isfinite(log10_emiss))] = -43.
        # fast_interp2d
        self._emiss_ssp_spline = fast_interp2d(
            [self._freq_array[0], self._z_array[0]],
            [self._freq_array[-1], self._z_array[-1]],
            [self._freq_array[1] - self._freq_array[0],
             self._z_array[1] - self._z_array[0]],
            log10_emiss,
            k=1, p=[False, False], e=[0, 0])

        # Free memory and log the time
        del kernel_emiss, log10_emiss
        self.logging_info('SSP emissivity: end')
        self._last_ssp = [
            yaml_data['path_SSP'], yaml_data['ssp_type'],
            yaml_data['file_name'], yaml_data['cut_popstar']]
        return

    def ebl_ssp_calculation(self, yaml_data, sfr=None):
        """
        Calculate the EBL SSP contribution.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        yaml_data: dictionary
            Data necessary to reconstruct the EBL component from an SSP.
        sfr: string or callable (spline, function...)
            Formula of the sfr used to calculate the emissivity.
        """
        self.emiss_ssp_calculation(yaml_data, sfr=sfr)

        if (self._ebl_intcube is None
                or np.shape(self._ebl_intcube) != np.shape(self._cube)):
            self.ebl_ssp_cubes()

        # logging.info(
        #     '%.2fs: %s' % (time.process_time()
        #                    - self._process_time, yaml_data['sfr_params']))
        # self._process_time = time.process_time()

        self.logging_info('SSP EBL: calculation of z cube')

        # Calculate integration values
        ebl_intcube = self._ebl_intcube * 10. ** self._emiss_ssp_spline(
            self._shifted_freq,
            self._ebl_z_intcube)
        self.logging_info('SSP EBL: calculation of kernel interpolation')

        # Integration of EBL from SSP
        self._ebl_ssp_cube = simpson(ebl_intcube,
                                     x=self._ebl_z_intcube,
                                     axis=-1)

        self.logging_info('SSP EBL: integration')

        # Spline of the SSP EBL intensity
        log10_ebl = np.log10(self._ebl_ssp_cube)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_ssp_spline = RectBivariateSpline(x=self._freq_array,
                                                   y=self._z_array,
                                                   z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del ebl_intcube, log10_ebl
        self.logging_info('SSP EBL: done')
        return

    def ebl_ssp_cubes(self, freq_positions=None):
        # Cubes needed to calculate the EBL from an SSP
        if freq_positions is None:
            self._ebl_z_intcube = (self._z_cube
                                   + self._steps_integration_cube
                                   * (np.max(self._z_array)
                                      - self._z_cube))

            self._shifted_freq = (self._log_freq_cube + np.log10(
                (1. + self._ebl_z_intcube) / (1. + self._z_cube)))

            self._ebl_intcube = (10. ** self._log_freq_cube
                                 * c.value / 4. / np.pi)
            self._ebl_intcube = (self._ebl_intcube
                                 / ((1. + self._ebl_z_intcube)
                                    * self._cosmo.H(
                                self._ebl_z_intcube).to(
                                u.s ** -1).value))
        else:
            # Careful, the expressions are only prepared for z=0
            self._ebl_z_intcube = np.ones((len(freq_positions),
                                           self._t_intsteps))
            self._ebl_z_intcube *= (self._z_cube[0, 0, :]
                                    + self._steps_integration_cube[0, 0, :]
                                    * (np.max(self._z_array)
                                       - self._z_cube[0, 0, :]))

            self._shifted_freq = (freq_positions[:, np.newaxis]
                                  + np.log10((1. + self._ebl_z_intcube)
                                             )
                                  )

            self._ebl_intcube = (c.value * 10 ** freq_positions
                                 / 4. / np.pi)[:, np.newaxis]
            self._ebl_intcube = (self._ebl_intcube
                                 / ((1. + self._ebl_z_intcube)
                                    * self._cosmo.H(
                                self._ebl_z_intcube).to(
                                u.s ** -1).value))

        self.logging_info('Initialize cubes: calculation of division')

        # Mpc^-3 -> m^-3, erg/s -> nW
        self._ebl_intcube *= (u.erg * u.Mpc ** -3 * u.s ** -1
                              ).to(u.nW * u.m ** -3)

        self.logging_info('Initialize cubes: calculation of kernel')
        return

    def ebl_ssp_individualData(self, yaml_data, x_data, sfr=None):
        """
        Calculate the EBL SSP contribution. for the specific wavelength
        data that we have available (and redshift z=0). Useful to avoid
        big calculations with a wide grid on wavelengths and redshifts.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1

        Parameters
        ----------
        yaml_data: dictionary
            Data necessary to reconstruct the EBL component from an SSP.
        sfr: string or callable (spline, function...)
            Formula of the sfr used to calculate the emissivity.
        """
        self.logging_info('SSP EBL: enter program')
        self.emiss_ssp_calculation(yaml_data, sfr=sfr)

        self.logging_info('SSP EBL: emissivity calculated')

        new_individualFreq = np.log10(c.value / x_data[::-1] * 1e6)

        if (self._ebl_intcube is None
                or np.shape(self._ebl_intcube)[0]
                == np.shape(self._cube)[0]):
            self.ebl_ssp_cubes(freq_positions=new_individualFreq)

        self.logging_info('SSP EBL: calculation of z cube')

        # Calculate integration values
        ebl_intcube = (self._ebl_intcube
                       * 10. ** self._emiss_ssp_spline(
                    self._shifted_freq,
                    self._ebl_z_intcube))
        self.logging_info('SSP EBL: calculation of kernel interpolation')

        return simpson(ebl_intcube,
                       x=self._ebl_z_intcube,
                       axis=-1)[::-1]

    def emiss_intrahalo_calculation(self, log10_Aihl, alpha):
        """
        Emissivity contribution from Intra-Halo Light (IHL).
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be
        emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the
        galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter
        alpha, as (1 + z)**alpha.

        Emissivity units:
        ----------
            -> Cubes: erg / s / Mpc**3 == 1e-7 W / Mpc**3
            -> Splines: log10(erg / s / Mpc**3 == 1e-7 W / Mpc**3)

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

        # Use the spectrum of a synthetic old galaxy (taken from
        # the SWIRE library), and normalize it at a wavelength
        # of 2.2 microns.
        old_spectrum = np.loadtxt('Swire_library/Ell13_template_norm.sed')
        old_spline = UnivariateSpline(old_spectrum[:, 0],
                                      old_spectrum[:, 1],
                                      s=0, k=1, ext=1)

        # S_lambda = F_lambda * lambda
        old_spectrum[:, 1] *= (old_spectrum[:, 0]
                               / old_spline(22000)
                               / 22000.)
        old_spectrum[:, 0] *= 1e-4
        old_spectrum_spline = UnivariateSpline(np.log10(old_spectrum[:, 0]),
                                               np.log10(old_spectrum[:, 1]),
                                               s=0, k=1)

        # Initialize an object to calculate dn/dM
        mf = MassFunction(cosmo_model=self._cosmo, Mmin=m_min, Mmax=m_max)

        # Total luminosity of a galaxy at 2.2 microns
        L22 = 5.64e12 * (self._h / 0.7) ** (-2) * (
                mf.m / 2.7e14 * self._h / 0.7) ** 0.72 / 2.2e-6

        # The object to calculate dn/dM returns this quantity for
        # a specific redshift
        kernel_intrahalo = np.zeros(
            (len(self._freq_array), len(self._z_array)))

        for nzi, zi in enumerate(self._z_array):
            mf.update(z=zi)
            lambda_luminosity = (
                    (f_ihl(mf.m) * L22 * (1 + zi) ** alpha)[:, np.newaxis]
                    * 10 ** old_spectrum_spline(
                np.log10(self._lambda_array[np.newaxis, :]))
            )
            kernel = (mf.m[:, np.newaxis]  # Variable change,
                      * np.log(10.)  # integration over log10(M)
                      * lambda_luminosity
                      * mf.dndm[:, np.newaxis]
                      )
            kernel_intrahalo[:, nzi] = simpson(kernel,
                                               x=np.log10(mf.m),
                                               axis=0)

        self._emiss_ihl_cube = kernel_intrahalo * c / (
                10 ** self._freq_array[:, np.newaxis]) ** 2. * u.s ** 2
        self._emiss_ihl_cube *= (
                u.solLum.to(u.W) * u.W / (u.Mpc * self._h) ** 3
        ).to(u.erg / u.s / u.Mpc ** 3)

        # Spline of the luminosity
        log10_lumin = np.log10(self._emiss_ihl_cube.value)
        log10_lumin[np.isnan(log10_lumin)] = -43.
        log10_lumin[np.invert(np.isfinite(log10_lumin))] = -43.
        self._emiss_ihl_spline = RectBivariateSpline(x=self._freq_array,
                                                     y=self._z_array,
                                                     z=log10_lumin,
                                                     kx=1, ky=1)

        # Free memory and log the time
        del old_spectrum, old_spectrum_spline, old_spline, mf, L22
        del kernel_intrahalo, lambda_luminosity, kernel
        del log10_lumin
        self.logging_info('Calculation time for emissivity ihl')
        return

    def ebl_intrahalo_calculation(self, log10_Aihl, alpha):
        """
        EBL contribution from Intra-Halo Light (IHL).
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        We assume a fraction of the light emitted by galaxies will be
        emitted as IHL (this fraction is f_ihl).
        This fraction is multiplied by the total halo luminosity of the
        galaxy and its typical spectrum.
        There is also a redshift dependency, coded with the parameter
        alpha, as (1 + z)**alpha.

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
        if self._emiss_ihl_spline is None:
            self.emiss_intrahalo_calculation(log10_Aihl, alpha)

        # Integrate the EBL intensity kernel over values of z
        z_integr = (self._z_array[np.newaxis, :, np.newaxis]
                    + (self._z_max - self._z_array[np.newaxis, :, np.newaxis])
                    * self._steps_integration_cube)

        kernel_ebl_intra = (10 **
                            ((self._emiss_ihl_spline.ev(
                                (self._log_freq_cube
                                 + np.log10((1. + z_integr)
                                            / (1. + self._z_cube))
                                 ).flatten(),
                                z_integr.flatten())
                             ).reshape(self._cube.shape))
                            / ((1. + z_integr)
                               * self._cosmo.H(z_integr).to(u.s ** -1).value))

        self._ebl_ihl_cube = simpson(kernel_ebl_intra, x=z_integr, axis=-1)

        self._ebl_ihl_cube *= u.erg * u.s / u.Mpc ** 3
        self._ebl_ihl_cube *= (c ** 2
                               / (self._lambda_array[:, np.newaxis] * 1e-6
                                  * u.m * 4. * np.pi))

        self._ebl_ihl_cube = self._ebl_ihl_cube.to(u.nW / u.m ** 2).value

        # Spline of the IHL EBL intensity
        log10_ebl = np.log10(self._ebl_ihl_cube)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_ihl_spline = RectBivariateSpline(x=self._freq_array,
                                                   y=self._z_array,
                                                   z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del kernel_ebl_intra, z_integr, log10_ebl
        self.logging_info('Calculation time for ebl ihl')
        return

    def emiss_axion_calculation(self, axion_mass, axion_gamma):
        """
        Emissivity contribution from axion decay.
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

        Emissivity units:
        ----------
            -> Cubes: erg / s / Mpc**3 == 1e-7 W / Mpc**3
            -> Splines: log10(erg / s / Mpc**3 == 1e-7 W / Mpc**3)

        Parameters
        ----------
        axion_mass: float [eV]
            Value of (m_a * c**2) of the decaying axion.
        axion_gamma: float [s**-1]
            Decay rate of the axion.
        """
        axion_mass = axion_mass * u.eV
        axion_gamma = axion_gamma * u.s ** -1

        z_star = (axion_mass
                  / (2. * h_plank.to(u.eV * u.s)
                     * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1)
                  - 1.)

        self._emiss_axion_cube = (((c / (4. * np.pi * u.sr)
                                    * self._cosmo.Odm(0.)
                                    * self._cosmo.critical_density0
                                    * c ** 2. * axion_gamma / axion_mass
                                    * 10 ** self._log_freq_cube[:, :,
                                            0] * u.s ** -1
                                    * h_plank * (1 + self._z_cube[:, :, 0])
                                    / self._cosmo.H(z_star)
                                    ).to(u.nW * u.m ** -2 * u.sr ** -1)
                                   ).value
                                  * (z_star > self._z_cube[:, :, 0]))

        # Spline of the axion EBL intensity
        log10_ebl = np.log10(self._emiss_axion_cube)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._emiss_axion_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del z_star, log10_ebl
        self.logging_info('Calculation time for ebl axions')
        return

    def ebl_axion_calculation(self, axion_mass, axion_gamma):
        """
        EBL contribution from axion decay.
        Based on the formula and expressions given by:
        http://arxiv.org/abs/2208.13794

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

        z_star = (axion_mass
                  / (2. * h_plank.to(u.eV * u.s)
                     * 10 ** self._log_freq_cube[:, :, 0] * u.s ** -1)
                  - 1.)

        self._ebl_axion_cube = (((c / (4. * np.pi * u.sr)
                                  * self._cosmo.Odm(0.)
                                  * self._cosmo.critical_density0
                                  * c ** 2. * axion_gamma / axion_mass
                                  * 10 ** self._log_freq_cube[:, :,
                                          0] * u.s ** -1
                                  * h_plank * (1 + self._z_cube[:, :, 0])
                                  / self._cosmo.H(z_star)
                                  ).to(u.nW * u.m ** -2 * u.sr ** -1)
                                 ).value
                                * (z_star > self._z_cube[:, :, 0]))

        # Spline of the axion EBL intensity
        log10_ebl = np.log10(self._ebl_axion_cube)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_axion_spline = RectBivariateSpline(x=self._freq_array,
                                                     y=self._z_array,
                                                     z=log10_ebl,
                                                     kx=1, ky=1)

        # Free memory and log the time
        del z_star, log10_ebl
        self.logging_info('Calculation time for ebl axions')
        return

    def emiss_sum_contributions(self):
        """
        Sum of the contributions to the emissivity which have been
        previously calculated.
        If any of the components has not been calculated,
        its contribution will be 0.
        Components are Single Stellar Populations (SSP),
        Intra-Halo Light (IHL) and axion decay.

        Emissivity units:
        ----------
            -> Cubes: erg / s / Mpc**3 == 1e-7 W / Mpc**3
            -> Splines: log10(erg / s / Mpc**3 == 1e-7 W / Mpc**3)
        """
        # Spline of the total emissivity
        log10_ebl = np.log10(
            self._emiss_ssp_cube + self._ebl_axion_cube + self._emiss_ihl_cube)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._emiss_total_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del log10_ebl
        self.logging_info('Calculation time for emissivity total')
        return

    def ebl_sum_contributions(self):
        """
        Sum of the contributions to the EBL which have been
        previously calculated.
        If any of the components has not been calculated,
        its contribution will be 0.
        Components are Single Stellar Populations (SSP),
        Intra-Halo Light (IHL) and axion decay.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)
        """
        # Spline of the total EBL intensity
        log10_ebl = np.log10(
            self._ebl_ssp_cube + self._ebl_axion_cube + self._ebl_ihl_cube)
        log10_ebl[np.isnan(log10_ebl)] = -43.
        log10_ebl[np.invert(np.isfinite(log10_ebl))] = -43.
        self._ebl_total_spline = RectBivariateSpline(
            x=self._freq_array,
            y=self._z_array,
            z=log10_ebl, kx=1, ky=1)

        # Free memory and log the time
        del log10_ebl
        self.logging_info('Calculation time for ebl total')
        return

    def ebl_all_calculations(self, ssp_yaml=None,
                             log10_Aihl=-3.23, alpha=1.,
                             axion_mass=1., axion_gamma=5e-23):
        """
        Calculate the EBL total contribution from our three components:
        Single Stellar Populations (SSP), Intra-Halo Light (IHL)
        and axion decay.

        EBL units:
        ----------
            -> Cubes: nW m**-2 sr**-1
            -> Splines: log10(nW m**-2 sr**-1)

        Parameters
        ----------
        ssp_yaml: dictionary
            Data necessary to reconstruct the EBL component from a SSP.
            Default: model from Kneiske02.
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
                        'sfr': 'lambda ci, x : ci[0]*((x+1)/(ci[1]+1))'
                               '**(ci[2]*(x<=ci[1]) - ci[3]*(x>ci[1]))',
                        'sfr_params': [0.15, 1.1, 3.4, 0.0],
                        'ssp_type': 'SB99',
                        'path_SSP': 'ssp/final_run_spectrum',
                        'dust_abs_models': ['kneiske2002', 'aaaa']}

        self.ebl_ssp_calculation(ssp_yaml)
        self.ebl_intrahalo_calculation(log10_Aihl, alpha)
        self.ebl_axion_calculation(axion_mass, axion_gamma)

        self.ebl_sum_contributions()

        return

    def emiss_all_calculations(self, ssp_yaml=None,
                               log10_Aihl=-3.23, alpha=1.,
                               axion_mass=1., axion_gamma=5e-23):
        """
        Calculate the emissivity contribution from our three components:
        Single Stellar Populations (SSP), Intra-Halo Light (IHL)
        and axion decay.

        Emissivity units:
        ----------
            -> Cubes: erg / s / Mpc**3 == 1e-7 W / Mpc**3
            -> Splines: log10(erg / s / Mpc**3 == 1e-7 W / Mpc**3)

        Parameters
        ----------
        ssp_yaml: dictionary
            Data necessary to reconstruct the EBL component from a SSP.
            Default: model from Kneiske02.
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
                        'sfr': 'lambda ci, x : ci[0]*((x+1)/(ci[1]+1))'
                               '**(ci[2]*(x<=ci[1]) - ci[3]*(x>ci[1]))',
                        'sfr_params': [0.15, 1.1, 3.4, 0.0],
                        'ssp_type': 'SB99',
                        'path_SSP': 'ssp/final_run_spectrum',
                        'dust_abs_models': ['kneiske2002', 'aaaa']}

        self.emiss_ssp_calculation(ssp_yaml)
        self.emiss_intrahalo_calculation(log10_Aihl, alpha)
        self.emiss_axion_calculation(axion_mass, axion_gamma)

        self.ebl_sum_contributions()

        return
