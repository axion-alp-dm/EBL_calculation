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