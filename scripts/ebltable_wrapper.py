import numpy as np
from astropy import units as u
import astropy.constants as c
from ebltable.ebl_from_model import EBL
from astropy.cosmology import Planck15 as cosmo


def cosmic_axion_contr(lmu, zz, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = (c.c.value / lmu * 1e6 * u.s ** -1)

    z_star = axion_mass / (2. * c.h.to(u.eV * u.s) * freq) - 1.

    ebl_axion_cube = ((cosmo.Odm(0.) * cosmo.critical_density0
                       * c.c ** 3. / (64. * np.pi * u.sr)
                       * axion_gayy ** 2. * axion_mass ** 2.
                       * freq
                       / cosmo.H(z_star)
                       ).to(u.nW * u.m ** -2 * u.sr ** -1)
                      ).value

    if type(zz) == np.float64 or len(zz) == 1:
        ebl_axion_cube = ebl_axion_cube * (z_star > zz)
    else:
        ebl_axion_cube = (ebl_axion_cube[np.newaxis, :]
                          * (z_star > zz[:, np.newaxis]))

    return ebl_axion_cube


class EBL_with_axion(EBL):
    def __init__(self, funct_ebl=None,
                 z=None, lmu=None, nuInu=None,
                 kx=1, ky=1,
                 axion_mass=0., axion_gayy=0.,
                 **kwargs):
        self._axion_mass = axion_mass
        self._axion_gayy = axion_gayy

        if funct_ebl is None:
            super(EBL_with_axion, self).__init__(
                lmu=lmu, z=z, nuInu=nuInu, kx=kx, ky=ky, **kwargs)
        else:
            aaa = funct_ebl(**kwargs)
            super(EBL_with_axion, self).__init__(
                lmu=10 ** aaa.x, z=aaa.y, nuInu=10 ** aaa.Z,
                kx=kx, ky=ky, **kwargs)

    def ebl_array(self, z, lmu):
        result = self.evaluate(lmu, z)
        result += cosmic_axion_contr(lmu, z, self._axion_mass,
                                     self._axion_gayy)
        return result

    @staticmethod
    def readmodel(model, kx=1, ky=1,
                  axion_mass=0., axion_gayy=0., **kwargs):
        return EBL_with_axion(
            funct_ebl=EBL.readmodel, kx=1, ky=1,
            axion_mass=axion_mass, axion_gayy=axion_gayy,
            model=model, **kwargs)

    @staticmethod
    def readascii(file_name, kx=1, ky=1, model_name=None,
                  axion_mass=0., axion_gayy=0., **kwargs):
        return EBL_with_axion(
            funct_ebl=EBL.readascii, kx=1, ky=1,
            axion_mass=axion_mass, axion_gayy=axion_gayy,
            file_name=file_name, model_name=model_name,
            **kwargs)

    @staticmethod
    def readfits(file_name,
                 hdu_nuInu_vs_z='NUINU_VS_Z',
                 hdu_wavelength='WAVELENGTHS',
                 zcol='REDSHIFT',
                 eblcol='EBL_DENS',
                 lcol='WAVELENGTH',
                 kx=1, ky=1,
                 model_name=None,
                 axion_mass=0., axion_gayy=0.,
                 **kwargs):
        return EBL_with_axion(
            funct_ebl=EBL.readfits, kx=1, ky=1,
            axion_mass=axion_mass, axion_gayy=axion_gayy,
            file_name=file_name, model_name=model_name,
            **kwargs)
