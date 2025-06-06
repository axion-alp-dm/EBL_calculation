import numpy as np
import matplotlib.pyplot as plt

from ebltable.ebl_from_model import EBL

from astropy import units as u
import astropy.constants as cc

fig, (ax_ebl, ax_opac, ax_ratio) = plt.subplots(1, 3, figsize=(16, 5))

my_ebl = ['3_grey_bodies.txt',
          'Chary.txt',
          'BOSA.txt']

zz = 0.034

e_array = np.geomspace(
    0.5,
    20.,
    num=500)
wv_array = np.geomspace(0.1, 1000, num=500)
opacities = {}

for nd, d in enumerate(my_ebl):
    print(d)

    ebl_finke = EBL.readascii(
        '../outputs/lhaaso/' + d, model_name='mine')

    ax_ebl.plot(wv_array, ebl_finke.ebl_array(z=zz, lmu=wv_array),
                   label=d)

    opacities[d] = ebl_finke.optical_depth(z0=zz, ETeV=e_array)

    ax_opac.plot(e_array, opacities[d], ls='-',
                   label=d)

plt.subplot(131)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('wavelength [mu]')
plt.ylabel('ebl intensity')

plt.legend()

plt.subplot(132)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Energy [TeV]')
plt.ylabel('opacity')
plt.title('z = ' + str(zz))

plt.legend()

plt.subplot(133)

plt.plot(
    e_array,
    np.exp(opacities['Chary.txt'] - opacities['BOSA.txt']),
label='bosa / Chary')

plt.plot(
    e_array,
    np.exp(opacities['Chary.txt'] - opacities['3_grey_bodies.txt']),
label='3_grey_bodies / Chary')

plt.xscale('log')
plt.yscale('linear')

plt.ylabel('Ratios of exp(-opacity)')
plt.xlabel('Energy [TeV]')

plt.legend()


plt.show()