# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.legend_handler import HandlerTuple

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

from ebl_codes.EBL_class import EBL_model
from data.cb_measurs.import_cb_measurs import import_cb_data, dictionary_datatype

from astropy import units as u
import astropy.constants as c

from ebltable.ebl_from_model import EBL

from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm
import matplotlib as mpl


all_size = 24
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=all_size)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

direct_franceschini_data = '/home/porrassa/Downloads/franceschini2017/'

table1 = np.loadtxt(
    direct_franceschini_data + 'table1.txt', dtype=float)
table2 = np.loadtxt(
    direct_franceschini_data + 'table2.txt', dtype=float)
table3 = np.loadtxt(
    direct_franceschini_data + 'table3.txt', dtype=float)

# []  ''  _  {}  p

zz_array1 = np.unique(table1[0, :])
table1 = table1[1:, :]
zz_array2 = np.unique(table2[0, :])
table2 = table2[1:, :]
zz_array3 = np.unique(table3[0, :])
table3 = table3[1:, :]

zz_array = np.concatenate((zz_array1, zz_array2, zz_array3))
table = np.concatenate((table1, table2, table3), axis=1)

energyarray = np.linspace(-3, 1.5, num=500)
dataarray = np.zeros((len(zz_array), len(energyarray)))

for ni, ii in enumerate(zz_array):
    xx = np.concatenate(
        ([table[0, 2*ni]*1.001], table[:, 2*ni], [table[-1, 2*ni]*1.001]))
    yy = np.concatenate(([-43], table[:, 2*ni+1], [-43]))
    aa = UnivariateSpline(xx, yy, k=1, s=0, ext=3)
    dataarray[ni, :] = aa(energyarray)

dataarray = c.c / 4. / np.pi * 10**energyarray * u.eV * 10**dataarray / u.cm**3
dataarray = dataarray.to(u.nW*u.m**-2) / (1 + zz_array[:, np.newaxis])**3
wavelength_array = (c.h*c.c/(10**(energyarray)*u.eV)).to(u.micron)

sort_array = np.argsort(wavelength_array)
dataarray = (dataarray.value)[:, sort_array]
wavelength_array = (wavelength_array.value)[sort_array]

aabigsline = RectBivariateSpline(
    x=zz_array, y=wavelength_array, z=dataarray, kx=1, ky=1, s=0)

ebl_franccc = EBL(z=zz_array, lmu=wavelength_array,
                nuInu=dataarray.T, model='franc')

waves_ebl = np.geomspace(5e-6, 1e4, num=int(1e6))

# We introduce the Finke22 and CUBA splines
ebl = {}
plt.figure(figsize=(16, 7))

models = ['kneiske', 'finke2022']
modls_labels = ['Kneiske et al. 2010', ]

for e in models:
    ebl = EBL.readmodel(model=e)
    plt.plot(waves_ebl, ebl.ebl_array(np.array([0.]), waves_ebl),
             ls='-', label=e, lw=3,
             path_effects=[pe.Stroke(linewidth=3, foreground='k'),
                           pe.Normal()])


models = ['dominguez']
for e in models:
    ebl = EBL.readmodel(model=e)
    plt.plot(waves_ebl, ebl.ebl_array(np.array([0.]), waves_ebl),
             ls='--', label=e, lw=3,
             path_effects=[pe.Stroke(linewidth=3, foreground='k'),
                           pe.Normal()])

aaa = waves_ebl < 350
plt.plot(waves_ebl[aaa], ebl_franccc.ebl_array(lmu=waves_ebl[aaa], z=0.),
             ls='--', label='frnaceschini', lw=3,
             path_effects=[pe.Stroke(linewidth=3, foreground='k'),
                           pe.Normal()])

models = ['gilmore']
for e in models:
    ebl = EBL.readmodel(model=e)
    plt.plot(waves_ebl, ebl.ebl_array(np.array([0.]), waves_ebl),
             ls=':', label=e, lw=3,
             path_effects=[pe.Stroke(linewidth=3, foreground='k'),
                           pe.Normal()])

plt.xscale('log')
plt.yscale('log')
plt.legend(ncol=3, loc=9, bbox_to_anchor=(0.5, 1.3))
plt.ylim(bottom=1, top=30)
plt.xlim(0.1, 1e3)

plt.ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')
plt.xlabel(r'Wavelength ($\mu$m)')

plt.savefig('outputs/figures_paper/cb_literature.pdf', bbox_inches='tight')
plt.savefig('outputs/figures_paper/cb_literature.png', bbox_inches='tight')

plt.show()