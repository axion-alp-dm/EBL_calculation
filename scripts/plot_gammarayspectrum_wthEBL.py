# IMPORTS --------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import c
from astropy.table import QTable


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 20
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=7)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=5)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

hess_dom = QTable.read('/home/porrassa/Downloads/'
                       'PG1553+113_2019_flux_points_hess.ecsv')
hess_dom['e_ref'] = hess_dom['e_ref'].to(u.TeV)
hess_dom['e_min'] = hess_dom['e_min'].to(u.TeV)
hess_dom['e_max'] = hess_dom['e_max'].to(u.TeV)
hess_dom['e2dnde'] = hess_dom['e2dnde'].to(u.Unit('TeV / (cm2 s)'))
hess_dom['e2dnde_err'] = hess_dom['e2dnde_err'].to(u.Unit('TeV / (cm2 s)'))
hess_dom['e2dnde_ul'] = hess_dom['e2dnde_ul'].to(u.Unit('TeV / (cm2 s)'))


fermi_dom = QTable.read('/home/porrassa/Downloads/'
                       'PG1553+113_2019_flux_points_fermi.ecsv')
fermi_dom['e_ref'] = fermi_dom['e_ref'].to(u.TeV)
fermi_dom['e_min'] = fermi_dom['e_min'].to(u.TeV)
fermi_dom['e_max'] = fermi_dom['e_max'].to(u.TeV)
fermi_dom['e2dnde'] = fermi_dom['e2dnde'].to(u.Unit('TeV / (cm2 s)'))
fermi_dom['e2dnde_err'] = fermi_dom['e2dnde_err'].to(u.Unit('TeV / (cm2 s)'))

plt.figure(figsize=(6, 4))

plt.errorbar(hess_dom['e_ref'], hess_dom['e2dnde'],
             xerr=(hess_dom['e_ref'] - hess_dom['e_min'],
                   hess_dom['e_max'] - hess_dom['e_ref']),
             yerr=hess_dom['e2dnde_err'],
             ls='', marker='o', c='r')

plt.errorbar(fermi_dom['e_ref'], fermi_dom['e2dnde'],
             xerr=(fermi_dom['e_ref'] - fermi_dom['e_min'],
                   fermi_dom['e_max'] - fermi_dom['e_ref'],),
             yerr=fermi_dom['e2dnde_err'],
             ls='', marker='+', c='b'
         )

plt.xscale('log')
plt.yscale('log')

plt.ylim(1e-16, 4e-10)
plt.xlim(8e-4, 20)

plt.show()