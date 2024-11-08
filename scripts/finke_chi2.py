# IMPORTS --------------------------------------------#
import os
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline

from ebl_codes.EBL_class import EBL_model

from ebltable.ebl_from_model import EBL

from data.emissivity_measurs.emissivity_read_data import emissivity_data
from data.cb_measurs.import_cb_measurs import import_cb_data
from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.constants import k_B

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
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


def chi2_measurs(x_model, x_obs, err_obs):
    return sum(((x_obs - x_model) / err_obs) ** 2.)


# We introduce the Finke22 and CUBA splines
waves_ebl = np.logspace(-1, 6, num=500)
ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)
nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)


# data_fit = np.loadtxt('outputs/outputs_dust_reem_wto_LOWdatapoints '
#               '2024-10-30 10:04:08/SB99_dustFinke3e10.txt')

data_fit_chary = np.loadtxt('outputs/outputs_dust_final1/'
                            'SB99_dustFinke3e10.txt')

wv = data_fit_chary[1:, 0]
ebl_fit = data_fit_chary[1:, 1]
spline_fit_chary = UnivariateSpline(wv, ebl_fit, s=0, k=1)

data_fit_bosa = np.loadtxt('outputs/outputs_dust_final1/'
                           'SB99_dustFinke_bosa.txt')

wv = data_fit_bosa[1:, 0]
ebl_fit = data_fit_bosa[1:, 1]
spline_fit_bosa = UnivariateSpline(wv, ebl_fit, s=0, k=1)

plt.figure()
plt.loglog(wv, ebl_fit)
# plt.show()


fig, (ax1, ax_below) = plt.subplots(2, 1, figsize=(10, 20))
plt.subplot(211)
upper_lims, lower_lims = import_cb_data(plot_measurs=False, ax1=ax1,
                              lambda_max_total=1e20)

lower_lims = lower_lims[lower_lims['ref'] != 'ISO/ISOCAM (Clements+ ‘99)']
lower_lims = lower_lims[lower_lims['ref'] != 'SCUBA-2 (Hsu+ ‘16)']
lower_lims = lower_lims[lower_lims['ref'] != 'ALMA (Fujimoto+ ‘16)']

markers = ['>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']
names_all_lower, index = np.unique(lower_lims['ref'], return_index=True)
names_all_lower = names_all_lower[np.argsort(index)]
i = 0
for ni, name in enumerate(names_all_lower):
    data = lower_lims[lower_lims['ref'] == name]
    color_i = next(ax1._get_lines.prop_cycler)['color']

    ax1.errorbar(x=data['lambda'], y=data['nuInu'],
                 yerr=[data['nuInu_errn'], data['nuInu_errp']],
                 linestyle='', color=color_i,
                 label=name,
                 marker=markers[i % len(markers)]
                 )
    i += 1

plt.plot(waves_ebl, spline_finke(waves_ebl),
            c='orange', label='Finke')
plt.plot(waves_ebl, spline_fit_chary(waves_ebl), c='b', label='Our fit chary')
plt.plot(waves_ebl, spline_fit_bosa(waves_ebl), c='r', label='Our fit bosa')

xx_nu = (c / waves_ebl * 1e6 / u.m).to(u.s**-1)
yyy = (2 * h_plank * xx_nu**4. / c**2.
       / (np.exp(h_plank * xx_nu / k_B / 2.725 / u.K) - 1.))
yyy = yyy.to(u.nW/u.m**2)
# yyy = yyy.to(u.MJy)
# plt.plot(waves_ebl, yyy, c='k')

# legend11 = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left",
#                       title=r'Measurements', ncol=1,
#                       fontsize=12)
#
# ax1.add_artist(legend11)

aaa = chi2_measurs(spline_finke(lower_lims['lambda']),
                   lower_lims['nuInu'],
                   lower_lims['1 sigma'])
bbb = chi2_measurs(spline_fit_chary(lower_lims['lambda']),
                   lower_lims['nuInu'],
                   lower_lims['1 sigma'])
ccc = chi2_measurs(spline_fit_bosa(lower_lims['lambda']),
                   lower_lims['nuInu'],
                   lower_lims['1 sigma'])
print(aaa, bbb, ccc)
#
# plt.text(x=100, y=30, s=r'$\chi^2 = $ %.2f''\n'r'$\chi^2/dof$ aka %i = %.2f'
#                         % (aaa, len(lower_lims), aaa/len(lower_lims)))

# plt.text(x=1e4, y=70, s='CMB spectrum')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'$\nu I_{\nu}$ (nW / m$^2$ sr)')
plt.xlim([.1, 1e6])
plt.ylim(9e-1, 20)

def tick_function(X):
    return (c / X / u.micron).to(u.s**-1).value
def tick_function_2(X):
    return (c / X / u.s**-1).to(u.micron).value

ax3 = ax1.secondary_xaxis('top',
                         functions=(tick_function, tick_function_2))
ax3.tick_params(axis='x', direction='in', pad=0)
ax3.set_xlabel('Photon frequency (Hz)', labelpad=12)


plt.subplot(212, sharex=ax1)
plt.scatter(lower_lims['lambda'],
         (spline_finke(lower_lims['lambda'])-lower_lims['nuInu'])
         / lower_lims['1 sigma'], c='orange',
            label=r'Finke $\chi^2 = $%.2f'
                  % sum(((spline_finke(lower_lims['lambda'])-lower_lims['nuInu'])
         / lower_lims['1 sigma'])**2.))

plt.scatter(lower_lims['lambda'],
         (spline_fit_chary(lower_lims['lambda'])-lower_lims['nuInu'])
         / lower_lims['1 sigma'], c='b',
            label=r'Our fit chary $\chi^2 = $%.2f'
                  % sum(((spline_fit_chary(lower_lims['lambda'])-lower_lims['nuInu'])
         / lower_lims['1 sigma'])**2.))
plt.scatter(lower_lims['lambda'],
         (spline_fit_bosa(lower_lims['lambda'])-lower_lims['nuInu'])
         / lower_lims['1 sigma'], c='r',
            label=r'Our fit bosa $\chi^2 = $%.2f'
                  % sum(((spline_fit_bosa(lower_lims['lambda'])-lower_lims[
                'nuInu'])
         / lower_lims['1 sigma'])**2.))
plt.legend()
plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(
    r'$\frac{\mathrm{model}(\lambda_\mathrm{i})- \nu I_{\nu,\mathrm{i}}}'
           r'{\sigma_\mathrm{i}}$')

plt.axhline(0, c='grey', zorder=0)

plt.xlim(0.09, 1e3)

# Save the figures
fig.savefig('outputs' + '/ebl_bare' + '.png',
                bbox_inches='tight')
fig.savefig('outputs' + '/ebl_bare' + '.pdf',
                bbox_inches='tight')

plt.show()
