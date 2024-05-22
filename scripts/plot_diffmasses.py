# IMPORTS -----------------------------------#
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
import matplotlib.cm
from matplotlib import colormaps
from matplotlib.ticker import ScalarFormatter

from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline

from astropy import units as u
from astropy.constants import c
from astropy.constants import h as h_plank
from astropy.cosmology import FlatLambdaCDM

from ebltable.ebl_from_model import EBL
from data.cb_measurs.import_cb_measurs import import_cb_data

all_size = 34
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)


def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index = True
            else:
                use_index = False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index == "auto":
        if cmap.N > 100:
            use_index = False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index = False
        elif isinstance(cmap, ListedColormap):
            use_index = True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return plt.cycler("color", cmap(ind))
    else:
        colors = cmap(np.linspace(0, 1, N))
        return plt.cycler("color", colors)


# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

lorri_trans = np.loadtxt('data/lorri_qe_v2.txt')
lorri_trans[:, 0] = lorri_trans[:, 0] * 1e-3
lorri_trans[:, 1] = lorri_trans[:, 1]  # * 1e-2  # / max(lorri_trans[:, 1])
spline_lorri = UnivariateSpline(lorri_trans[:, 0], lorri_trans[:, 1],
                                s=0, k=1, ext=1)

waves_ebl = np.geomspace(0.1, 1.3, num=20000)

pivot_vw = (simpson(lorri_trans[:, 1] * lorri_trans[:, 0],
                    x=lorri_trans[:, 0])
            / simpson(lorri_trans[:, 1] / lorri_trans[:, 0],
                      x=lorri_trans[:, 0])
            ) ** 0.5
print('pivot wv: ', pivot_vw)

ebl = {}
for m in EBL.get_models():
    ebl[m] = EBL.readmodel(m)

nuInu = {}
for m, e in ebl.items():
    nuInu[m] = e.ebl_array(np.array([0.]), waves_ebl)
spline_finke = UnivariateSpline(waves_ebl, nuInu['finke2022'], s=0, k=1)
spline_cuba = UnivariateSpline(waves_ebl, nuInu['cuba'], s=0, k=1)

h = 0.7
omegaM = 0.3
omegaBar = 0.0222 / 0.7 ** 2.
cosmo = FlatLambdaCDM(H0=h * 100., Om0=omegaM,
                      Ob0=omegaBar, Tcmb0=2.7255)


def avg_flux(f_nu, trans_spline, lambda_array):
    yyy = trans_spline(lambda_array) / lambda_array
    yyy_num = simpson(y=yyy * f_nu(lambda_array), x=lambda_array)
    yyy_den = simpson(y=yyy, x=lambda_array)
    return yyy_num / yyy_den


def avg_lmbd_v1(f_nu, trans_spline, lambda_array):
    yyy = trans_spline(lambda_array) * f_nu(lambda_array)
    yyy_num = simpson(y=yyy, x=lambda_array)
    yyy_den = simpson(y=yyy / lambda_array, x=lambda_array)
    return yyy_num / yyy_den


def cosmic_axion_contr(lmbd, mass, gayy):
    axion_mass = mass * u.eV
    axion_gayy = gayy * u.GeV ** -1

    freq = c.value / lmbd * 1e6 * u.s ** -1

    z_star = (axion_mass / (2. * h_plank.to(u.eV * u.s) * freq)
              - 1.)

    ebl_axion_cube = (
            ((cosmo.Odm(0.) * cosmo.critical_density0
              * c ** 3. / (64. * np.pi * u.sr)
              * axion_gayy ** 2. * axion_mass ** 2.
              * freq
              / cosmo.H(z_star)
              ).to(u.nW * u.m ** -2 * u.sr ** -1)
             ).value
            * (z_star > 0.))
    return ebl_axion_cube


D_factor = 2.20656e22 * u.GeV * u.cm ** -2


def host_axion_contr(xx, mass, gay, v_dispersion=220.):
    sigma = (2. * 2.48 / mass
             * (v_dispersion * u.km * u.s ** -1 / c).to(1))
    nuInu_values = (
            14.53 * mass ** 3. * (gay * 1e10) ** 2.
            * (D_factor / (1.11e22 * u.GeV * u.cm ** -2)).to(1)
            * (v_dispersion / 220.) ** -1
            * np.exp(-0.5 * ((xx - 2.48 / mass) / sigma) ** 2.))

    return nuInu_values


mean_lmbd_cuba = avg_lmbd_v1(
    f_nu=UnivariateSpline(
        waves_ebl,
        nuInu['cuba'] * pivot_vw * 1e-6 / c.value,
        s=0, k=1),
    trans_spline=spline_cuba,
    lambda_array=waves_ebl)

print('pivot only cuba: ', mean_lmbd_cuba)
print('avg flux of cuba: ',
      avg_flux(
          f_nu=UnivariateSpline(
              waves_ebl,
              nuInu['cuba'] * pivot_vw * 1e-6 / c.value,
              s=0, k=1),
          trans_spline=spline_lorri,
          lambda_array=waves_ebl)
      * c.value / mean_lmbd_cuba * 1e6)

axion_mass_array = [3., 4., 5., 6., 7., 8.]
axion_gayy_array = np.geomspace(4e-11, 1e-9, num=1)
print(axion_gayy_array)
colors_list = ['darkviolet', 'deeppink', 'orange']

cividis_mine = colormaps['cividis']._resample(len(axion_mass_array))
cividis_mine.colors[-1, 1] = 0.7
cividis_mine.colors[-1, 2] = 0.
cividis_mine.colors = cividis_mine.colors[::-1]
print(cividis_mine.colors)

cool_map = [to_rgba('indigo'),
            to_rgba('darkviolet'),
            to_rgba('deeppink'),
            to_rgba('darkorange'),
            # to_rgba('FFC400')
            '#FFC400'
            ]
print(cool_map)
cool_long = LinearSegmentedColormap.from_list('mymap', cool_map)
print(cool_map)
N = len(axion_mass_array)
plt.rcParams["axes.prop_cycle"] = get_cycle(cool_long, N)

# ----------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(12, 8))
ax4 = ax3.twinx()


def tick_function_2(X):
    return 2.48 / X


axtop = ax3.secondary_xaxis('top',
                            functions=(tick_function_2, tick_function_2))
axtop.tick_params(axis='x', direction='in', pad=0)
axtop.set_xlabel(r'$m_a c^2$ (eV)', labelpad=14)

ax4.plot(lorri_trans[:, 0], lorri_trans[:, 1],
         lw=2, c='grey', zorder=-1)
ax3.plot(waves_ebl, spline_cuba(waves_ebl), lw=2, c='k', ls='dotted',
         zorder=-1)

for na, aa in enumerate(axion_mass_array):
    for nb, bb in enumerate(axion_gayy_array):
        total_yy = (host_axion_contr(waves_ebl, aa, bb,
                                     v_dispersion=220)
                    + cosmic_axion_contr(waves_ebl, aa, bb)
                    + spline_cuba(waves_ebl)
                    )
        intensity_points = total_yy * waves_ebl * 1e-6 / c.value
        total_spline = UnivariateSpline(
            waves_ebl, intensity_points, s=0, k=1)
        mean_lambda = avg_lmbd_v1(f_nu=total_spline,
                                  trans_spline=spline_lorri,
                                  lambda_array=waves_ebl)
        mean_flux = avg_flux(f_nu=total_spline,
                             trans_spline=spline_lorri,
                             lambda_array=waves_ebl
                             ) * c.value / mean_lambda * 1e6

        if na % 1 == 0 and nb % 1 == 0:
            next_c = next(ax3._get_lines.prop_cycler)['color']
            # next_c = colors[na]
            ax3.axvline(mean_lambda, c=next_c, ls='--', alpha=0.7, lw=2)
            ax3.plot(waves_ebl, total_yy, c=next_c,
                     label=aa, lw=2)
            ax3.errorbar(x=mean_lambda, y=mean_flux,
                         linestyle='', color=next_c,
                         marker='d', alpha=0.8,
                         mfc='white',
                         markersize=20, markeredgewidth=2,
                         zorder=1e5
                         )
            ax3.errorbar(x=mean_lambda, y=mean_flux,
                         linestyle='', color='k',
                         marker='.',
                         mfc='k',
                         markersize=8, zorder=5e5
                         )
        total_yy = (host_axion_contr(waves_ebl, aa, bb,
                                     v_dispersion=50)
                    + cosmic_axion_contr(waves_ebl, aa, bb)
                    + spline_cuba(waves_ebl)
                    )
        # intensity_points = total_yy * waves_ebl * 1e-6 / c.value
        # total_spline = UnivariateSpline(
        #     waves_ebl, intensity_points, s=0, k=1)
        # mean_lambda = avg_lmbd_v1(f_nu=total_spline,
        #                           trans_spline=spline_lorri,
        #                           lambda_array=waves_ebl)
        # mean_flux = avg_flux(f_nu=total_spline,
        #                      trans_spline=spline_lorri,
        #                      lambda_array=waves_ebl
        #                      ) * c.value / mean_lambda * 1e6
        #
        # if na % 1 == 0 and nb % 1 == 0:
        #     next_c = next(ax3._get_lines.prop_cycler)['color']
        #     # next_c = colors[na]
        #     ax3.axvline(mean_lambda, c=next_c, ls='--', alpha=0.7, lw=2)
        #     ax3.plot(waves_ebl, total_yy, c=next_c,
        #              label=aa, lw=2, ls='--')
        #     ax3.errorbar(x=mean_lambda, y=mean_flux,
        #                  linestyle='', color=next_c,
        #                  marker='*', alpha=0.8,
        #                  mfc='white',
        #                  markersize=20, markeredgewidth=2,
        #                  zorder=1e5
        #                  )
        #     ax3.errorbar(x=mean_lambda, y=mean_flux,
        #                  linestyle='', color='k',
        #                  marker='.',
        #                  mfc='k',
        #                  markersize=8, zorder=5e5
        #                  )

plt.figure(fig3)
# lg1 = ax3.legend([plt.Line2D([], [], linestyle='-', lw=2,
#                              color=colors[i])
#                   for i in range(3)],
#                  ['%.0f eV' % axion_mass_array[i] for i in range(3)],
#                  fontsize=20, title='Axion mass',
#                  title_fontsize=22, loc=2)
# lg1 = ax3.legend(fontsize=20, title='Axion mass',
#                  title_fontsize=22, loc=2)
lg2 = plt.legend([plt.Line2D([], [], linestyle='-', lw=2,
                             color='r'),
                  plt.Line2D([], [], linestyle='--', lw=2,
                             color='r'),
                  (plt.Line2D([], [], linestyle='',
                              color='r', markerfacecolor='w',
                              marker='d', markersize=18,
                              markeredgewidth=2),
                   plt.Line2D([], [], linestyle='',
                              color='k', markerfacecolor='k',
                              marker='.', markersize=8)
                   )
                  ],
                 ['Spectrum', 'Mean wavelength', 'Integrated intensity'],
                 handler_map={tuple: HandlerTuple(ndivide=1)},
                 fontsize=18, loc=1)

ax3.errorbar(x=0.608, y=21.98,
             linestyle='', color='limegreen',
             marker='*',
             mfc='white',
             markersize=30, markeredgewidth=3,
             zorder=1e5
             )
ax3.errorbar(x=0.608, y=21.98,
             yerr=[1.23],
             linestyle='', color='k',
             marker='.',
             mfc='k',
             markersize=8, zorder=5e5
             )
ax3.set_yscale('log')
ax4.set_yscale('linear')
plt.xlim(0.25, 1.)
ax4.set_ylim(0., 0.7)
ax3.set_ylim(2., 3000)
ax3.axvline(0.608, c='k', alpha=0.9, lw=2)
ax3.set_xticks(ticks=[0.3, 0.4, 0.6, 0.8, 1.],
               labels=['0.3', '0.4', '0.6', '0.8', '1.'])
axtop.set_xticks(ticks=axion_mass_array,
                 labels=['%1.f' % i for i in axion_mass_array])
ax4.tick_params(axis='y', labelsize=26)
ax3.tick_params(axis='x', pad=10)

ax4.set_ylabel('Quantum efficiency of LORRI', fontsize=26, labelpad=10)
ax3.set_xlabel(r'Wavelength ($\mu$m)', fontsize=34)
ax3.set_ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')
ax3.annotate(r'$g_{a\gamma}$ = $4 \times 10^{-11}$ GeV$^{-1}$', xy=(0.88, 430),
             c='k',
             fontsize=18, horizontalalignment='center')

ax3.set_zorder(1)
ax3.patch.set_visible(False)
plt.savefig('outputs/figures_paper/axiondecay_v2.pdf',
            bbox_inches='tight')
plt.savefig('outputs/figures_paper/axiondecay_v2.png',
            bbox_inches='tight')
# plt.show()
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 8))
plt.tight_layout()

def tick_function_2(X):
    return 2.48/X
axion_mass_array2 = np.geomspace(2, 10., num=700)

mean_lambda_host = np.zeros(len(axion_mass_array2))
mean_lambda_cosmic = np.zeros(len(axion_mass_array2))
mean_lambda_total = np.zeros(len(axion_mass_array2))
mean_lambda_only = np.zeros(len(axion_mass_array2))

for i in range(len(mean_lambda_cosmic)):
    total_yy = (cosmic_axion_contr(waves_ebl,
                                   axion_mass_array2[i], 1e-10)
                + spline_cuba(waves_ebl)
                ) * waves_ebl * 1e-6 / c.value

    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_cosmic[i] = avg_lmbd_v1(f_nu=total_spline,
                                        trans_spline=spline_lorri,
                                        lambda_array=waves_ebl)
    # -----------------
    total_yy = (host_axion_contr(waves_ebl, axion_mass_array2[i], 1e-10,
                                 v_dispersion=220)
                + spline_cuba(waves_ebl)
                ) * waves_ebl * 1e-6 / c.value

    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_host[i] = avg_lmbd_v1(f_nu=total_spline,
                                      trans_spline=spline_lorri,
                                      lambda_array=waves_ebl)
    # -----------------
    total_yy = (host_axion_contr(waves_ebl, axion_mass_array2[i], 1e-10,
                                 v_dispersion=220)
                + cosmic_axion_contr(waves_ebl,
                                     axion_mass_array2[i], 1e-10)
                + spline_cuba(waves_ebl)
                ) * waves_ebl * 1e-6 / c.value

    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_total[i] = avg_lmbd_v1(f_nu=total_spline,
                                       trans_spline=spline_lorri,
                                       lambda_array=waves_ebl)
    # -----------------

    total_yy = (cosmic_axion_contr(waves_ebl,
                                   axion_mass_array2[i], 1e-10)
                ) * waves_ebl * 1e-6 / c.value
    total_spline = UnivariateSpline(
        waves_ebl, total_yy, s=0, k=1)
    mean_lambda_only[i] = avg_lmbd_v1(f_nu=total_spline,
                                      trans_spline=spline_lorri,
                                      lambda_array=waves_ebl)

ax2.plot(axion_mass_array2, mean_lambda_cosmic,  # marker='.',
         label='CUBA + cosmic', lw=2.5, c='tab:blue')
ax2.plot(axion_mass_array2, mean_lambda_host,  # marker='.',
         label='CUBA + host', lw=2.5, c='tab:orange')
ax2.plot(axion_mass_array2, mean_lambda_total,  # marker='.',
         label='CUBA + cosmic + host', lw=2.5, c='tab:green')
plt.rcParams['axes.formatter.min_exponent'] = 2
axtop2 = ax2.secondary_xaxis('top',
                             functions=(tick_function_2, tick_function_2))
axtop2.tick_params(axis='x', direction='in', pad=0)

ax2.set_xscale('log')
ax2.set_xticks(ticks=np.arange(0., 12.),
               labels=['%1.f' % i for i in np.arange(0., 12.)])
ax2.set_xlabel(r'$m_a c^2$ (eV)')  # , labelpad=14)

ax2.legend(fontsize=22, loc=3)

ax2.set_ylim(0.47, 0.73)
ax2.set_xlim(2, 9)

plt.axhline(0.608, ls='--', c='k', alpha=0.8, lw=1.5)
plt.annotate('pivot\nwavelength', xy=(3., 0.582), c='k',
             fontsize=22, horizontalalignment='center')

plt.annotate(r'$g_{a\gamma}$ = $10^{-10}$ GeV$^{-1}$', xy=(6.7, 0.70),
             c='k',
             fontsize=22, horizontalalignment='center')

axtop2.set_xlabel(r'Wavelength ($\mu$m)', fontsize=34, labelpad=14)

ax2.set_ylabel(r'$\langle\lambda\rangle$ ($\mu$m)', labelpad=12)

fig2.savefig('outputs/figures_paper/meanlambda_mass_v3.pdf',
             bbox_inches='tight')
fig2.savefig('outputs/figures_paper/meanlambda_mass_v3.png',
             bbox_inches='tight')
plt.show()
