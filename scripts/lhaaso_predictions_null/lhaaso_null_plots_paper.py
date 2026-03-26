import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from scipy.integrate import simpson
import astropy.units as u
from scipy.interpolate import UnivariateSpline

from iminuit import Minuit
from iminuit.cost import LeastSquares

from ebltable.ebl_from_model import EBL

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
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=10)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=10)
plt.rc('xtick.minor', size=5, width=1.)
plt.rc('ytick.minor', size=5, width=1.)
os.chdir("..")
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


# bbb = '_vs_bosa'
# vert_lines_against = 'bosa.txt'
# my_ebl = ['bosa.txt', '2bb.txt', 'chary.txt']
#
bbb = '_vs_chary'
vert_lines_against = 'chary.txt'
my_ebl = ['chary.txt', 'bosa.txt', '2bb.txt']
#
# bbb = '_vs_2bb'
# vert_lines_against = '2bb.txt'
# my_ebl = ['2bb.txt', 'bosa.txt', 'chary.txt']


hatches = ['/', '', '']
colors = {'bosa.txt': 'tab:blue',
          '2bb.txt': 'tab:green',
          'chary.txt': 'darkorange'}

labelss = {'bosa.txt': 'BOSA',
          '2bb.txt': '2BB',
          'chary.txt': 'Chary'}

xx_array = np.geomspace(0.5, 25.)
'''
# -------------HEGRA spectrum fits -------------------------------------
# ----------------------------------------------------------------------
mkr501_flux_1 = np.loadtxt(
    'data/lhasso_characteristics/mkr501_flare1997_paper1.txt')
mkr501_flux_1[:, 1] = (mkr501_flux_1[:, 1]
                       * mkr501_flux_1[:, 0] ** 2. * (u.TeV.to(u.erg))
                       * 1e12)
mkr501_flux_1[:, 2] = (mkr501_flux_1[:, 2]
                       * mkr501_flux_1[:, 0] ** 2. * (u.TeV.to(u.erg))
                       * 1e12)
mkr501_flux = np.loadtxt(
    'data/lhasso_characteristics/mkr501_flare1997_table_reanalysis.txt')

mkr501_flux = np.concatenate((mkr501_flux_1[:8, :3], mkr501_flux))

zz = 0.034

e_array = np.geomspace(
    np.min(mkr501_flux[:, 0]) * 0.5,
    np.max(mkr501_flux[:, 0]) * 1.5,
    num=500)


opacities_array = {}
for d in my_ebl:
    ebl_finke = EBL.readascii(
            'outputs/lhaaso_syst10percent/' + d, model_name='mine')
    opacityy = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
    opacities_array[d] = UnivariateSpline(
            np.log10(e_array), opacityy, k=1, s=0)

# -------------HEGRA spectrum fits 2 subfigs ---------------------------
# ----------------------------------------------------------------------

fig_spectrum2, ax_spectrum2 = plt.subplots(1, 2, figsize=(12, 4.5))
fig_spectrum2.subplots_adjust(hspace=0., wspace=0.)
plt.subplot(121)
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2], ls='', marker='.',
             capsize=3, capthick=1.5,
             color='b',
             ms=5, zorder=20)
plt.subplot(122)
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2], ls='', marker='.',
             capsize=3, capthick=1.5,
             color='b',
             ms=5, zorder=20)

fig_spectrum, ax_spectrum = plt.subplots(figsize=(10, 4.5))
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2], ls='', marker='.',
             capsize=3, capthick=1.5,
             color='b',
             ms=5, zorder=20)


xx_plot = np.linspace(0, 150, num=500)

print('\nPL + EBL cutoff')

# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_PL' + bbb + '.yaml')

for nd, d in enumerate(my_ebl):
    print(d)

    opacity = opacities_array[d]

    def funct_mk501_inside(ee_array, phi0, gamma):
        return (phi0 * ee_array ** 2.
                * ee_array ** (-gamma)
                * np.exp(-opacity(np.log10(ee_array)))
                )

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               phi0=150, gamma=2.)

    m.migrad()
    m.hesse()

    print(m.params)

    print(f"{m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    print(1 - stats.chi2.cdf(x=m.fval, df=m.ndof))

    ax_spectrum.plot(xx_array,
                     funct_mk501_inside(xx_array, *m.values),
                     label=labelss[d], c=colors[d], lw=1)


    ax_spectrum2[0].plot(xx_array,
                     funct_mk501_inside(xx_array, *m.values),
                     c=colors[d], lw=1)

print('\nPL1 + CPL12 + EBL absorption')
for nd, d in enumerate(my_ebl):
    print(d)

    opacity = opacities_array[d]

    def funct_mk501_inside(xx, N0, gamma1, gamma2, Ebreak, fi):
        return (N0 * xx ** 2.
                * xx ** (-gamma1)
                * (1. + (xx / Ebreak) ** fi) ** ((gamma1 - gamma2) / fi)
                * np.exp(-opacity(np.log10(xx)))
                )

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               N0=200, gamma1=2., gamma2=2.45, Ebreak=2.1, fi=2.)

    # m.limits['gamma1'] = (0., 10.)
    # m.limits['gamma2'] = (0., 10.)
    # m.limits['fi'] = (0., 10.)
    # m.limits['Ebreak'] = (1., 10.)

    m.fixed['fi'] = True

    m.migrad()
    m.hesse()

    print(m.params)

    print(f"{m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}")
    print(1 - stats.chi2.cdf(x=m.fval, df=m.ndof))

    ax_spectrum.plot(xx_array,
                     funct_mk501_inside(xx_array, *m.values),
                     c=colors[d],
                     ls=':', lw=1)

    ax_spectrum2[1].plot(xx_array,
                     funct_mk501_inside(xx_array, *m.values),
                     c=colors[d],
                     label=labelss[d], lw=1)

plt.xscale('log')
plt.yscale('log')

plt.xlim(0.5, 23)
plt.ylim(1, 200)

legend22 = plt.legend()
handles, labels = ax_spectrum.get_legend_handles_labels()
handles = [mpatches.Patch(color=handles[i].get_c(), alpha=0.8)
           for i in range(len(handles))]
legend22 = plt.legend(handles, labels, loc=3, handlelength=0.9)

handles = (plt.Line2D([], [], linestyle='-', color='k'),
           plt.Line2D([], [], linestyle=':', color='k')
                      )
# legend11 = plt.legend(handles, ['PL + EBL', 'BPL + EBL'],
#                       loc=3,
#                       # fontsize=11.5,
#                       )

# ax_spectrum.add_artist(legend11)
ax_spectrum.add_artist(legend22)

plt.text(x=1.3, y=50, s='PL + EBL')

plt.xlabel('E [TeV]')
plt.ylabel(r'$E^2dN/dE$ [10$^{−12}$ erg cm$^{−2}$ s$^{−1}$]')

plt.savefig('outputs/lhaaso_syst10percent/fit_to_hegra_spectrum.png',
            bbox_inches='tight', dpi=300)
plt.savefig('outputs/lhaaso_syst10percent/fit_to_hegra_spectrum.pdf',
            bbox_inches='tight')


plt.figure(fig_spectrum2)
plt.subplot(121)
plt.xscale('log')
plt.yscale('log')

plt.xlim(0.5, 23)
plt.ylim(1, 200)

plt.xlabel(r'E $\left(\mathrm{TeV}\right)$')
plt.ylabel(r'$E^2dN/dE$ '
           r'$\left(10^{-12}\, \mathrm{erg} \,\mathrm{cm}^{'
           r'-2}\,\mathrm{s}^{-1}\right)$',
           size=20)

plt.text(x=1, y=10, s='PL + EBL')

plt.subplot(122)
plt.xscale('log')
plt.yscale('log')

plt.xlim(0.5, 23)
plt.ylim(1, 200)

plt.tick_params('y', labelleft=False)

plt.text(x=1, y=10, s='BPL + EBL')
plt.xlabel(r'E $\left(\mathrm{TeV}\right)$')

legend22 = plt.legend(loc=2, bbox_to_anchor=(1.01, 0.95))
# handles, labels = ax_spectrum2[1].get_legend_handles_labels()
# handles = [mpatches.Patch(color=handles[i].get_c(), alpha=0.8)
#            for i in range(len(handles))]
# legend22 = plt.legend(handles, labels, loc=6, handlelength=0.9)

# handles = (plt.Line2D([], [], linestyle='-', color='k'),
#            plt.Line2D([], [], linestyle=':', color='k')
#                       )
# legend11 = plt.legend(handles, ['PL + EBL', 'BPL + EBL'],
#                       loc=3,
#                       # fontsize=11.5,
#                       )

# ax_spectrum2.add_artist(legend11)
# ax_spectrum2.add_artist(legend22)


plt.savefig('outputs/lhaaso_syst10percent/fit_to_hegra_spectrum2.png',
            bbox_inches='tight')
plt.savefig('outputs/lhaaso_syst10percent/fit_to_hegra_spectrum2.pdf',
            bbox_inches='tight')
plt.show()
'''
all_size = 20
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=False, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)


# -------------- Histograms --------------------------------------------
# Power law + EBL cutoff
aaa = 'PL'
spectral_shape = r'PL + EBL'
bins_hist = [np.linspace(190, 220., num=40),
             np.linspace(2.04, 2.25, num=40)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')


fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))

plt.subplot(121)
plt.title(spectral_shape)

fig_cum, ax_cum = plt.subplots(1, 2, figsize=(12, 4))
plt.subplot(121)
# plt.title(spectral_shape)
plt.text(s=spectral_shape,
         x=0.6, y=1.05, color='k', ha='right',
         transform=ax_cum[0].transAxes)

yy_positions = 1.1

for ni, i in enumerate(my_ebl):

    plt.figure(fig_hist)
    plt.subplot(121)
    hist_distr = (2 * (np.array(output_data[i]['logL' + bbb])
                       - np.array(output_data['logL_poisson_with_itself'])))
    asimov_vert_line = (2 * (output_data[i]['logL_asimov' + bbb]
                             - output_data[vert_lines_against][
                                 'logL_asimov' + bbb]))

    plt.hist(hist_distr,
             alpha=0.4, color=colors[i],
             bins=30, label=labelss[i], hatch=hatches[ni],
              histtype='stepfilled')

    plt.axvline(asimov_vert_line, ls='-', color=colors[i])

    plt.figure(fig_cum)
    plt.subplot(121)
    nn = plt.hist(hist_distr,
                  alpha=0.4, color=colors[i],
                  bins=30, label=labelss[i], hatch=hatches[ni], cumulative=True,
                  density=True, histtype='stepfilled')
    bins = np.array(nn[0])
    nn = nn[1]

    if i == vert_lines_against:
        def cost_funct_kstest(df):
            chi2_distrib = stats.chi2(df=df)
            ks_results = stats.kstest(hist_distr, chi2_distrib.cdf)
            return -ks_results.pvalue


        m_p = Minuit(cost_funct_kstest, df=16.)

        # m_p.limits['df'] = (0., 25.)

        m_p.migrad()
        m_p.hesse()

        xx_plot = np.linspace(0, 141., num=500)
        plt.xlim(0, 89)
        plt.plot(xx_plot, stats.chi2.cdf(x=xx_plot, df=m_p.values),
                 c=colors[i])
        plt.text(s='Fit to cdf:\ndf=%.4f +- %.4f\np-value=%.4f'
                   % (m_p.values[0], m_p.errors[0], -m_p.fval),
                 x=0.8, y=1.02, color='k',
                 transform=ax_cum[0].transAxes,
                 fontsize=18, horizontalalignment='center')

    else:
        xx_int = np.linspace(asimov_vert_line, 150, num=500)
        int_result = simpson(y=stats.chi2.pdf(x=xx_int, df=m_p.values),
                             x=xx_int)
        plt.text(s='%.4f' % (1 - int_result),
                 x=asimov_vert_line + 3., y=yy_positions, color=colors[i])
        yy_positions += 0.1

    plt.axvline(asimov_vert_line, ls='-', color=colors[i])


# # 2 Power law + EBL cutoff
aaa = '2PL'
spectral_shape = r'BPL + EBL'
bins_hist = [np.linspace(190., 340., num=40),
             np.linspace(1.2, 2.1, num=30),
             np.linspace(2.1, 3.4, num=50),
             np.linspace(1., 10., num=40)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')

plt.figure(fig_hist)
plt.subplot(122)
plt.title(spectral_shape)

plt.figure(fig_cum)
plt.subplot(122)
# plt.title(spectral_shape)
plt.text(s=spectral_shape,
         x=0.6, y=1.05, color='k', ha='right',
         transform=ax_cum[1].transAxes)

hatches = ['/', '', '']
yy_positions = 1.1

for ni, i in enumerate(my_ebl):
    color = colors[i]

    plt.figure(fig_hist)
    plt.subplot(122)
    hist_distr = (2 * (np.array(output_data[i]['logL' + bbb])
                       - np.array(output_data['logL_poisson_with_itself'])))
    asimov_vert_line = (2 * (output_data[i]['logL_asimov' + bbb]
                             - output_data[vert_lines_against][
                                 'logL_asimov' + bbb]))

    plt.hist(hist_distr,
             alpha=0.4, color=color,
             bins=30, label=labelss[i], hatch=hatches[ni],
             histtype='stepfilled')

    plt.axvline(asimov_vert_line, ls='-', color=color)

    plt.figure(fig_cum)
    plt.subplot(122)
    nn = plt.hist(hist_distr,
                  alpha=0.4, color=color,
                  bins=30, label=labelss[i], hatch=hatches[ni],
                  cumulative=True,
                  density=True, histtype='stepfilled')
    bins = np.array(nn[0])
    nn = nn[1]

    if i == vert_lines_against:
        def cost_funct_kstest(df):
            chi2_distrib = stats.chi2(df=df)
            ks_results = stats.kstest(hist_distr, chi2_distrib.cdf)
            return -ks_results.pvalue


        m_p = Minuit(cost_funct_kstest, df=16.)

        # m_p.limits['df'] = (0., 25.)

        m_p.migrad()
        m_p.hesse()

        xx_plot = np.linspace(0, 42.118, num=500)
        plt.xlim(0, 37.05)
        plt.plot(xx_plot, stats.chi2.cdf(x=xx_plot, df=m_p.values),
                 c=color)
        plt.text(s='Fit to cdf:\ndf=%.4f +- %.4f\np-value=%.4f'
                   % (m_p.values[0], m_p.errors[0], -m_p.fval),
                 x=0.8, y=1.02, color='k',
                 transform=ax_cum[1].transAxes,
                 fontsize=18, horizontalalignment='center')

    else:
        xx_int = np.linspace(asimov_vert_line, 150, num=500)
        int_result = simpson(y=stats.chi2.pdf(x=xx_int, df=m_p.values),
                             x=xx_int)
        plt.text(s='%.4f' % (1 - int_result),
                 x=asimov_vert_line + 3., y=yy_positions, color=color)
        yy_positions += 0.1

    plt.axvline(asimov_vert_line, ls='-', color=color)

plt.legend(framealpha=0.9)

plt.xlabel(r'$- 2 \Delta$log $L$')
plt.ylim(0, yy_positions)

plt.subplot(121)
plt.xlabel(r'$- 2 \Delta$log $L$')
plt.ylim(0, yy_positions)

plt.savefig('outputs/lhaaso_syst10percent/cum_hist_3eblmodels_'
            + aaa + bbb + '_paper.png',
            bbox_inches='tight')
plt.savefig('outputs/lhaaso_syst10percent/cum_hist_3eblmodels_'
            + aaa + bbb + '_paper.pdf',
            bbox_inches='tight')



plt.figure(fig_hist)

plt.legend()
plt.xlabel(r'$- 2 \Delta$log $L$')

plt.subplot(121)
plt.xlabel(r'$- 2 \Delta$log $L$')

plt.savefig('outputs/lhaaso_syst10percent/hist_3eblmodels_'
            + aaa + bbb + '_paper.png',
            bbox_inches='tight')
plt.savefig('outputs/lhaaso_syst10percent/hist_3eblmodels_'
            + aaa + bbb + '_paper.pdf',
            bbox_inches='tight')

# plt.show()


# -------------- Histograms --------------------------------------------
# Power law + EBL cutoff
aaa = 'PL'
spectral_shape = r'PL + EBL'
bins_hist = [np.linspace(190, 220., num=40),
             np.linspace(2.04, 2.25, num=40)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')

plt.rc('ytick.major', size=10, width=1., right=True, pad=5)
plt.rc('xtick.major', top=True)
px = 1/plt.rcParams['figure.dpi']
fig_hist, ax_hist = plt.subplots(2, 2, figsize=(1500*px, 860*px))
fig_hist.subplots_adjust(hspace=0., wspace=0.)

plt.subplot(221)
plt.title(spectral_shape)


for ni, i in enumerate(my_ebl):

    plt.figure(fig_hist)
    plt.subplot(221)
    hist_distr = (2 * (np.array(output_data[i]['logL' + bbb])
                       - np.array(output_data['logL_poisson_with_itself'])))
    asimov_vert_line = (2 * (output_data[i]['logL_asimov' + bbb]
                             - output_data[vert_lines_against][
                                 'logL_asimov' + bbb]))

    plt.hist(hist_distr,
             alpha=0.4, color=colors[i],
             bins=30, label=labelss[i], hatch=hatches[ni],
             histtype='stepfilled')

    plt.axvline(asimov_vert_line, ls='-', color=colors[i], lw=2)

    plt.subplot(223)
    nn = plt.hist(hist_distr,
                  alpha=0.4, color=colors[i],
                  bins=30, label=labelss[i], hatch=hatches[ni], cumulative=True,
                  density=True, histtype='stepfilled')
    bins = np.array(nn[0])
    nn = nn[1]

    if i == vert_lines_against:
        def cost_funct_kstest(df):
            chi2_distrib = stats.chi2(df=df)
            ks_results = stats.kstest(hist_distr, chi2_distrib.cdf)
            return -ks_results.pvalue


        m_p = Minuit(cost_funct_kstest, df=16.)

        m_p.limits['df'] = (0., 25.)

        m_p.migrad()
        m_p.hesse()

        xx_plot = np.linspace(0, 95., num=500)
        plt.plot(xx_plot, stats.chi2.cdf(x=xx_plot, df=m_p.values),
                 c='k')

        plt.text(s='Fit to cdf:\n'
                   'k = %.3f'
                   r' $\pm$ '
                   '%.3f'
                   # '\n'
                   # 'p-value=%.3f'
                   # % (m_p.values[0], m_p.errors[0], -m_p.fval),
                   % (m_p.values[0], m_p.errors[0]),
                 x=69, y=0.15, color='k',
                 fontsize=18, horizontalalignment='center',
                 bbox=dict(facecolor='w', alpha=1, edgecolor='grey'),)

    plt.axvline(asimov_vert_line, ls='-', color=colors[i], lw=2)


# # 2 Power law + EBL cutoff
aaa = '2PL'
spectral_shape = r'BPL + EBL'
bins_hist = [np.linspace(190., 340., num=40),
             np.linspace(1.2, 2.1, num=30),
             np.linspace(2.1, 3.4, num=50),
             np.linspace(1., 10., num=40)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')

plt.subplot(222)
plt.title(spectral_shape)

hatches = ['/', '', '']
yy_positions = 1.1

for ni, i in enumerate(my_ebl):
    color = colors[i]

    plt.subplot(222)
    hist_distr = (2 * (np.array(output_data[i]['logL' + bbb])
                       - np.array(output_data['logL_poisson_with_itself'])))
    asimov_vert_line = (2 * (output_data[i]['logL_asimov' + bbb]
                             - output_data[vert_lines_against][
                                 'logL_asimov' + bbb]))

    plt.hist(hist_distr,
             alpha=0.4, color=color,
             bins=30, label=labelss[i], hatch=hatches[ni],
             histtype='stepfilled')

    plt.axvline(asimov_vert_line, ls='-', color=color, lw=2)

    plt.subplot(224)
    nn = plt.hist(hist_distr,
                  alpha=0.4, color=color,
                  bins=30, label=labelss[i], hatch=hatches[ni], cumulative=True,
                  density=True, histtype='stepfilled')
    bins = np.array(nn[0])
    nn = nn[1]

    if i == vert_lines_against:
        def cost_funct_kstest(df):
            chi2_distrib = stats.chi2(df=df)
            ks_results = stats.kstest(hist_distr, chi2_distrib.cdf)
            return -ks_results.pvalue


        m_p = Minuit(cost_funct_kstest, df=16.)

        m_p.limits['df'] = (0., 25.)

        m_p.migrad()
        m_p.hesse()

        xx_plot = np.linspace(0, 38, num=500)
        plt.plot(xx_plot, stats.chi2.cdf(x=xx_plot, df=m_p.values),
                 c='k')

        plt.text(s='Fit to cdf:\n'
                   'k = %.4f'
                   r' $\pm$ '
                   '%.4f'
                   % (m_p.values[0], m_p.errors[0]),
                 x=23, y=0.15, color='k',
                 fontsize=18, horizontalalignment='center',
                 bbox=dict(facecolor='w', alpha=1, edgecolor='grey'), )

    plt.axvline(asimov_vert_line, ls='-', color=color, lw=2)


plt.subplot(221)
plt.ylim(0, 130)
plt.xlim(-0.5, 89)
plt.tick_params('x', labelbottom=False)

plt.ylabel('Histogram')

import matplotlib.transforms as trans
offset = trans.ScaledTranslation(0, 8/72., fig_hist.dpi_scale_trans)
for nn, label in enumerate(ax_hist[0][0].yaxis.get_majorticklabels()):
    if nn == 0:
        label.set_transform(label.get_transform() + offset)


plt.subplot(222)
plt.ylim(0, 130)
plt.xlim(-0.1, 37.05)
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.legend(framealpha=0.9)


plt.subplot(223)
plt.xlabel(r'$- 2 \Delta$log $L$')
plt.ylim(0, 1.05)
plt.xlim(-0.5, 89)
plt.tick_params('x', top=True)

plt.ylabel('Cumulative histogram')

# offset = trans.ScaledTranslation(0, 6/72., fig_hist.dpi_scale_trans)
# for nn, label in enumerate(ax_hist[1][0].yaxis.get_majorticklabels()):
#     if nn == 4:
#         label.set_transform(label.get_transform() - offset)

plt.subplot(224)
plt.xlabel(r'$- 2 \Delta$log $L$')
plt.ylim(0, 1.05)
plt.xlim(-0.1, 37.05)
plt.tick_params('y', labelleft=False)
plt.tick_params('x', top=True)

plt.savefig('outputs/lhaaso_syst10percent/histandcum_3eblmodels_'
            + aaa + bbb + '_paper.png',
            bbox_inches='tight', dpi=200)
plt.savefig('outputs/lhaaso_syst10percent/histandcum_3eblmodels_'
            + aaa + bbb + '_paper.pdf',
            bbox_inches='tight')

# -------------- Histograms --------------------------------------------
'''
# Power law + EBL cutoff
aaa = 'PL'
spectral_shape = r'PL + EBL'
bins_hist = [np.linspace(190, 220., num=40),
             np.linspace(2.04, 2.25, num=40)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')

plt.rc('ytick.major', size=10, width=1., right=True, pad=5)
plt.rc('xtick.major', top=True)
px = 1/plt.rcParams['figure.dpi']
fig_hist, ax_hist = plt.subplots(2, 1, figsize=(787*px, 819*px))
fig_hist.subplots_adjust(hspace=0., wspace=0.)


for ni, i in enumerate(my_ebl):

    plt.figure(fig_hist)
    plt.subplot(211)
    hist_distr = (2 * (np.array(output_data[i]['logL' + bbb])
                       - np.array(output_data['logL_poisson_with_itself'])))
    asimov_vert_line = (2 * (output_data[i]['logL_asimov' + bbb]
                             - output_data[vert_lines_against][
                                 'logL_asimov' + bbb]))

    plt.hist(hist_distr,
             alpha=0.4, color=colors[i],
             bins=30, label=labelss[i], hatch=hatches[ni],
             histtype='stepfilled')

    plt.axvline(asimov_vert_line, ls='-', color=colors[i], lw=2)

    plt.subplot(212)
    nn = plt.hist(hist_distr,
                  alpha=0.4, color=colors[i],
                  bins=30, label=labelss[i], hatch=hatches[ni], cumulative=True,
                  density=True, histtype='stepfilled')
    bins = np.array(nn[0])
    nn = nn[1]

    if i == vert_lines_against:
        def cost_funct_kstest(df):
            chi2_distrib = stats.chi2(df=df)
            ks_results = stats.kstest(hist_distr, chi2_distrib.cdf)
            return -ks_results.pvalue


        m_p = Minuit(cost_funct_kstest, df=16.)

        m_p.limits['df'] = (0., 25.)

        m_p.migrad()
        m_p.hesse()

        xx_plot = np.linspace(0, 141., num=500)
        plt.plot(xx_plot, stats.chi2.cdf(x=xx_plot, df=m_p.values),
                 c='k')

    plt.axvline(asimov_vert_line, ls='-', color=colors[i], lw=2)

plt.subplot(211)
plt.ylim(0, 130)
plt.xlim(-0.5, 117)
plt.tick_params('x', labelbottom=False)

plt.ylabel('Histogram')

import matplotlib.transforms as trans
offset = trans.ScaledTranslation(0, 8/72., fig_hist.dpi_scale_trans)
for nn, label in enumerate(ax_hist[0].yaxis.get_majorticklabels()):
    if nn == 0:
        label.set_transform(label.get_transform() + offset)


plt.subplot(212)
plt.xlabel(r'$- 2 \Delta$log $L$')
plt.ylim(0, 1.05)
plt.xlim(-0.5, 117)
plt.tick_params('x', top=True)

plt.ylabel('Cumulative histogram')

offset = trans.ScaledTranslation(0, 6/72., fig_hist.dpi_scale_trans)
for nn, label in enumerate(ax_hist[1].yaxis.get_majorticklabels()):
    if nn == 4:
        label.set_transform(label.get_transform() - offset)


plt.savefig('outputs/lhaaso_syst10percent/histandcum_3eblmodels_'
            + aaa + bbb + '_poster.png',
            bbox_inches='tight', dpi=300)
plt.savefig('outputs/lhaaso_syst10percent/histandcum_3eblmodels_'
            + aaa + bbb + '_poster.pdf',
            bbox_inches='tight')

plt.show()
'''
# -----------------------------------------------------------------------------
plt.rc('ytick.major', size=5, width=1., right=False, pad=5)
fig, axes = plt.subplots(2, 4, layout="constrained", figsize=(14, 8))
fig.subplots_adjust(hspace=0.8, wspace=0.22)  # hspace=1., wspace=0.4
# fig.tight_layout()

hatches = ['/', '', '']
colors = {'bosa.txt': 'tab:blue',
          '2bb.txt': 'tab:green',
          'chary.txt': 'darkorange'}

# Power law + EBL cutoff
aaa = 'PL'
spectral_shape = r'PL + EBL'

plt.text(x=1.13, y=1.1, s=spectral_shape, fontweight='bold',
         horizontalalignment='center',
         transform=axes[0][1].transAxes, fontsize=22)

bins_hist = [np.linspace(194, 220., num=40) * (u.erg.to(u.TeV)),
             np.linspace(2.0875, 2.266, num=40)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')

number_of_params = 2

plt.rc('text', usetex=False)
param_names = [
    r'$\phi_0$',
    r'$\Gamma$']

plt.text(x=0.5, y=-0.45,
         s=r'$\left(10^{-12}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{TeV}^{-1}\right)$',
         horizontalalignment='center',
         transform=axes[0][1].transAxes, fontsize=18)

ax = axes[0][1]
ax.set_xlabel(param_names[0])
ax.set_xlim(bins_hist[0][0], bins_hist[0][-1])

for ni, i in enumerate(my_ebl):
    color = colors[i]
    param_array = np.array(output_data[i]['params_values' + bbb])

    if vert_lines_against == i:
        ax.axvline(
            output_data[i]['asimov']['params_values'][0]* (u.erg.to(u.TeV)),
            c='k', lw=1, ls=':')

    if number_of_params - 1 == 0:
        ax.hist(param_array[:, 0] * (u.erg.to(u.TeV)), color=color,
                bins=bins_hist[0], hatch=hatches[ni], alpha=0.4,
                label=labelss[i], histtype='stepfilled')
    else:
        ax.hist(param_array[:, 0] * (u.erg.to(u.TeV)), color=color,
                bins=bins_hist[0], hatch=hatches[ni], alpha=0.4,
                histtype='stepfilled')

ax.tick_params(axis='y', labelsize=18)

ax = axes[0][2]
ax.set_xlabel(param_names[1])
ax.set_xlim(bins_hist[1][0], bins_hist[1][-1])

for ni, i in enumerate(my_ebl):
    color = colors[i]
    param_array = np.array(output_data[i]['params_values' + bbb])

    if vert_lines_against == i:
        ax.axvline(
            output_data[i]['asimov']['params_values'][1],
            c='k', lw=1, ls=':')

    if number_of_params - 1 == 1:
        ax.hist(param_array[:, 1], color=color,
                bins=bins_hist[1], hatch=hatches[ni], alpha=0.4,
                label=labelss[i], histtype='stepfilled')
    else:
        ax.hist(param_array[:, 1], color=color,
                bins=bins_hist[1], hatch=hatches[ni], alpha=0.4,
                histtype='stepfilled')

ax.tick_params(axis='y', labelsize=18)

ax.legend(loc=6, bbox_to_anchor=(1.02, 0.5))

# # 2 Power law + EBL cutoff
aaa = '2PL'
spectral_shape = r'BPL + EBL'

plt.text(x=1.13, y=1.1, s=spectral_shape, fontweight='bold',
         horizontalalignment='center',
         transform=axes[1][1].transAxes, fontsize=22)

bins_hist = [np.linspace(188., 300., num=40)* (u.erg.to(u.TeV)),
             np.linspace(1.3, 2.09, num=40),
             np.linspace(2.2, 3.5, num=40),
             np.linspace(0.77, 10., num=40)]

# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso_syst10percent/asimov_dict_' + aaa + bbb + '.yaml')

param_names = [
    r'$\phi_0$',
    r'$\Gamma_1$',
    r'$\Gamma_2$',
    r'$E_\mathrm{break}$ $\left(\mathrm{TeV}\right)$']


plt.text(x=0.5, y=-0.45,
         s=r'$\left(10^{-12}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{TeV}^{-1}\right)$',
         horizontalalignment='center',
         transform=axes[1][0].transAxes, fontsize=18)

ax = axes[1][0]
ax.set_xlabel(param_names[0])
ax.tick_params(axis='y', labelsize=18)  # labelrotation=90
ax.set_xlim(bins_hist[0][0], bins_hist[0][-1])

for ni, i in enumerate(my_ebl):
    color = colors[i]

    param_array = np.array(output_data[i]['params_values' + bbb])

    if vert_lines_against == i:
        ax.axvline(
            output_data[i]['asimov']['params_values'][0]
            * (u.erg.to(u.TeV)),
            c='k', lw=1, ls=':')

    if number_of_params - 1 == 0:
        ax.hist(param_array[:, 0]* (u.erg.to(u.TeV)), color=color,
                 bins=bins_hist[0], hatch=hatches[ni], alpha=0.4,
                 label=labelss[i], histtype='stepfilled')
    else:
        ax.hist(param_array[:, 0]* (u.erg.to(u.TeV)), color=color,
                 bins=bins_hist[0], hatch=hatches[ni], alpha=0.4,
                histtype='stepfilled')

for param in range(1, 4):
    ax = axes[1][param]
    ax.set_xlabel(param_names[param])
    ax.tick_params(axis='y', labelsize=18)  # labelrotation=90
    ax.set_xlim(bins_hist[param][0], bins_hist[param][-1])

    for ni, i in enumerate(my_ebl):
        color = colors[i]

        param_array = np.array(output_data[i]['params_values' + bbb])

        if vert_lines_against == i:
            ax.axvline(
                output_data[i]['asimov']['params_values'][param],
                c='k', lw=1, ls=':')

        if param == number_of_params - 1:
            ax.hist(param_array[:, param], color=color,
                     bins=bins_hist[param], hatch=hatches[ni], alpha=0.4,
                     label=labelss[i], histtype='stepfilled')
        else:
            ax.hist(param_array[:, param], color=color,
                     bins=bins_hist[param], hatch=hatches[ni], alpha=0.4,
                    histtype='stepfilled')

axes[0, 0].axis("off")
axes[0, 3].axis("off")

axes[1, 3].set_xticks([1, 5, 10])

plt.savefig('outputs/lhaaso_syst10percent/params_3eblmodels_' + aaa + bbb + '_paper4.png',
            bbox_inches='tight', dpi=500)
plt.savefig('outputs/lhaaso_syst10percent/params_3eblmodels_' + aaa + bbb + '_paper4.pdf',
            bbox_inches='tight')


plt.show()
