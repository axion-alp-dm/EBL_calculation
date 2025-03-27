import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import simpson

from iminuit import Minuit
from iminuit.cost import LeastSquares

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


bbb = '_vs_bosa'
vert_lines_against = 'BOSA.txt'
my_ebl = ['BOSA.txt', '3_grey_bodies.txt', 'Chary.txt']
#
bbb = '_vs_chary'
vert_lines_against = 'Chary.txt'
my_ebl = ['Chary.txt', 'BOSA.txt', '3_grey_bodies.txt']
#
bbb = '_vs_3body'
vert_lines_against = '3_grey_bodies.txt'
my_ebl = ['3_grey_bodies.txt', 'BOSA.txt', 'Chary.txt']

# ------------------------------------------------
# LogParabola
# aaa = 'logParabola'

# Power law + EBL cutoff
aaa = 'PWandEBL'
spectral_shape = r'$\phi (E) = \phi_0  E^{-\Gamma} e^{-\tau}$'
bins_hist = [np.linspace(190, 220., num=40),
             np.linspace(2.04, 2.25, num=40)]

# # 2 Power law + EBL cutoff
# aaa = '2PL'
# spectral_shape = r'$\phi (E) = \phi_0  E^{-\Gamma_1}' \
#                  r' \left[1 + \left(\frac{E}{E_\mathrm{' \
#                  r'break}}\right)^{f_i}\right]^{(' \
#                  r'\Gamma_1-\Gamma_2)/f_i}' \
#                  r' e^{-\tau}$'
# bins_hist = [np.linspace(190., 340., num=40),
#              np.linspace(1.2, 2.1, num=30),
#              np.linspace(2.1, 3.4, num=50),
#              np.linspace(1., 10., num=40)]

# Power law + exp cutoff + EBL cutoff
# aaa = 'PWandEXPandEBL'
# spectral_shape = (r'$\phi(E) = \phi_0 '
#                   r'\left(\frac{E}{E_0}\right)^{-\Gamma}'
#                   r' e^{-E/E_\mathrm{cut} - \tau}$')
# bins_hist = [np.linspace(135., 160., num=40),
#              np.linspace(1.8, 2.1, num=30),
#              np.linspace(1.18, 1.23, num=50),
#              np.linspace(0., 60., num=70)]


# We initialize the class with the input file
output_data = read_config_file(
    'outputs/lhaaso/asimov_dict_' + aaa + bbb + '.yaml')

number_of_params = len(output_data['param_names'])
print(output_data['param_names'])

fig_params, ax_params = plt.subplots(
    1, number_of_params, figsize=(4 * number_of_params + 2, 5))
plt.suptitle(spectral_shape)

for param in range(number_of_params):
    plt.subplot(1, number_of_params, param + 1)
    plt.xlabel(output_data['param_names'][param])

fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
plt.suptitle(spectral_shape)

fig_cum, ax_cum = plt.subplots(figsize=(8, 7))
plt.text(s=spectral_shape,
         x=0.6, y=1.05, color='k', ha='right',
         transform=ax_cum.transAxes)

hatches = ['/', '', '']
colors = {'BOSA.txt': 'tab:blue',
          '3_grey_bodies.txt': 'tab:green',
          'Chary.txt': 'darkorange'}
yy_positions = 1.1

for ni, i in enumerate(my_ebl):
    color = colors[i]

    plt.figure(fig_params)

    param_array = np.array(output_data[i]['params_values' + bbb])

    for param in range(number_of_params):
        plt.subplot(1, number_of_params, param + 1)

        if vert_lines_against == i:
            plt.axvline(output_data[i]['asimov']['params_values'][param],
                        c='k')

        if param == number_of_params - 1:
            plt.hist(param_array[:, param], color=color,
                     bins=bins_hist[param], hatch=hatches[ni], alpha=0.4,
                     label=i)
        else:
            plt.hist(param_array[:, param], color=color,
                     bins=bins_hist[param], hatch=hatches[ni], alpha=0.4)

    # --------------------------------------------------------
    plt.figure(fig_hist)
    hist_distr = (2 * (np.array(output_data[i]['logL' + bbb])
                       - np.array(output_data['logL_poisson_with_itself'])))
    asimov_vert_line = (2 * (output_data[i]['logL_asimov' + bbb]
                             - output_data[vert_lines_against][
                                 'logL_asimov' + bbb]))

    plt.hist(hist_distr,
             alpha=0.4, color=color,
             bins=30, label=i, hatch=hatches[ni])

    plt.axvline(asimov_vert_line, ls='-', color=color)

    plt.figure(fig_cum)
    nn = plt.hist(hist_distr,
                  alpha=0.4, color=color,
                  bins=30, label=i, hatch=hatches[ni], cumulative=True,
                  density=True)
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

        xx_plot = np.linspace(0, 125, num=500)
        plt.plot(xx_plot, stats.chi2.cdf(x=xx_plot, df=m_p.values),
                 c=color)
        plt.text(s='Fit to cdf:\ndf=%.4f +- %.4f\np-value=%.4f'
                   % (m_p.values[0], m_p.errors[0], -m_p.fval),
                 x=0.8, y=1.02, color='k',
                 transform=ax_cum.transAxes,
                 fontsize=18, horizontalalignment='center')

    else:
        xx_int = np.linspace(asimov_vert_line, 150, num=500)
        int_result = simpson(y=stats.chi2.pdf(x=xx_int, df=m_p.values),
                             x=xx_int)
        plt.text(s='%.4f' % (1 - int_result),
                 x=asimov_vert_line + 3., y=yy_positions, color=color)
        yy_positions += 0.1

    plt.axvline(asimov_vert_line, ls='-', color=color)

plt.legend()

plt.xlabel(r'- 2 $\Delta$log $L$')
plt.ylim(0, yy_positions)
plt.savefig('outputs/lhaaso/cum_hist_3_eblmodels_' + aaa + bbb + '.png',
            bbox_inches='tight')

plt.figure(fig_hist)
plt.legend()

plt.xlabel(r'- 2 $\Delta$log $L$')

plt.savefig('outputs/lhaaso/hist_3_eblmodels_' + aaa + bbb + '.png',
            bbox_inches='tight')

plt.figure(fig_params)
plt.subplot(1, number_of_params, param + 1)
plt.legend(loc=2, bbox_to_anchor=(1.02, 0.99))
plt.savefig('outputs/lhaaso/params_3_eblmodels_' + aaa + bbb + '.png',
            bbox_inches='tight')

# plt.subplot(1, 4, 4)
# plt.ylim(top=100)
# plt.savefig('outputs/lhaaso/params_3_eblmodels_'
#                 + aaa + bbb + '_zoom.png',
#                 bbox_inches='tight')

plt.show()
