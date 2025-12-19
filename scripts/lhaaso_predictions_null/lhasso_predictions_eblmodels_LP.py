import os
import time
import yaml

import iminuit.cost
import numpy as np
import scipy.optimize
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

import astropy.units as u

from scripts.e_disp_code import EDispGauss

from iminuit import Minuit
from iminuit.cost import LeastSquares

from ebltable.ebl_from_model import EBL
from ebltable.tau_from_model import OptDepth

os.chdir("..")
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


# ----------------------------------------------------------------------
def likelihood_poisson(n_i_array_obs, mu_i_array_expected):
    return sum(n_i_array_obs * np.log(mu_i_array_expected)
               - mu_i_array_expected
               )


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

plt.figure()
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='o', label='Reanalysis')
plt.errorbar(mkr501_flux_1[:, 0], mkr501_flux_1[:, 1],
             yerr=mkr501_flux_1[:, 2],
             ls='', marker='o', label='Original analysis')

mkr501_flux = np.concatenate((mkr501_flux_1[:8, :3], mkr501_flux))

plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
             yerr=mkr501_flux[:, 2],
             ls='', marker='+', label='Total')
plt.show()

zz = 0.034

e_array = np.geomspace(
    np.min(mkr501_flux[:, 0]) * 0.5,
    np.max(mkr501_flux[:, 0]) * 1.5,
    num=500)

# ----------------------------------------------------------------------
xxx_bins = np.geomspace(
    min(mkr501_flux[:, 0]) - 0.5,
    max(mkr501_flux[:, 0]) + 0.5,
    num=1000) * u.TeV
# xxx_means = (xxx_bins[1:] + xxx_bins[:-1]) / 2.
xxx_means = np.sqrt(xxx_bins[1:] * xxx_bins[:-1])

recovered_bins = xxx_bins.copy()  # [:-2]
recovered_means = xxx_means.copy()  # [:-2]

edisp_obj = EDispGauss(sigma=0.2, bias=0.)
edisp_obj.fill(e_true_edges=xxx_bins.value,
               e_reco_edges=recovered_bins.value)
plt.figure()
plt.title('e disp matrix we use in the calculations')
edisp_obj.plot()

eff_area = np.loadtxt('data/lhasso_characteristics/'
                      'WCD_eff_area_0to15deg.txt')
eff_area = eff_area[np.argsort(eff_area[:, 0]), :]
eff_area[:, 1] = eff_area[:, 1] * (u.m ** 2).to(u.cm ** 2)

spline_eff_area = UnivariateSpline(
    x=eff_area[:, 0] - 3., y=np.log10(eff_area[:, 1]),
    s=0, k=1, ext=1
)

energy_bins_obs = np.geomspace(0.5, 21., num=20) * u.TeV
energy_means_obs = np.sqrt(energy_bins_obs[1:] * energy_bins_obs[:-1])

total_int_time = (820. * u.h).to(u.s)


def calculate_number_counts(flux_spectrum_funct, iminuit_object_values):
    integral_kernel = flux_spectrum_funct(
        xxx_means.value, *iminuit_object_values)
    integral_kernel = (integral_kernel * 1e-12
                       * u.erg * u.cm ** -2 * u.s ** -1)
    integral_kernel = (integral_kernel / xxx_means ** 2.).to(
        u.TeV ** -1 * u.cm ** -2 * u.s ** -1)

    integral_kernel *= 10 ** spline_eff_area(
        x=np.log10(xxx_means.value)) * u.cm ** 2

    integral_kernel *= np.log(10) * xxx_means
    integral_kernel = integral_kernel.to(1 / u.s)[:, np.newaxis]

    integral_kernel = integral_kernel * edisp_obj._pdf_matrix

    integral_kernel = simpson(
        y=integral_kernel, x=np.log10(xxx_means.value), axis=0)

    count_numb = []

    integral_kernel = integral_kernel * np.log(10) * recovered_means

    for i in range(len(energy_bins_obs) - 1):
        where_bin = ((recovered_means > energy_bins_obs[i])
                     * (recovered_means < energy_bins_obs[i + 1]))

        if sum(where_bin) > 0:
            count_numb.append(simpson(
                y=integral_kernel[where_bin],
                x=np.log10(recovered_means[where_bin].value)))
        else:
            count_numb.append(0.)

    count_numb = (count_numb * total_int_time).value

    return count_numb


# ----------------------------------------------------------------------
nn_array_mine = []
nn_errors_mine = []
likelihoods_asimov_mine = {}

xx_array = np.geomspace(0.5, 25.)

# aaa_str = '_vs_bosa'
# my_ebl = ['2bb.txt', 'chary.txt', 'bosa.txt']

# aaa_str = '_vs_chary'
# my_ebl = ['2bb.txt', 'bosa.txt', 'chary.txt']

aaa_str = '_vs_2bb'
my_ebl = ['chary.txt', 'bosa.txt', '2bb.txt']

name_model = '_LP'
param_names = ['$\phi_0$', '$E_0$', '$\Gamma$', r'$\beta$']

opacities_array = {}
for d in my_ebl:
    ebl_finke = EBL.readascii(
            'outputs/lhaaso_new/' + d, model_name='mine')
    opacityy = ebl_finke.optical_depth(z0=zz, ETeV=e_array)
    opacities_array[d] = UnivariateSpline(
            np.log10(e_array), opacityy, k=1, s=0)

colors = ['b', 'orange', 'limegreen']

fig_spectrum, ax_spectrum = plt.subplots()
plt.errorbar(mkr501_flux[:, 0], mkr501_flux[:, 1],
        yerr=mkr501_flux[:, 2], ls='', marker='o')

# plt.show()
fig_counts, ax_counts = plt.subplots()

for nd, d in enumerate(my_ebl):
    print(d)

    likelihoods_asimov_mine[d] = {}
    likelihoods_asimov_mine[d]['asimov'] = {}

    opacity = opacities_array[d]

    def funct_mk501_inside(ee_array, phi0, E0, gamma, beta):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (
                        - gamma - beta * np.log(ee_array / E0))
                * np.exp(-opacity(np.log10(ee_array))))

    combined_likelihood = LeastSquares(
        mkr501_flux[:, 0], mkr501_flux[:, 1],
        mkr501_flux[:, 2], funct_mk501_inside)

    m = Minuit(combined_likelihood,
               phi0=140, E0=1., gamma=5., beta=0.5)

    m.migrad()
    m.hesse()

    print(m.params)
    print('$%.3f / %i = %.2f $'
          % (m.fval, m.ndof, m.fmin.reduced_chi2))
    nn_array_mine.append([*m.values])
    nn_errors_mine.append([*m.errors])

    count_number_asimov = calculate_number_counts(
        funct_mk501_inside, m.values)
    ax_spectrum.plot(xx_array,
                        funct_mk501_inside(xx_array, *m.values),
                        label=d + ' fit to obs', c=colors[nd])
    ax_counts.scatter(energy_means_obs, count_number_asimov,
                      label=d + ' from obs', c=colors[nd])

    likelihoods_asimov_mine[d]['asimov']['logL'] = float(
        -likelihood_poisson(
            n_i_array_obs=count_number_asimov,
            mu_i_array_expected=count_number_asimov))
    likelihoods_asimov_mine[d]['asimov']['params_values'] = [*m.values]
    likelihoods_asimov_mine[d]['asimov']['params_errors'] = [*m.errors]

    print(likelihoods_asimov_mine[d]['asimov']['logL'])

likelihoods_asimov_mine['param_names'] = param_names



for nd, d in enumerate(my_ebl):

    opacity = opacities_array[d]

    def funct_mk501_inside(ee_array, phi0, E0, gamma, beta):
        return (phi0 * ee_array ** 2.
                * (ee_array / E0) ** (
                        - gamma - beta * np.log(ee_array / E0))
                * np.exp(-opacity(np.log10(ee_array))))

    def cost_funct(phi0, E0, gamma, beta):

        expected_counts = calculate_number_counts(
            funct_mk501_inside, [phi0, E0, gamma, beta])

        poiss_like = -likelihood_poisson(
            mu_i_array_expected=expected_counts,
            n_i_array_obs=count_number_asimov)

        return poiss_like

    m_p = Minuit(
        cost_funct,
        phi0=likelihoods_asimov_mine[d]['asimov']['params_values'][0],
        E0=likelihoods_asimov_mine[d]['asimov']['params_values'][1],
        gamma=likelihoods_asimov_mine[d]['asimov']['params_values'][2],
        beta=likelihoods_asimov_mine[d]['asimov']['params_values'][3],)

    m_p.migrad()
    m_p.hesse()
    print(m_p.params)
    print(m_p.fval)

    ax_counts.plot(energy_means_obs, calculate_number_counts(
            funct_mk501_inside, [*m_p.values]), c=colors[nd], ls='--',
                   label=d + ' logL min' + aaa_str)

    if d not in likelihoods_asimov_mine.keys():
        likelihoods_asimov_mine[d] = {}

    likelihoods_asimov_mine[d]['logL_asimov' + aaa_str] = float(m_p.fval)
    likelihoods_asimov_mine[d]['logL' + aaa_str] = []
    likelihoods_asimov_mine[d]['params_values' + aaa_str] = []
    likelihoods_asimov_mine[d]['params_errors' + aaa_str] = []


plt.figure(fig_spectrum)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Mkr 501 flare 1997')
plt.xlabel('E [TeV]')
plt.ylabel('E2dN/dE [10−12 erg cm−2 s−1]')
plt.savefig('outputs/lhaaso_new/spectra_ours' + name_model + aaa_str + '.png',
            bbox_inches='tight')
plt.figure(fig_counts)
plt.xscale('log')
plt.legend()
plt.xlabel('E [TeV]')
plt.ylabel('Count number')
plt.savefig('outputs/lhaaso_new/counts_ours' + name_model + aaa_str + '.png',
            bbox_inches='tight')
print(likelihoods_asimov_mine)

with open(
        'outputs/lhaaso_new/asimov_dict' + name_model + aaa_str +
        '_short.yaml', 'w'
) as file:
    yaml.dump(likelihoods_asimov_mine, file)

plt.show()

# ----------------------------------------------------------------------

nn_array = []
nn_errors = []
likelihoods_array = []

likelihoods_asimov_mine['logL_poisson_with_itself'] = []

rng = np.random.default_rng()
init_time = time.process_time()

for nn in range(1000):

    poiss = rng.poisson(lam=count_number_asimov)
    likelihoods_asimov_mine['logL_poisson_with_itself'].append(
        float(-likelihood_poisson(n_i_array_obs=poiss,
                                  mu_i_array_expected=poiss))
    )

    for d in my_ebl:

        opacity = opacities_array[d]

        def funct_mk501_inside(ee_array, phi0, E0, gamma, beta):
            return (phi0 * ee_array ** 2.
                    * (ee_array / E0) ** (
                            - gamma - beta * np.log(ee_array / E0))
                    * np.exp(-opacity(np.log10(ee_array))))

        def cost_funct(phi0, E0, gamma, beta):
            expected_counts = calculate_number_counts(
                funct_mk501_inside, [phi0, E0, gamma, beta])

            poiss_like = -likelihood_poisson(
                mu_i_array_expected=expected_counts,
                n_i_array_obs=poiss)

            return poiss_like


        m_p = Minuit(
            cost_funct,
            phi0=likelihoods_asimov_mine[d]['asimov']['params_values'][0],
            E0=likelihoods_asimov_mine[d]['asimov']['params_values'][1],
            gamma=likelihoods_asimov_mine[d]['asimov']['params_values'][2],
            beta=likelihoods_asimov_mine[d]['asimov']['params_values'][3],
        )

        m_p.migrad()
        m_p.hesse()

        likelihoods_asimov_mine[d]['logL' + aaa_str].append(m_p.fval)
        likelihoods_asimov_mine[d]['params_values' + aaa_str].append(
            [*m_p.values])
        likelihoods_asimov_mine[d]['params_errors' + aaa_str].append(
            [*m_p.errors])

    if nn % 50 == 0:
        print('%i Time: %.2fmin'
              % (nn, (time.process_time() - init_time)/60.))
        init_time = time.process_time()

        with open('outputs/lhaaso_new/asimov_dict' + name_model + aaa_str +
                  '.yaml',
                  'w') as file:
            yaml.dump(likelihoods_asimov_mine, file)

with open('outputs/lhaaso_new/asimov_dict' + name_model + aaa_str + '.yaml',
          'w') as file:
    yaml.dump(likelihoods_asimov_mine, file)
