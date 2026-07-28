# IMPORTS --------------------------------------------#
import os
import sys
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as trans
from matplotlib.legend_handler import HandlerTuple

from scipy.integrate import simpson, quad

from scipy.interpolate import UnivariateSpline, RectBivariateSpline

all_size = 16
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
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=10)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=10)
plt.rc('xtick.minor', size=5, width=1.)
plt.rc('ytick.minor', size=5, width=1.)


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


direct_name = ('../outputs/outputs_systematics_14perct')
print(direct_name)

# Configuration file reading and data input/output ---------#
with open(direct_name + '/input_data.yml', 'r') as file:
    config_data = yaml.safe_load(file)

for ni, key in enumerate(config_data['ssp_models']):
    # np.fill_diagonal(copy_cov, 0.)
    # print(np.max(copy_cov), np.min(copy_cov))
    print('\n', key)
    copy_cov = np.array(config_data['ssp_models'][key]['cov_matrix'])
    len_cov = int(np.rint(len(copy_cov) ** 0.5))
    copy_cov = copy_cov.reshape((len_cov, len_cov))

    corr_matrix = np.zeros((len_cov, len_cov))

    for ii in range(len_cov):
        for jj in range(len_cov):
            corr_matrix[ii, jj] = (
                    copy_cov[ii, jj]
                    / copy_cov[ii, ii]**0.5
                    / copy_cov[jj, jj]**0.5)

    corr_matrix = np.delete(corr_matrix, 7, axis=0)
    corr_matrix = np.delete(corr_matrix, 7, axis=1)

    if key == '2bb':
        corr_matrix = np.delete(corr_matrix, 7, axis=0)
        corr_matrix = np.delete(corr_matrix, 7, axis=1)

    print(np.shape(corr_matrix))

    plt.figure()
    plt.title(key)
    plt.imshow(corr_matrix.T, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()

    aa = ''
    new_len = np.shape(corr_matrix)[0]
    for ii in range(new_len):
        for jj in range(new_len):
            if jj < ii:
                aa = aa + ('%.2f'
                           % np.around(corr_matrix[ii, jj],
                                       decimals=2))
            elif jj == ii:
                aa = aa + ('%.0f'
                           % np.around(corr_matrix[ii, jj],
                                       decimals=2))
            aa = aa + ' & '
        print(aa)
        aa = ''

    plt.axvline(3.5, c='grey', alpha=0.5)
    plt.axhline(3.5, c='grey', alpha=0.5)

    if key != 'bosa':
        plt.axvline(6.5, c='grey', alpha=0.5)
        plt.axhline(6.5, c='grey', alpha=0.5)


    plt.xlim(-0.5, new_len-0.5)
    plt.ylim(new_len-0.5, -0.5)

direct_name = ('../outputs/outputs_dust_final_new')
print(direct_name)

# Configuration file reading and data input/output ---------#
with open(direct_name + '/input_data.yml', 'r') as file:
    config_data = yaml.safe_load(file)

key = 'chary'
print('\n', key)
copy_cov = np.array(config_data['ssp_models'][key]['cov_matrix'])
len_cov = int(np.rint(len(copy_cov) ** 0.5))
copy_cov = copy_cov.reshape((len_cov, len_cov))

corr_matrix = np.zeros((len_cov, len_cov))

for ii in range(len_cov):
    for jj in range(len_cov):
        corr_matrix[ii, jj] = (
                copy_cov[ii, jj]
                / copy_cov[ii, ii]**0.5
                / copy_cov[jj, jj]**0.5)

for aa in range(24, 6, -1):
    corr_matrix = np.delete(corr_matrix, aa, axis=0)
    corr_matrix = np.delete(corr_matrix, aa, axis=1)

print(np.shape(corr_matrix))

plt.figure()
plt.title(key)
plt.imshow(corr_matrix.T, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar()

aa = ''
new_len = np.shape(corr_matrix)[0]
for ii in range(new_len):
    for jj in range(new_len):
        aa = aa + ' & '
        if jj < ii:
            aa = aa + ('%.2f'
                       % np.around(corr_matrix[ii, jj],
                                   decimals=2))
        elif jj == ii:
            aa = aa + ('%.0f'
                       % np.around(corr_matrix[ii, jj],
                                   decimals=2))
    print(aa)
    aa = ''

key = 'bosa'
print('\n', key)
copy_cov = np.array(config_data['ssp_models'][key]['cov_matrix'])
len_cov = int(np.rint(len(copy_cov) ** 0.5))
copy_cov = copy_cov.reshape((len_cov, len_cov))

corr_matrix = np.zeros((len_cov, len_cov))

for ii in range(len_cov):
    for jj in range(len_cov):
        corr_matrix[ii, jj] = (
                copy_cov[ii, jj]
                / copy_cov[ii, ii]**0.5
                / copy_cov[jj, jj]**0.5)

corr_matrix = np.delete(corr_matrix, 7, axis=0)
corr_matrix = np.delete(corr_matrix, 7, axis=1)

plt.figure()
plt.title(key)
plt.imshow(corr_matrix.T, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar()


print(np.shape(corr_matrix))


aa = ''
new_len = np.shape(corr_matrix)[0]
for ii in range(new_len):
    for jj in range(new_len):
        aa = aa + ' & '
        if jj < ii:
            aa = aa + ('%.2f'
                       % np.around(corr_matrix[ii, jj],
                                   decimals=2))
        elif jj == ii:
            aa = aa + ('%.0f'
                       % np.around(corr_matrix[ii, jj],
                                   decimals=2))
    print(aa)
    aa = ''

key = '2bb'
print('\n', key)
copy_cov = np.array(config_data['ssp_models'][key]['cov_matrix'])
len_cov = int(np.rint(len(copy_cov) ** 0.5))
copy_cov = copy_cov.reshape((len_cov, len_cov))

corr_matrix = np.zeros((len_cov, len_cov))

for ii in range(len_cov):
    for jj in range(len_cov):
        corr_matrix[ii, jj] = (
                copy_cov[ii, jj]
                / copy_cov[ii, ii]**0.5
                / copy_cov[jj, jj]**0.5)

corr_matrix = np.delete(corr_matrix, 26, axis=0)
corr_matrix = np.delete(corr_matrix, 26, axis=1)
corr_matrix = np.delete(corr_matrix, 25, axis=0)
corr_matrix = np.delete(corr_matrix, 25, axis=1)

for aa in range(22, 6, -1):
    corr_matrix = np.delete(corr_matrix, aa, axis=0)
    corr_matrix = np.delete(corr_matrix, aa, axis=1)

print(np.shape(corr_matrix))

plt.figure()
plt.title(key)
plt.imshow(corr_matrix.T, cmap='RdBu', vmin=-1, vmax=1)
plt.colorbar()

aa = ''
new_len = np.shape(corr_matrix)[0]
for ii in range(new_len):
    for jj in range(new_len):
        aa = aa + ' & '
        if jj < ii:
            aa = aa + ('%.2f'
                       % np.around(corr_matrix[ii, jj],
                                   decimals=2))
        elif jj == ii:
            aa = aa + ('%.0f'
                       % np.around(corr_matrix[ii, jj],
                                   decimals=2))
    print(aa)
    aa = ''


# plt.show()
# ----------------------------------------------------------------------
print('aaa')


# 0.1,0.5,100.
def salpeter55(m):
    alpha = 2.35
    return m ** -alpha


def millerscalo79(m):
    return np.where(m > 1, salpeter55(m), salpeter55(1))


def chabrier03individual(m):
    k = 0.158 * np.exp(-(-np.log10(0.08)) ** 2 / (2 * 0.69 ** 2))
    print(k)

    return np.where(
        m <= 1,
        0.158 * (1. / m) * np.exp(
            -(np.log10(m) - np.log10(0.079)) ** 2 / (2 * 0.69 ** 2)),
        k * m ** -2.3)


def chabrier03system(m):
    k = 0.086 * np.exp(-(-np.log10(0.22)) ** 2 / (2 * 0.57 ** 2))
    return np.where(m <= 1,
                    0.086 * (1. / m) * np.exp(
                        -(np.log10(m) - np.log10(0.22)) ** 2 / (
                                    2 * 0.57 ** 2)),
                    k * m ** -2.3)


def kroupa01(m):
    return np.where(m < 0.08, m ** -0.3,
                    np.where(m < 0.5, 0.08 ** -0.3 * (m / 0.08) ** -1.3,
                             0.08 ** -0.3 * (0.5 / 0.08) ** -1.3 * (
                                         m / 0.5) ** -2.3))


plt.figure(figsize=(6, 5))
m_array = np.logspace(-2, 2, 400)

labelss = ['Salpeter+55', 'Kroupa', 'Chabrier+03',
           'Chabrier+03, Eq.18']
i=0

for label, imf in zip(''
                      'Salpeter55 '
                      # 'MillerScalo79'
                      ' Kroupa01 '
                      'Chabrier03individual'
                      # ' Chabrier03system '
                      # 'Mine'
                      ''.split(), \
                      [
                          salpeter55,
                       # millerscalo79,
                       kroupa01,
                       chabrier03individual,
                       # chabrier03system,
                       # chabrier03_mine
                       ]):
    plt.plot(m_array, imf(m_array) / imf(1), label=labelss[i])
    i += 1

plt.axvline(0.1)
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.xlim(1e-2, 100)
plt.ylim(1e-3, 1e3)
plt.legend(loc='best',)
plt.xlabel('Mass [Solar mass]')
plt.ylabel(r'Mass Function $\xi(m)\Delta m$')
plt.savefig('imf.pdf', bbox_inches='tight')
plt.savefig('imf.png', bbox_inches='tight', dpi=300)

for i in [-2, -1.5, -1, 0, 1]:
    m_array = np.logspace(i, 2, 400)
    print(i)
    # print()
    print((simpson(
        y=m_array ** 2. * np.log(10) * salpeter55(m_array) / salpeter55(1),
        x=np.log10(m_array))
           /
           simpson(y=m_array ** 2. * np.log(10) * chabrier03individual(
               m_array) / chabrier03individual(1), x=np.log10(m_array))))

    print((simpson(
        y=m_array ** 2. * np.log(10) * salpeter55(m_array) / salpeter55(1),
        x=np.log10(m_array))
           /
           simpson(y=m_array ** 2. * np.log(10) * chabrier03system(
               m_array) / chabrier03system(1), x=np.log10(m_array))))
    print()

    print(simpson(y=m_array ** 2. * np.log(10) * salpeter55(m_array),
                  x=np.log10(m_array)))
    # print(simpson(y=m_array * salpeter55(m_array), x=m_array))
    # print(simpson(y=m_array**2. * np.log(10) * salpeter55(m_array) / salpeter55(1), x=np.log10(m_array)))
    # print(simpson(y=m_array * salpeter55(m_array) / salpeter55(1), x=m_array))
    # print(simpson(y=salpeter55(m_array) / salpeter55(1), x=m_array))
    # print()
    #
    print(simpson(y=m_array ** 2. * np.log(10) * chabrier03individual(
        m_array) / chabrier03individual(1), x=np.log10(m_array)))
    # print(simpson(y=m_array * chabrier03individual(m_array)/chabrier03individual(1), x=m_array))
    # print(simpson(y=chabrier03individual(m_array)/chabrier03individual(1), x=m_array))
    # print()
    #
    print(simpson(y=m_array ** 2. * np.log(10) * chabrier03system(
        m_array) / chabrier03system(1), x=np.log10(m_array)))
    # print(simpson(y=m_array * chabrier03system(m_array)/chabrier03system(1), x=m_array))
    # print(simpson(y=chabrier03system(m_array)/chabrier03system(1), x=m_array))
    # print()
    #
    #
    # print(simpson(y=m_array**2. * np.log(10) * millerscalo79(m_array)/millerscalo79(1), x=np.log10(m_array)))
    # print(simpson(y=m_array * millerscalo79(m_array)/millerscalo79(1), x=m_array))
    # print(simpson(y=millerscalo79(m_array)/millerscalo79(1), x=m_array))
    print()


def imf_salpeter(m):
    return m ** (-2.35)


# Chabrier IMF (2003) piecewise form:
# log-normal below 1 Msun, power law above
m_c = 0.079
sigma = 0.69

m_array = np.logspace(-2, 2, 400)


# def imf_chabrier(m):
#     if m < 1.0:
#         return 0.158 * np.exp(-(np.log10(m) - np.log10(m_c))**2
#                                 / (2*sigma**2))
#     else:
#         return 4.43e-2 * m**(-1.3)

def imf_chabrier(m):
    if m < 1.0:
        # lognormal in ln(m)
        return 0.158 * np.exp(
            -(np.log(m) - np.log(0.079)) ** 2 / (2 * 0.69 ** 2)) \
               / m
    else:
        return 0.0443 * m ** (-2.3)

# plt.show()
# plt.plot(m_array, np.array(list(map(lambda x: imf_salpeter(x), m_array))) /
#          imf_salpeter(1), ls='--',
#          lw=3, alpha=0.7)
# plt.plot(m_array, np.array(list(map(lambda x: imf_chabrier(x), m_array))) /
#          imf_chabrier(1), ls='--',
#          lw=3, alpha=0.7)


# We integrate m * IMF(m) dm to get total stellar mass
def total_mass_salpeter():
    print()
    return quad(lambda m: m * imf_salpeter(m), 0.1, 100)[0]


def total_mass_chabrier():
    return quad(lambda m: m * imf_chabrier(m), 0.1, 100)[0]


Ms = total_mass_salpeter()
Mc = total_mass_chabrier()

print('integrations salpeter and chabrier')
print(Ms)
print(Mc)

conversion_factor = Ms / Mc
print("Salpeter / Chabrier mass ratio =", conversion_factor)


# Normalization: match the IMFs at the high-mass end (1–100 Msun)
def normalize(imf):
    # number of high-mass stars (1–100 Msun)
    Nh = quad(lambda m: imf(m), 1.0, 100.0)[0]
    return 1.0 / Nh


# Normalized IMFs
A_salp = normalize(imf_salpeter)
A_chab = normalize(imf_chabrier)

mmin, mmax = 0.1, 100.0

# Total stellar mass (0.1–100 Msun), after matching high-mass normalization
M_salp = quad(lambda m: m * imf_salpeter(m) * A_salp, mmin, mmax)[0]
M_chab = quad(lambda m: m * imf_chabrier(m) * A_chab, mmin, mmax)[0]

ratio = M_salp / M_chab
print("Mass ratio (Salpeter / Chabrier) =", ratio)


def chabrier(m):
    if m < 1.0:
        # lognormal in ln(m)
        return 0.158 * np.exp(-(np.log10(m)
                                - np.log10(0.079)) ** 2/(2 *0.692)) / m
    else:
        return 0.0443 * m ** (-2.3)


# ----------------------------
# Salpeter IMF
# ----------------------------
def salpeter(m):
    return m ** (-2.35)


# ----------------------------
# Normalize both IMFs to match number of massive stars
# ----------------------------
def normalize(imf):
    Nh = quad(lambda m: imf(m), 1.0, 100.0)[0]
    return 1.0 / Nh


A_s = normalize(salpeter)
A_c = normalize(chabrier)

# ----------------------------
# Integrate total mass
# ----------------------------
M_s = quad(lambda m: m * salpeter(m) * A_s, 0.1, 100)[0]
M_c = quad(lambda m: m * chabrier(m) * A_c, 0.1, 100)[0]

print("Mass ratio (Salpeter / Chabrier) =", M_s / M_c)

# plt.savefig('imf.svg', bbox_inches='tight')
# plt.savefig('imf.png', bbox_inches='tight')



import numpy as np
from scipy.integrate import quad


# ----------------------------
# Correct Chabrier (2003) IMF
# ----------------------------
def chabrier(m):
    if m < 1.0:
        # lognormal in ln(m)
        return 0.158 * np.exp(
            -(np.log(m) - np.log(0.079)) ** 2 / (2 * 0.69 ** 2)) \
               / m
    else:
        return 0.0443 * m ** (-2.3)


# ----------------------------
# Salpeter IMF
# ----------------------------
def salpeter(m):
    return m ** (-2.35)


# ----------------------------
# Normalize both IMFs to match number of massive stars
# ----------------------------
def normalize(imf):
    Nh = quad(lambda m: imf(m), 1.0, 100.0)[0]
    return 1.0 / Nh


A_s = normalize(salpeter)
A_c = normalize(chabrier)

# ----------------------------
# Integrate total mass
# ----------------------------
M_s = quad(lambda m: m * salpeter(m) * A_s, 0.1, 100)[0]
M_c = quad(lambda m: m * chabrier(m) * A_c, 0.1, 100)[0]

print("Mass ratio (Salpeter / Chabrier) =", M_s / M_c)

plt.show()
