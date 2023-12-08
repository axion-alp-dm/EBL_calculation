import os
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table, vstack

from scipy.special import ndtr
from scipy.special import erfinv


def import_spectrum_data(parent_dir='',
                         obs_not_taken=None, import_one_type=None,
                         lambda_min=0., lambda_max=1e20):
    if obs_not_taken is None:
        obs_not_taken = []

    nu = []
    nuI_nu = []
    dnuI_nu = [[], []]
    lim = []
    ref = []
    length = []

    with open(parent_dir + 'all_freq/CB_complete.txt', 'r') as filein:
        d = filein.readlines()
        while len(d) != 0:
            aaa = ' '.join(d[0].split())
            length.append(int(d[1].split()[0]))

            for i in range(2, length[-1] + 2):
                ref.append(aaa)
                nu.append(float(d[i].split()[0]))
                nuI_nu.append(float(d[i].split()[1]))
                dnuI_nu[0].append(float(d[i].split()[2]))
                dnuI_nu[1].append(float(d[i].split()[3]))
                lim.append(int(float(d[i].split()[4])))
            d = np.delete(d, range(0, length[-1] + 2))
    nu = np.array(nu)
    nuI_nu = np.array(nuI_nu)
    dnuI_nu[0] = np.array(dnuI_nu[0])
    dnuI_nu[1] = np.array(dnuI_nu[1])
    lim = np.array(lim)

    data = Table(data=(nu, nuI_nu, dnuI_nu[0], dnuI_nu[1], lim, ref),
                 units=(u.Hz, u.W / (u.m ** 2 * u.sr),
                        u.W / (u.m ** 2 * u.sr), u.W / (u.m ** 2 * u.sr),
                        1, 1),
                 names=('lambda', 'nuInu',
                        'nuInu_errn', 'nuInu_errp',
                        'type', 'ref'))

    data['lambda'] = data['lambda'].to(u.um, equivalencies=u.spectral())
    data['nuInu'] = data['nuInu'].to(u.nW / (u.m ** 2 * u.sr))
    data['nuInu_errn'] = data['nuInu_errn'].to(u.nW / (u.m ** 2 * u.sr))
    data['nuInu_errp'] = data['nuInu_errp'].to(u.nW / (u.m ** 2 * u.sr))

    data = data[((data['lambda'] >= lambda_min)
                 * (data['lambda'] <= lambda_max))]

    for i in obs_not_taken:
        data = data[data['ref'] != i]

    if import_one_type == 0:
        data = data[data['type'] == 0]
    if import_one_type == 1:
        data = data[data['type'] == 1]
    if import_one_type == 2:
        data = data[data['type'] == 2]

    return data


def dictionary_datatype(parent_dir, obs_type,
                        lambda_min=0., lambda_max=1e15,
                        plot_measurs=False,
                        obs_not_taken=None):
    if obs_not_taken is None:
        obs_not_taken = []
    list_dirs = os.listdir(parent_dir)
    list_dirs.sort()

    lambdas = np.array([])
    nuInu = np.array([])
    nuInu_errn = np.array([])
    nuInu_errp = np.array([])
    type = np.array([], dtype=int)
    ref = np.array([])

    for directory in list_dirs:

        list_files = os.listdir(parent_dir + '/' + directory)
        list_files.sort()

        for ni, name in enumerate(list_files):

            data = Table.read(parent_dir + '/' + directory + '/' + name,
                              format='ascii.ecsv')

            if data.meta['observable_type'] != obs_type:
                continue

            if name in str(obs_not_taken):
                continue

            x_data = data.colnames[0]

            # Change of units to our standard
            if data['nuInu'].unit.is_equivalent(u.Jy / u.sr):
                data['nuInu'] = (
                        data['nuInu'].to(
                            u.W / u.m ** 2 / u.Hz / u.sr)
                        * data[x_data].to(u.Hz,
                                          equivalencies=u.spectral()))
                data['nuInu_errn'] = (
                        data['nuInu_errn'].to(
                            u.W / u.m ** 2 / u.Hz / u.sr)
                        * data[x_data].to(u.Hz,
                                          equivalencies=u.spectral()))
                data['nuInu_errp'] = (
                        data['nuInu_errp'].to(
                            u.W / u.m ** 2 / u.Hz / u.sr)
                        * data[x_data].to(u.Hz,
                                          equivalencies=u.spectral()))

            data[x_data] = data[x_data].to(u.um,
                                           equivalencies=u.spectral())
            data['nuInu'] = data['nuInu'].to(
                u.nW / u.m ** 2 / u.sr)
            data['nuInu_errn'] = data['nuInu_errn'].to(
                u.nW / u.m ** 2 / u.sr)
            data['nuInu_errp'] = data['nuInu_errp'].to(
                u.nW / u.m ** 2 / u.sr)

            lambda_accepted = (
                    (data[x_data] >= lambda_min)
                    * (data[x_data] <= lambda_max))

            # If there is no datapoint on the accepted wavelengths,
            # we don't use anything from this file and continue
            if sum(lambda_accepted) == 0:
                continue

            lambdas = np.append(lambdas,
                                data[x_data][lambda_accepted])
            nuInu = np.append(nuInu,
                              data['nuInu'][lambda_accepted])
            nuInu_errn = np.append(nuInu_errn,
                                   data['nuInu_errn'][lambda_accepted])
            nuInu_errp = np.append(nuInu_errp,
                                   data['nuInu_errp'][lambda_accepted])
            type = np.append(
                type,
                [int((obs_type == 'IGL') * 3 + (obs_type == 'UL') * 1)
                 for i in range(len(data['nuInu'][lambda_accepted]))])
            ref = np.append(
                ref, [data.meta['label'] for i in range(
                    len(data['nuInu'][lambda_accepted]))])

    t = Table(np.column_stack((lambdas, nuInu, nuInu_errn, nuInu_errp,
                               type, ref)),
              names=('lambda', 'nuInu', 'nuInu_errn', 'nuInu_errp',
                     'type', 'ref'),
              units=(u.um, u.nW / u.m ** 2 / u.sr,
                     u.nW / u.m ** 2 / u.sr, u.nW / u.m ** 2 / u.sr,
                     None, None),
              dtype=(np.float64, np.float64, np.float64, np.float64,
                     np.int, str))
    return t


def import_cb_data(lambda_min_total=0., lambda_max_total=5,
                   plot_measurs=False, ax1=None):
    # Datapoints we use in the study
    upper_lims_cob = dictionary_datatype(
        'ebl_measurements/optical_data_2023', obs_type='UL',
        lambda_max=5.,
        plot_measurs=False)

    upperlims_cub = import_spectrum_data(
        parent_dir='ebl_measurements/',
        obs_not_taken=[
            '$\mathrm{COBE \ (Arendt \ & \ Dwek \ 2003)}$',
            '$\mathrm{Voyager \ I/II \ (Edelstein \ et \ al. \ 2000)}$',
            '$\mathrm{COBE \ (Sano \ et \ al. \ 2015)}$',
            '$\mathrm{COBE \ (Sano \ et \ al. \ 2016)}$',
            '$\mathrm{HST \ (Bernstein \ 2007)}$',
            '$\mathrm{HST \ (Kawara \ et \ al. \ 2017)}$',
            '$\mathrm{UVX \ (Martin \ et \ al. \ 1991)}$',
            '$\mathrm{UVX \ (Murthy \ et \ al. \ 1989)}$',
            '$\mathrm{UVX \ (Murthy \ et \ al. \ 1990)}$',
            '$\mathrm{CIBER \ (Matsuura \ et \ al. \ 2017)}$',
            '$\mathrm{AKARI \ (Tsumura \ et \ al. \ 2013)}$',
            '$\mathrm{IRTS \ (Matsumoto \ et \ al. \ 2015)}$',
            '$\mathrm{Pioneer \ 10/11 \ (Matsuoka \ et \ al. \ 2011)}$',
            '$\mathrm{HST \ (Brown \ et \ al. \ 2000)}$'], lambda_max=5.,
        import_one_type=1)

    upper_lims_cxb = import_spectrum_data(
        parent_dir='ebl_measurements/',
        obs_not_taken=[
            '$\mathrm{ASCA \ (Miyaji \ et \ al. \ 1998)}$',
            '$\mathrm{Apollo \ Soyuz \ (Stern \ & \ Bowyer \ 1979)}$',
            '$\mathrm{BeppoSAX \ (Frontera \ et \ al. \ 2007)}$',
            '$\mathrm{Compton \ (Strong \ et \ al. \ 2003)}$',
            '$\mathrm{Compton \ (Weidenspointner \ 2000)}$',
            '$\mathrm{DUVE \ (Korpela \ et \ al. \ 1998)}$',
            '$\mathrm{EUVE \ (Jelinsky \ et \ al. \ 1995)}$',
            '$\mathrm{EUVE \ (Lieu \ et \ al. \ 1993)}$',
            '$\mathrm{Fermi \ (Ackermann \ et \ al. \ 2015)}$',
            '$\mathrm{HEAO \ (Gruber \ et \ al. \ 1999)}$',
            r'$\mathrm{INTEGRAL \ (T\"{u}rler\ et\ al. \ 2010)}$',
            '$\mathrm{ROSAT \ (Miyaji \ et \ al. \ 1998)}$',
            '$\mathrm{SMM \ (Watanabe \ et \ al. \ 2000)}$',
            '$\mathrm{Swift \ (Moretti \ et \ al. \ 2009)}$',
            '$\mathrm{Voyager \ I/II \ (Edelstein \ et \ al. \ 2000)}$',
            '$\mathrm{XMM}$-$\mathrm{Newton \ (De\,Luca \ & \ Molendi \ 2004)}$'],
        lambda_max=0.01)

    def sigma_from_UL(ul_value, sigma):
        return ul_value / erfinv(ndtr(sigma))

    upper_lims_all = vstack([upper_lims_cob, upperlims_cub, upper_lims_cxb])

    upper_lims_all = upper_lims_all[
        ((upper_lims_all['lambda'] >= lambda_min_total)
         * (upper_lims_all['lambda'] <= lambda_max_total))]

    upper_lims_all.add_column(
        (upper_lims_all['nuInu_errn']
         + upper_lims_all['nuInu_errp']) / 2., name='1 sigma', index=4)

    dict_sigmas = {
        '$\mathrm{Dark \ Cloud \ (Mattila et \ al. \ 2012)}$': 2,
        '$\mathrm{Apollo \ Soyuz \ (Stern \ & \ Bowyer \ 1979)}$': 1,
        '$\mathrm{EUVE \ (Lieu \ et \ al. \ 1993)}$': 3,
        '$\mathrm{Voyager \ I/II \ (Murthy \ et \ al. \ 1999)}$': 1,
        '$\mathrm{DUVE \ (Korpela \ et \ al. \ 1998)}$': 2,
        '$\mathrm{EUVE \ (Jelinsky \ et \ al. \ 1995)}$': 3
    }

    upper_lims_all['nuInu_errn'][
        upper_lims_all['ref']
        == '$\mathrm{Voyager \ I/II \ (Murthy \ et \ al. \ 1999)}$'] \
        = upper_lims_all['nuInu'][
        upper_lims_all['ref']
        == '$\mathrm{Voyager \ I/II \ (Murthy \ et \ al. \ 1999)}$']

    for obs in dict_sigmas:
        individual = ((upper_lims_all['nuInu_errp'] == 0)
                      & (upper_lims_all['ref'] == obs))

        upper_lims_all['1 sigma'][individual] = sigma_from_UL(
            upper_lims_all['nuInu'][individual], dict_sigmas[obs])

        upper_lims_all['nuInu'][individual] = 0.

        upper_lims_all['type'][individual] = 0

    upper_lims_all['type'][upper_lims_all['type'] == 2] = 3
    upper_lims_all['type'][upper_lims_all['ref']
                           == r'NH/LORRI (Lauer+ ’22)'] = 2

    upper_lims_all['ref'][upper_lims_all['ref']
                           == r'COBE/DIRBE (Arendt \& Dwek ‘03)'] = \
    r'COBE/DIRBE (Arendt & Dwek ‘03)'


    order = np.argsort(upper_lims_all['type'])
    upper_lims_all = upper_lims_all[order][:]

    names_all_upper, index = np.unique(upper_lims_all['ref'],
                                       return_index=True)
    names_all_upper = names_all_upper[np.argsort(index)]

    lowerlimits_cub = import_spectrum_data(
        parent_dir='ebl_measurements/',
        obs_not_taken=[
            '$\mathrm{FOCA \ (Milliard \ et \ al. \ 1992)}$',
            '$\mathrm{GALEX \ (Xu \ et \ al. \ 2005)}$'],
        lambda_max=5.,
        import_one_type=0)

    lowerlimits_cob = dictionary_datatype(
        'ebl_measurements/optical_data_2023', 'IGL',
        plot_measurs=False)

    lowerlimits_all = vstack([lowerlimits_cub, lowerlimits_cob])

    lowerlimits_all = lowerlimits_all[
        ((lowerlimits_all['lambda'] >= lambda_min_total)
         * (lowerlimits_all['lambda'] <= lambda_max_total))]

    lowerlimits_all.add_column(
        (lowerlimits_all['nuInu_errn']
         + lowerlimits_all['nuInu_errp']) / 2., name='1 sigma', index=4)

    names_all_lower, index = np.unique(lowerlimits_all['ref'],
                                       return_index=True)
    names_all_lower = names_all_lower[np.argsort(index)]

    if plot_measurs:
        markers = ['>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']
        i = 0

        for ni, name in enumerate(names_all_upper):
            data_total = upper_lims_all[upper_lims_all['ref'] == name]
            type_i = np.unique(data_total['type'])
            color_i = next(ax1._get_lines.prop_cycler)['color']

            for datatype in type_i:
                data = data_total[data_total['type'] == datatype]
                if datatype == 1:
                    ax1.errorbar(x=data['lambda'], y=data['nuInu'],
                                 yerr=[data['nuInu_errn'],
                                       data['nuInu_errp']],
                                 linestyle='', color=color_i,
                                 label=name,
                                 marker=markers[i % len(markers)],
                                 mfc='white'
                                 )

                elif datatype == 3:
                    ax1.errorbar(x=data['lambda'], y=data['nuInu'],
                                 yerr=[data['nuInu_errn'],
                                       data['nuInu_errp']],
                                 linestyle='', color=color_i,
                                 label=name,
                                 marker=markers[i % len(markers)]
                                 )

                elif datatype == 2:
                    ax1.errorbar(x=data['lambda'], y=data['nuInu'],
                                 yerr=[data['nuInu_errn'],
                                       data['nuInu_errp']],
                                 linestyle='', color=color_i,
                                 label=name,
                                 marker='*',
                                 mfc='white',
                                 markersize=16
                                 )

                elif datatype == 0:
                    if type_i.__contains__(1):
                        label_i = ''
                    else:
                        label_i = name
                    ax1.errorbar(x=data['lambda'],
                                 y=data['nuInu_errn'],
                                 yerr=data['nuInu_errn'] * 0.4,
                                 linestyle='', color=color_i,
                                 label=label_i,
                                 marker=markers[i % len(markers)],
                                 mfc='white',
                                 uplims=True
                                 )
            i += 1

        for ni, name in enumerate(names_all_lower):
            data = lowerlimits_all[lowerlimits_all['ref'] == name]
            color_i = next(ax1._get_lines.prop_cycler)['color']

            ax1.errorbar(x=data['lambda'], y=data['nuInu'],
                         yerr=[data['nuInu_errn'], data['nuInu_errp']],
                         linestyle='', color=color_i,
                         label=name,
                         marker=markers[i % len(markers)]
                         )
            i += 1

        ax1.set_xlabel(r'Wavelength ($\mu$m)')
        ax1.set_ylabel(r'$\nu \mathrm{I}_{\nu}$ (nW / m$^2$ / sr)')

        ax1.set_xscale('log')
        ax1.set_yscale('log')

    return upper_lims_all, lowerlimits_all
