import numpy as np
import matplotlib.pyplot as plt


def emissivity_data(directory=None, z_min=None, z_max=None, lambda_min=0,
                    lambda_max=5,
                    take1ref=None, plot_fig=False):
    if directory is None:
        directory = 'data/emissivity_measurs/'
    data = np.loadtxt(directory + 'finke22.txt',
                      dtype={'names': ('z', 'lambda', 'eje',
                                       'eje_p', 'eje_n', 'reference'),
                             'formats': (float, float, float,
                                         float, float, 'S1')})
    if z_min is not None:
        data = data[data['z'] >= z_min]
    if z_max is not None:
        data = data[data['z'] <= z_max]

    data = data[(data['lambda'] >= lambda_min)
                * (data['lambda'] <= lambda_max)]

    if take1ref is not None:
        data = data[data['reference'].astype('str') == take1ref]

    if plot_fig is True:
        list_references = np.unique(data['reference'])
        markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

        for nref, ref in enumerate(list_references):
            data_individual = data[data['reference'] == ref]
            plt.errorbar(x=data_individual['z'],
                         y=data_individual['eje'],
                         yerr=(data_individual['eje_n'],
                               data_individual['eje_p']),
                         marker=markers[nref % len(markers)],
                         label=ref,
                         linestyle='', zorder=1e4)

    return data


def plot_emissivities(data, z_min=None, z_max=None,
                      lambda_min=0, lambda_max=5,):

    if z_min is not None:
        data = data[data['z'] >= z_min]
    if z_max is not None:
        data = data[data['z'] <= z_max]

    data = data[(data['lambda'] >= lambda_min)
                * (data['lambda'] <= lambda_max)]

    list_references = np.unique(data['reference'])

    markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

    for nref, ref in enumerate(list_references):
        data_individual = data[data['reference'] == ref]
        plt.errorbar(x=data_individual['z'],
                     y=data_individual['eje'],
                     yerr=(data_individual['eje_n'],
                           data_individual['eje_p']),
                     marker=markers[nref % len(markers)],
                     label=ref,
                     linestyle='', zorder=1e4)

    return


# tabla = emissivity_data(lambda_max=15)
#
# tabla = tabla[tabla['lambda'] == 5.8]
# print(tabla)
#
# plt.figure()
# plot_emissivities(tabla)
#
# plt.xlabel('lambda [microns]')
# plt.ylabel('redshift')
#
# # plt.xscale('log')
# plt.yscale('log')
#
# plt.xlim(0, 10)
# plt.ylim(1e33, 2e35)
#
# plt.legend()
#
# plt.show()
