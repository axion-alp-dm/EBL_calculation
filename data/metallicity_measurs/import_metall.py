import numpy as np


def import_met_data(z_sun=0.02, ax=None):
    data_array = [[0.55, -0.16, 0.11, 0.14],
                  [1.64, -0.4, 0.09, 0.09],
                  [2.26, -0.56, 0.11, 0.1],
                  [2.70, -0.41, 0.30, 0.21],
                  [3.21, -0.79, 0.12, 0.11],
                  [3.95, -0.82, 0.17, 0.14],
                  [4.54, -1.03, 0.20, 0.12]
                  ]
    data_array = np.array(data_array, dtype=float)
    # data_array[:, 2] = (z_sun * 10 ** (data_array[:, 1] - data_array[:, 2]))
    # data_array[:, 3] = (z_sun * 10 ** (data_array[:, 3] + data_array[:, 1]))
    data_array[:, 2] = (z_sun * (-10 ** (data_array[:, 1] - data_array[:, 2])
                                 + 10 ** data_array[:, 1]))
    data_array[:, 3] = (z_sun * (10 ** (data_array[:, 3] + data_array[:, 1])
                                 - 10 ** data_array[:, 1]))
    data_array[:, 1] = 10 ** data_array[:, 1] * z_sun

    if ax is not None:
        ax.errorbar(x=data_array[:, 0], y=data_array[:, 1],
                    yerr=[data_array[:, 2],
                          data_array[:, 3]],
                    linestyle='', color='k',
                    marker='.'
                    )
    return data_array
