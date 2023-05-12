import numpy as np
import matplotlib.pyplot as plt


def sfr_data_dict(ext_path=''):
    md_data = np.loadtxt(ext_path + 'sfr_data/MadauDickinson_uvdata.txt',
                         usecols=range(1, 7))
    z_lo = md_data[:, 0]
    z_up = md_data[:, 1]
    z_uv_dat = 0.5 * (z_up + z_lo)
    z_uv_lo = np.absolute(z_uv_dat - z_lo)
    z_uv_up = np.absolute(z_uv_dat - z_up)
    sfr_uv = 10 ** md_data[:, 3]
    sfr_uv_lo = sfr_uv * np.log(10) * md_data[:, 5]
    sfr_uv_up = sfr_uv * np.log(10) * md_data[:, 4]

    sfr1_table = np.column_stack((z_uv_dat, z_uv_lo, z_uv_up,
                                 sfr_uv, sfr_uv_lo, sfr_uv_up,
                                 np.ones(len(z_uv_dat), dtype=int)))

    md_data = np.loadtxt(ext_path + 'sfr_data/MadauDickinson_irdata.txt',
                         usecols=range(1, 6))

    z_lo = md_data[:, 0]
    z_up = md_data[:, 1]
    z_ir_dat = 0.5 * (z_up + z_lo)
    z_ir_lo = np.absolute(z_ir_dat - z_lo)
    z_ir_up = np.absolute(z_ir_dat - z_up)
    sfr_ir = 10 ** md_data[:, 2]
    sfr_ir_lo = sfr_ir * np.log(10) * md_data[:, 3]
    sfr_ir_up = sfr_ir * np.log(10) * md_data[:, 4]

    sfr2_table = np.column_stack((z_ir_dat, z_ir_lo, z_ir_up,
                                 sfr_ir, sfr_ir_lo, sfr_ir_up,
                                 2 * np.ones(len(z_ir_dat), dtype=int)))

    dr_data = np.loadtxt(ext_path + 'sfr_data/Driver_SFR.txt')
    z_lo = dr_data[:, 1]
    z_up = dr_data[:, 2]
    z_dr = 0.5 * (z_up + z_lo)
    z_dr = 10 ** (0.5 * (np.log10(z_up) + np.log10(z_lo)))
    z_dr_lo = np.absolute(z_dr - z_lo)
    z_dr_up = np.absolute(z_dr - z_up)
    sfr_dr = 10 ** dr_data[:, 3] / 0.63
    sfr_dr_lo = sfr_dr * np.log(10) * np.sum(dr_data[:, 5:], axis=1)
    sfr_dr_up = sfr_dr_lo

    sfr3_table = np.column_stack((z_dr, z_dr_lo, z_dr_up,
                                 sfr_dr, sfr_dr_lo, sfr_dr_up,
                                 3 * np.ones(len(z_dr), dtype=int)))

    bo_data = np.loadtxt(ext_path + 'sfr_data/Bourne2017.txt')
    z_bo_lo = bo_data[:, 2]
    z_bo_up = bo_data[:, 3]
    z_bo = bo_data[:, 0]
    sfr_bo = bo_data[:, 1] / 0.63
    sfr_bo_lo = bo_data[:, 4]
    sfr_bo_up = bo_data[:, 5]

    sfr4_table = np.column_stack((z_bo, z_bo_lo, z_bo_up,
                                 sfr_bo, sfr_bo_lo, sfr_bo_up,
                                 4 * np.ones(len(z_bo), dtype=int)))

    z_bouw = np.array([3.8, 4.9, 5.9])
    sfr_bouw = 10 ** (-np.array([1.0, 1.26, 1.55])) * (1.15 / 1.4)
    dsfr_bouw = np.array([0.06, 0.06, 0.06])
    dsfr_bouw = sfr_bouw * np.log(10) * dsfr_bouw

    sfr5_table = np.column_stack((z_bouw, np.zeros(len(z_bouw)),
                                  np.zeros(len(z_bouw)),
                                 sfr_bouw, dsfr_bouw, dsfr_bouw,
                                 5 * np.ones(len(z_bouw), dtype=int)))

    return np.concatenate((sfr1_table, sfr2_table, sfr3_table,
                           sfr4_table, sfr5_table))


def plot_sfr_data(dict_srd):

    refs = np.unique(dict_srd[:, -1])
    ref_names = [r'Madau $&$ Dickinson UV data',
                 r'Madau $&$ Dickinson IR data',
                 r'Driver et al. 2018',
                 'Bourne et al. 2017',
                 'Bouwens et al. 2015']

    markers = ['*', '<', '>', 'H', '^', 'd']

    for nref, ref in enumerate(refs):
        dict_values = dict_srd[dict_srd[:, -1] == ref, :]

        plt.errorbar(x=dict_values[:, 0], xerr=(dict_values[:, 1],
                                                dict_values[:, 2]),
                     y=dict_values[:, 3], yerr=(dict_values[:, 4],
                                                dict_values[:, 5]),
                     label=ref_names[int(ref)-1],
                     linestyle='', marker=markers[nref % len(markers)])
