# IMPORTS --------------------------------------------#
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset

all_size = 18
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
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

direct_name = 'outputs/test'

params = np.load(direct_name + '/axion_params.npy')
axion_mac2 = params[:, 0]
axion_gay = params[:, 1]

fig_params, ax1 = plt.subplots(figsize=(18, 10))

plt.xscale('log')
plt.yscale('log')

# NEW INSET FIGURE
ax2 = plt.axes([0, 0, 1, 1])

posx = 0.5
posy = 0.62
ip = InsetPosition(ax1, [posx, posy, 1-posx, 1-posy])
ax2.set_axes_locator(ip)
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

plt.xscale('log')
plt.yscale('log')

list_models = np.load(direct_name + '/list_models.npy')
print(list_models)

list_colors = []

for ni, model in enumerate(list_models[:, 0]):
    values_gay_array = np.load(direct_name + '/' +
                               str(model) + '_params_UL.npy')
    values_gay_array_NH = np.load(direct_name + '/' +
                                  str(model) + '_params_measur.npy')

    color_i = next(ax1._get_lines.prop_cycler)['color']
    list_colors.append(color_i)

    ax1.contour(axion_mac2, axion_gay,
                (values_gay_array.T - np.min(values_gay_array)),
                levels=[4.61], origin='lower',
                colors=color_i, alpha=0.7, linewidths=5)

    ax2.contour(axion_mac2, axion_gay,
                (values_gay_array.T - np.min(values_gay_array)),
                levels=[4.61], origin='lower',
                colors=color_i, alpha=0.7, linewidths=5)
    # bbb = ax1.pcolor(axion_mac2, axion_gay,
    #                  (values_gay_array.T - np.min(values_gay_array)),
    #                  vmin=0., vmax=10.,  rasterized=True,
    #                  cmap='Oranges', shading='auto'
    #                  )
    # ax1.clabel(aaa, inline=True, fontsize=16, levels=[4.61],
    #            fmt={4.61: r'95%'})

    # NH datapoint
    values = values_gay_array_NH.T - np.min(values_gay_array_NH)
    alpha_grid = (43. - values) / 43.
    alpha_grid = alpha_grid * 0.7
    alpha_grid = alpha_grid * (values <= 43.)

    ax1.pcolor(axion_mac2, axion_gay, values,
               vmin=0., vmax=100., rasterized=True,
               alpha=alpha_grid, cmap='bone', shading='auto')
    ax1.contour(axion_mac2, axion_gay, values,
                levels=[2.30, 5.99], origin='lower',
                colors=('r', 'cyan'))

    ax2.pcolor(axion_mac2, axion_gay, values,
               vmin=0., vmax=100., rasterized=True,
               alpha=alpha_grid, cmap='bone', shading='auto')
    ax2.contour(axion_mac2, axion_gay, values,
                levels=[2.30, 5.99], origin='lower',
                colors=('r', 'cyan'))

ax2.set_xlim(2, 30)
ax2.set_ylim(2e-11, 5e-10)

# ax2.get_xticklabels()[2].set_color("white")
# ax2.get_yticklabels()[2].set_color("white")
# ax2.get_yticklabels()[3].set_color("white")
# ax2.get_yticklabels()[4].set_color("white")

ax1.set_xlabel(r'm$_a\,$c$^2$ (eV)')
ax1.set_ylabel(r'$g_{a\gamma}$ (GeV$^{-1}$)')

ax1.set_xlim(4e-1, 1e2)
ax1.set_ylim(1e-12, 1e-7)

ax1.legend(
    [plt.Line2D([], [], linewidth=2, linestyle='-',
                color=list_colors[i]) for i in range(len(list_models))],
    [list_models[i, 1] for i in range(len(list_models))], loc=3,
    title=r'Models')

plt.savefig(direct_name + '/param_space.png', bbox_inches='tight')
plt.savefig(direct_name + '/param_space.pdf', bbox_inches='tight')

plt.close('all')
# plt.show()
