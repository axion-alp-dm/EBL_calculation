import yaml
import numpy as np
import matplotlib.pyplot as plt


def plot_ebl_measurement_collection(file, color='.75', cm=plt.cm.rainbow, yearmin=None, yearmax=None, nolabel=False): # Dark2

    with open(file, 'r') as stream:
        try:
            ebl_m = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ebl_m = sorted(ebl_m.items(), key=lambda t: t[1]['year'])

    if yearmin or yearmax:
        ebl_m_n = []
        for m in ebl_m:
            if yearmin and m['year'] >= yearmin:
                ebl_m_n.append(m)
            if yearmax and m['year'] < yearmax:
                ebl_m_n.append(m)
        ebl_m = ebl_m_n

    markers = ['*', '<', '>', 'H', '^', 'd', 'h', 'o', 'p', 's', 'v']

    for mi, m in enumerate(ebl_m):
        yerr = [m[1]['data']['ebl_err_low'], m[1]['data']['ebl_err_high']]

        if m[1]['is_upper_limit'] or m[1]['is_lower_limit']:
            yerr = [10. ** (np.log10(x) + 0.14) - x for x in m[1]['data']['ebl']]
            if m[1]['is_upper_limit'] :
                yerr = [yerr, list(map(lambda x: 0., range(len(yerr))))]
            else:
                yerr = [list(map(lambda x: 0., range(len(yerr)))), yerr]
        if cm:
            color = cm(float(mi) / (len(ebl_m) - 1.))
        label = m[1]['full_name']
        if nolabel:
            label = ""
        plt.errorbar(x=m[1]['data']['lambda'], y=m[1]['data']['ebl'],
                     yerr=yerr, label=label,
                     marker=markers[mi % len(markers)], color=color, mec=color,
                     linestyle='None', lolims=m[1]['is_lower_limit'], uplims=m[1]['is_upper_limit'])

