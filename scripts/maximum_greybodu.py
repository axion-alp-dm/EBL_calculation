import os
import yaml
import numpy as np
from scipy.optimize import newton
from scipy.integrate import simpson
import astropy.constants as c
from astropy import units as u

from ebl_codes.EBL_class import EBL_model


import matplotlib.pyplot as plt

input_file_dir = 'scripts/input_files/'

# Check that the working directory is correct for the paths
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            # print(pa)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


models = ['solid', 'dashed', 'dotted', 'dashdot']
colors = ['b', 'r', 'g', 'grey',
          'purple', 'k', 'cyan', 'brown']

linstyles_ssp = ['solid', '--', 'dotted', '-.', (0, (3, 5, 1, 5, 1, 5))]

markers = ['.', 'x', '+', '*', '^', '>', '<']


# We initialize the class with the input file
config_data = read_config_file(input_file_dir + 'input_manyT.yml')
ebl_class = EBL_model.input_yaml_data_into_class(config_data,
                                                 log_prints=True)

def maximum_T(wv_array, T):
    zz = np.linspace(0, 40, num=500)
    zzp1 = zz + 1.

    zzp1, wv_array = np.meshgrid(zzp1, wv_array)

    all_inside = (c.h * c.c / c.k_B / T / u.K).to(u.micron).value

    exp_full = np.exp(all_inside * zzp1 / wv_array)

    yyy = (zzp1**2. / np.sqrt(0.7 + 0.3 * zzp1**3.))
    yyy *= (all_inside * zzp1 * exp_full
            - 4. * wv_array * (exp_full - 1.))
    yyy /= (exp_full - 1.)**2.

    yyy[np.isnan(yyy)] = 0.

    return simpson(yyy, x=zz, axis=1)

def find_peak(T):
    return newton(maximum_T, x0=200, args=[T])

tt_array = np.linspace(30, 450, num=100)
lamb_max = np.zeros(len(tt_array))
print('60.5  ', find_peak(60.5))
print('70.  ', newton(maximum_T, x0=200, args=[70.]))
print('450.  ', newton(maximum_T, x0=200, args=[450.]))

for ni, ii in enumerate(tt_array):
    lamb_max[ni] = newton(maximum_T, x0=200, args=[ii])

plt.figure()
plt.loglog(tt_array, lamb_max, marker='.')

aaa = np.polyfit(np.log10(tt_array), np.log10(lamb_max), 1)
print(aaa)

plt.xlabel('T of the grey body (K)')
plt.ylabel('Grey Body Peak (microns)')


waves_ebl = np.logspace(-1, 3, num=500)
wv_max = [232.6662911331458, 204.46606665791157, 179.68383907677193,
          163.8433645577983, 146.6670866343967, 136.22867641416454,
          124.21909954526161, 117.52771174629451, 109.16317341936147,
          101.39394576752917, 95.93209477938241, 90.76446072885356,
          85.87519484845171, 81.24930210614048, 78.30465404301182,
          74.0865683493956, 71.40151304013396, 68.81376986415663,
          65.10693555481454, 62.74732129711575, 60.47322449462643,
          58.2815458123084, 56.16929824163804, 54.133603029653756,
          53.14359158053236, 51.21755443364236, 49.36132098237912,
          48.45858644816491, 46.70234388327331, 45.009751296080466,
          44.18659956385942, 43.37850187558989, 41.80637241455758,
          41.04180502904709, 40.29122027951348, 38.83098049059605,
          38.12082795843888, 37.42366290721981, 36.73924781802074,
          35.4077390896527, 34.76019181541977, 34.124487078528894,
          33.50040829913133, 32.8877428582551, 32.286282025367306,
          31.695820887261153, 31.1161582782436, 30.547096711599682,
          29.988442312310276, 29.440004751000338, 28.901597179095088,
          28.373036165162116, 27.85414163241766, 27.3447367973758,
          26.844648109619627, 26.35370519267393, 25.87174078595921,
          25.39859068780724, 25.39859068780724, 24.93409369951878,
          24.478091570444402, 24.03042894406967, 23.590953305086337,
          23.15951492743152, 23.15951492743152, 22.735966823277153,
          22.320164692952325, 22.320164692952325, 21.911966875781474,
          21.511234301821652, 21.117830444482422, 21.117830444482422,
          20.73162127401229, 20.352475211835756, 20.352475211835756,
          19.980263085725486, 19.614858085794292, 19.614858085794292,
          19.256135721291965, 19.256135721291965, 18.903973778192196,
          18.55825227755517, 18.55825227755517, 18.218853434651653,
          17.885661618834618, 17.885661618834618, 17.55856331414469,
          17.55856331414469, 17.237447080636198, 17.237447080636198,
          16.922203516410374, 16.61272522034293, 16.61272522034293,
          16.308906755493325, 16.308906755493325, 16.010644613183178,
          16.010644613183178, 15.717837177731628, 15.717837177731628,
          15.430384691835645]

ebl_class.logging_prints = False
tt_array = np.geomspace(30, 450, num=50)
wv_max = []
for ti in tt_array:
    print()
    config_data['ssp_models']['SB99_Finke_bosa'][
        'dust_reem_params']['T'] = ti
    ebl_class.ebl_ssp_calculation(
        config_data['ssp_models']['SB99_Finke_bosa'])

    ebl_yyy = ebl_class.ebl_ssp_spline(waves_ebl, 0.)

    wv_max.append(waves_ebl[np.argmax(ebl_yyy)])
print(wv_max)
print(np.polyfit(np.log10(tt_array), np.log10(wv_max), 1))


plt.figure()
plt.loglog(tt_array, wv_max)
print(np.polyfit(np.log10(tt_array), np.log10(wv_max), 1))

plt.figure()
tt_array = np.linspace(30, 450, num=5)

for nn, ti in enumerate(tt_array):
    print()
    cc = plt.cm.CMRmap(nn / float(len(tt_array)))
    config_data['ssp_models']['SB99_Finke_bosa'][
        'dust_reem_params']['T'] = ti
    ebl_class.ebl_ssp_calculation(
        config_data['ssp_models']['SB99_Finke_bosa'])

    ebl_yyy = ebl_class.ebl_ssp_spline(waves_ebl, 0.)

    plt.loglog(waves_ebl, ebl_yyy, c=cc)
    plt.axvline(waves_ebl[np.argmax(ebl_yyy)], c=cc)


plt.show()
