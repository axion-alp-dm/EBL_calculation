import numpy as np


def calculate_dust(model_name, freq_array):

    if model_name == 'att_kn2002':
        return att_kn2002(freq_array)

    else:
        no_abs()
        return np.zeros(len(freq_array))


def no_abs():
    print('   -> No dust absorption definition matched the name given.')
    return


def att_kn2002(fr):
    return -.4 * .68 * 3.2 * (1. / fr - .35)


'''
class DustAbsorption(object):
    def __init__(self, model_name, freq_array):
        if model_name == 'att_kn2002':
            return -.4 * .68 * 3.2 * (1. / freq_array - .35)

        else:
            return self.no_abs()
        print('wwww')
        self.aaaa = 12

    @staticmethod
    def return_result(model_name, freq_array):
        if model_name == 'att_kn2002':
            return DustAbsorption()

        else:
            return self.no_abs()


    def att_kn2002(fr):
        return -.4 * .68 * 3.2 * (1. / fr - .35)

    def no_abs():
        print('No dust absorption definition matched the name given.')
        return
aaa = DustAbsorption('att_kn2002', zzz)
print(DustAbsorption.return_result('att_kn2002', zzz))'''

