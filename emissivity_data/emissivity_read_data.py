import numpy as np


def emissivity_data():
    data = np.loadtxt('emissivity_data/finke22.txt', usecols=(0, 1, 2, 3, 4))
    return data
