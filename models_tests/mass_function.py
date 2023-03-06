#matplotlib inline
#config InlineBackend.figure_format = 'retina'
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, camb_path)
sys.path.insert(0, '/home/porrassa/miniconda3/envs/EBL_calculation/bin')
sys.path.insert(0, '/home/porrassa/miniconda3/pkgs')
sys.path.insert(0, '/home/porrassa/miniconda3/lib/python3.9/site-packages')

#sys.path.insert(0, '')
#sys.path.insert(0, '')
#sys.path.insert(0, '')

#sys.path.insert(0, '/home/porrassa/miniconda3/envs/EBL_calculation/share')
#sys.path.insert(0, '/home/porrassa/miniconda3/envs/EBL_calculation/conda-meta')
print(sys.path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

print('aaa')



