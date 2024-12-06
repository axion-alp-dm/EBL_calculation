import numpy as np
from numpy.testing import assert_allclose
import pytest

import ebl_codes.sfr_models as sfr

def test_madau14():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr.madau14(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.01496149, 0.03173497, 0.00556202],
                    atol=1e-6)


def sfr_finke22a():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr.madau14(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.00912011, 0.02203551, 0.0015306],
                    atol=1e-6)


def sfr_cuba():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr.madau14(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.0069, 0.03602158, 0.00632803],
                    atol=1e-6)


def sfr_model():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr.madau14(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.01496149, 0.03173497, 0.00556202],
                    atol=1e-6)

