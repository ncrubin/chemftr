"""Test cases for compute_lambda_sf.py
"""
import numpy as np
from os import path
from chemftr import sf


def test_reiher_sf_lambda():
    """ Reproduce Reiher et al orbital SF lambda from paper """

    CHOL_DIM = 200

    NAME     = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    VERIFY   = True
    n_orb, lambda_tot = sf.compute_lambda(cholesky_dim=CHOL_DIM,
        integral_path=NAME,verify_eri=VERIFY)
    assert n_orb == 108
    assert np.isclose(lambda_tot,4258.0)
