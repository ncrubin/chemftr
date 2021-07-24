"""Test cases for compute_lambda_sf.py
"""
import numpy as np
from chemftr.sf.compute_lambda_sf import single_factor_lambda


def test_reiher_sf_lambda():
    """ Reproduce Reiher et al orbital SF lambda from paper """

    CHOL_DIM = 200
    NAME     = 'reiher'
    VERIFY   = True
    lambda_tot = single_factor_lambda(cholesky_dim=CHOL_DIM,integral_name=NAME,verify_eri=VERIFY)
    assert np.isclose(lambda_tot,4258.0)
