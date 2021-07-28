"""Test cases for compute_lambda_df.py
"""
import numpy as np
from os import path
from chemftr import df


def test_reiher_df_lambda():
    """ Reproduce Reiher et al orbital DF lambda from paper """

    THRESH = 0.00125
    NAME = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    VERIFY=True
    n_orbital, total_lambda, L, Lxi = df.compute_lambda(thresh=THRESH,
        integral_path=NAME,verify_eri=VERIFY)
    assert n_orbital == 108
    assert L == 360
    assert Lxi == 13031
    assert np.isclose(np.round(total_lambda,decimals=1),294.8)
