"""Test cases for compute_lambda_sf.py
"""
import numpy as np
from os import path
from chemftr import sf
from chemftr.molecule import load_casfile_to_pyscf

def test_reiher_sf_lambda():
    """ Reproduce Reiher et al orbital SF lambda from paper """

    CHOL_DIM = 200

    NAME     = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    VERIFY   = True
    _, reiher_mf = load_casfile_to_pyscf(NAME, num_alpha = 27, num_beta = 27) 
    eri_rr, LR = sf.rank_reduce(reiher_mf._eri, cholesky_dim=CHOL_DIM, verify_eri=VERIFY)
    lambda_tot = sf.compute_lambda(reiher_mf, LR)
    assert eri_rr.shape[0] * 2 == 108
    assert np.isclose(lambda_tot,4258.0)
