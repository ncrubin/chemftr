"""Test cases for compute_ccsd_t.py
"""
import numpy as np
from os import path
from chemftr import sf
from chemftr.molecule import load_casfile_to_pyscf

def test_reiher_sf_ccsd_t():
    """ Reproduce Reiher et al orbital SF lambda from paper """

    NAME     = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    mol, mf = load_casfile_to_pyscf(NAME, num_alpha = 27, num_beta = 27)
    VERIFY   = True
    escf, ecorr, etot = sf.compute_ccsd_t(mf, cholesky_dim=None, verify_eri=VERIFY)
    exact_energy = ecorr
    appx_energy = []
    # FIXME: can reduce the time of test by testing just one rank-reduction
    # I'm happy to keep it a bit more rigorous for now 
    ranks = [100,200,300]
    for CHOL_DIM in ranks:
        escf, ecorr, etot = sf.compute_ccsd_t(mf, cholesky_dim=CHOL_DIM, verify_eri=VERIFY)
        appx_energy.append(ecorr)

    appx_energy = np.asarray(appx_energy)
    error = (appx_energy - exact_energy)*1E3  # mEh

    assert np.allclose(np.round(error,decimals=2),[1.55,0.10,0.18])
