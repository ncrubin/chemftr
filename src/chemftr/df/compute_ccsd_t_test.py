"""Test cases for compute_ccsd_t.py
"""
import numpy as np
from os import path
from chemftr import df
from chemftr.molecule import load_casfile_to_pyscf

def test_reiher_df_ccsd_t():
    """ Reproduce Reiher et al orbital DF lambda from paper """

    NAME     = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    mol, mf = load_casfile_to_pyscf(NAME, num_alpha = 27, num_beta = 27)
    VERIFY   = True
    escf, ecorr, etot = df.compute_ccsd_t(mf, thresh = 0.0, verify_eri=VERIFY)
    exact_energy = ecorr
    appx_energy = []
    # FIXME: can reduce the time of test by testing just one rank-reduction
    # I'm happy to keep it a bit more rigorous for now 
    thresholds = [0.1, 0.00125, 0.00005]
    for thresh in thresholds:
        escf, ecorr, etot = df.compute_ccsd_t(mf, thresh, verify_eri=VERIFY)
        appx_energy.append(ecorr)

    appx_energy = np.asarray(appx_energy)
    error = (appx_energy - exact_energy)*1E3  # mEh

    assert np.allclose(np.round(error,decimals=2),[-87.91,0.44,0.00])
