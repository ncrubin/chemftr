"""Test cases for rank reduced CCSD(T) using DF algorithm 
"""
import numpy as np
from os import path
from chemftr import df
from chemftr.molecule import load_casfile_to_pyscf, rank_reduced_ccsd_t

def test_reiher_df_ccsd_t():
    """ Reproduce Reiher et al orbital DF lambda from paper """

    NAME     = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    mol, mf = load_casfile_to_pyscf(NAME, num_alpha = 27, num_beta = 27)
    VERIFY   = True
    escf, ecorr, etot = rank_reduced_ccsd_t(mf, eri_rr = None)  # use full (local) ERIs for 2-body
    exact_energy = ecorr
    appx_energy = []
    # FIXME: can reduce the time of test by testing just one rank-reduction
    # I'm happy to keep it a bit more rigorous for now 
    thresholds = [0.1, 0.00125, 0.00005]
    for THRESH in thresholds:
        eri_rr, _, _, _ = df.rank_reduce(mf._eri, thresh=THRESH, verify_eri=VERIFY)
        escf, ecorr, etot = rank_reduced_ccsd_t(mf, eri_rr)
        appx_energy.append(ecorr)

    appx_energy = np.asarray(appx_energy)
    error = (appx_energy - exact_energy)*1E3  # mEh

    assert np.allclose(np.round(error,decimals=2),[-87.91,0.44,0.00])
