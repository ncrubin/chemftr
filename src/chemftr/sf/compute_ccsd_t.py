""" Compute CCSD(T) for single low rank factorization method of Berry, et al """
from typing import Tuple
import numpy as np
from chemftr.rank_reduce import single_factorize
from chemftr.util import read_cas, ccsd_t
from chemftr.molecule import pyscf_to_cas

def compute_ccsd_t(pyscf_mf, cholesky_dim: int, num_alpha = None, num_beta = None, \
    reduction: str = 'eigendecomp', verify_eri: bool = False, use_kernel = True) -> Tuple[int, float]:
    """ Compute CCSD(T) energy for Hamiltonian using SF method of Berry, et al.

    Args:
        pyscf_mf - PySCF mean field object
        cholesky_dim (int) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        e_scf (float) - SCF energy
        e_cor (float) - Correlation energy from CCSD(T)
        e_tot (float) - Total energy; i.e. SCF energy + Correlation energy from CCSD(T)
    """
    h1, eri_full, ecore, num_alpha, num_beta = pyscf_to_cas(pyscf_mf, num_alpha, num_beta)

    # compute the rank-reduced eri tensors (LR.LR^T = eri_rr ~= eri_full)
    eri_rr, _ = single_factorize(eri_full, cholesky_dim, reduction, verify_eri)

    e_scf, e_cor, e_tot = ccsd_t(h1, eri_rr, ecore, num_alpha, num_beta, eri_full, use_kernel)

    return e_scf, e_cor, e_tot


if __name__ == '__main__':

    from chemftr.molecule import load_casfile_to_pyscf
    NAME = '../integrals/eri_reiher.h5'
    mol, mf = load_casfile_to_pyscf(NAME, num_alpha = 27, num_beta = 27)
    VERIFY=True
    escf, ecorr, etot = compute_ccsd_t(mf, cholesky_dim=None, verify_eri=VERIFY)
    exact_energy = ecorr
    appx_energy = []
    ranks = [100]
    for CHOL_DIM in ranks:
        escf, ecorr, etot = compute_ccsd_t(mf, cholesky_dim=CHOL_DIM, verify_eri=VERIFY)
        appx_energy.append(ecorr)

    appx_energy = np.asarray(appx_energy)
    error = (appx_energy - exact_energy)*1E3  # mEh

    for i,rank in enumerate(ranks):
        print('{:>12}  {:>12.6f}  {:>12.2f}'.format(rank, appx_energy[i], error[i]))
