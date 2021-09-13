""" Compute lambda for single low rank factorization method of Berry, et al """
from typing import Tuple
import numpy as np
from chemftr.rank_reduce import single_factorize
from chemftr.molecule import pyscf_to_cas


def compute_lambda(pyscf_mf, cholesky_dim: int, reduction: str = 'eigendecomp', \
    verify_eri: bool = False) -> Tuple[int, float]:
    """ Compute lambda for Hamiltonian using SF method of Berry, et al.

    Args:
        pyscf_mf - PySCF mean field object
        cholesky_dim (int) - dimension to retain in Cholesky for low-rank reconstruction of ERIs

    Kwargs:
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orb (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
        eri_rr (ndarray) - rank-reduced ERIs from the single factorization algorithm
    """

    # grab tensors from pyscf_mf object
    h1, eri_full, _, _, _ = pyscf_to_cas(pyscf_mf)

    # compute the rank-reduced eri tensors (LR.LR^T ~= ERI)
    eri_rr, LR = single_factorize(eri_full, cholesky_dim, reduction, verify_eri)

    # Effective one electron operator contribution
    T = h1 - 0.5 * np.einsum("pqqs->ps", eri_full, optimize=True) +\
        np.einsum("pqrr->pq", eri_full, optimize = True)

    lambda_T = np.sum(np.abs(T))

    # Two electron operator contributions
    lambda_W = 0.25 * np.einsum("ijP,klP->",np.abs(LR), np.abs(LR), optimize=True)
    lambda_tot = lambda_T + lambda_W

    return h1.shape[0] * 2, lambda_tot, eri_rr  


if __name__ == '__main__':
    from chemftr.molecule import load_casfile_to_pyscf

    CHOL_DIM = 200
    mol, mf = load_casfile_to_pyscf('../integrals/eri_reiher.h5',num_alpha = 27, num_beta = 27)
    VERIFY=True
    number_orbitals, total_lambda, _  = compute_lambda(mf, cholesky_dim=CHOL_DIM, verify_eri=VERIFY)
    print(number_orbitals, total_lambda, CHOL_DIM)
    assert number_orbitals == 108
    assert np.isclose(total_lambda,4258.0)
