""" Compute lambda for double low rank factorization method of von Burg, et al """
from typing import Tuple
import numpy as np
from chemftr.rank_reduce import double_factorize
from chemftr.molecule import pyscf_to_cas


def compute_lambda(pyscf_mf, thresh: float, reduction: str = 'eigendecomp', \
    verify_eri: bool = False) -> Tuple[int, float]:
    """ Compute lambda for Hamiltonian using DF method of von Burg, et al.

    Args:
        pyscf_mf - Pyscf mean field object
        thresh (float) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orbitals (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the double factorized Hamiltonian
        L (int) - the rank of the first decomposition
        Lxi (int) - the total number of eigenvectors
    """

    # grab tensors from pyscf_mf object
    h1, eri_full, _, _, _ = pyscf_to_cas(pyscf_mf)

    # rank-reduced ints do not exist, so create them
    n_orb = h1.shape[0]  # number of orbitals

    _, lambda_F, R, M = double_factorize(eri_full, thresh, reduction, verify_eri)

    T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum("llij->ij", eri_full)
    e, _ = np.linalg.eigh(T)
    lambda_T = np.sum(np.abs(e))

    lambda_tot = lambda_T + lambda_F

    return n_orb * 2, lambda_tot, R, M  # return spinorbital number from spatial

if __name__ == '__main__':

    from chemftr.molecule import load_casfile_to_pyscf

    THRESH = 0.00125
    NAME = '../integrals/eri_reiher.h5'
    mol, mf = load_casfile_to_pyscf(NAME, num_alpha = 27, num_beta = 27)
    VERIFY=True
    n_orbital, total_lambda, rank, num_eigen = compute_lambda(mf, thresh=THRESH, verify_eri=VERIFY)
    assert n_orbital == 108
    assert rank == 360
    assert num_eigen == 13031
    assert np.isclose(np.round(total_lambda,decimals=1),294.8)
