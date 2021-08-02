""" Compute lambda for single low rank factorization method of Berry, et al """
from typing import Tuple
import numpy as np
from chemftr.rank_reduce import single_factorize
from chemftr.util import read_cas


def compute_lambda(cholesky_dim: int, integral_path: str, reduction: str = 'eigendecomp', \
    verify_eri: bool = False) -> Tuple[int, float]:
    """ Compute lambda for Hamiltonian using SF method of Berry, et al.

    Args:
        cholesky_dim (int) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        integral_path (str) - path to file which integrals to use; assumes hdf5 with 'h0' and 'eri'
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orb (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """

    # read in integrals, we don't care about num_electrons here so pass in dummy variables
    h1, eri_full, _, _ = read_cas(integral_path, num_alpha=-1, num_beta=-1)

    # compute the rank-reduced eri tensors (LR.LR^T ~= ERI)
    _, LR = single_factorize(eri_full, cholesky_dim, reduction, verify_eri)

    # Effective one electron operator contribution
    T = h1 - 0.5 * np.einsum("pqqs->ps", eri_full, optimize=True) +\
        np.einsum("pqrr->pq", eri_full, optimize = True)

    lambda_T = np.sum(np.abs(T))

    # Two electron operator contributions
    lambda_W = 0.25 * np.einsum("ijP,klP->",np.abs(LR), np.abs(LR), optimize=True)
    lambda_tot = lambda_T + lambda_W

    return h1.shape[0] * 2, lambda_tot  #  return num spin orbitals from spatial and combined lambda


if __name__ == '__main__':

    CHOL_DIM = 200
    NAME = '../integrals/eri_reiher.h5'
    VERIFY=True
    number_orbitals, total_lambda = compute_lambda(cholesky_dim=CHOL_DIM,
        integral_path=NAME,verify_eri=VERIFY)
    print(number_orbitals, total_lambda, CHOL_DIM)
    assert number_orbitals == 108
    assert np.isclose(total_lambda,4258.0)
