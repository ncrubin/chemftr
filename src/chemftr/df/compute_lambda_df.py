""" Compute lambda for double low rank factorization method of von Burg, et al """
from typing import Tuple
import numpy as np
from chemftr.rank_reduce import double_factorize
from chemftr.util import read_cas


def compute_lambda(thresh: float, integral_path: str, reduction: str = 'eigendecomp', \
    verify_eri: bool = False) -> Tuple[int, float]:
    """ Compute lambda for Hamiltonian using DF method of von Burg, et al.

    Args:
        thresh (float) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        integral_path (str) - path to file which integrals to use; assumes hdf5 with 'h0' and 'eri'
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orbitals (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the double factorized Hamiltonian
        L (int) - the rank of the first decomposition
        Lxi (int) - the total number of eigenvectors
    """

    h1, eri_full, _, _ = read_cas(integral_path, num_alpha=-1, num_beta=-1)

    # rank-reduced ints do not exist, so create them
    n_orb = h1.shape[0]  # number of orbitals

    _, lambda_F, R, M = double_factorize(eri_full, thresh, reduction, verify_eri)

    T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum("llij->ij", eri_full)
    e, _ = np.linalg.eigh(T)
    lambda_T = np.sum(np.abs(e))

    lambda_tot = lambda_T + lambda_F

    return n_orb * 2, lambda_tot, R, M  # return spinorbital number from spatial

if __name__ == '__main__':

    THRESH = 0.00125
    NAME = '../integrals/eri_reiher.h5'
    VERIFY=True
    n_orbital, total_lambda, rank, num_eigen = compute_lambda(thresh=THRESH,
        integral_path=NAME,verify_eri=VERIFY)
    assert n_orbital == 108
    assert rank == 360
    assert num_eigen == 13031
    assert np.isclose(np.round(total_lambda,decimals=1),294.8)
