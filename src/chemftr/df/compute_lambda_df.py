""" Compute lambda for double low rank factorization method of von Burg, et al """
from typing import Tuple
import numpy as np
from chemftr.molecule import pyscf_to_cas


def compute_lambda(pyscf_mf, LR):
    """ Compute lambda for Hamiltonian using DF method of von Burg, et al.

    Args:
        pyscf_mf - Pyscf mean field object
        LR (ndarray) - (N x N x cholesky_dim) array of SF factors from rank reduction of ERI

    Returns:
        n_orb (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
        eri_rr (ndarray) - rank-reduced ERIs from the single factorization algorithm
    """
    # grab tensors from pyscf_mf object
    h1, eri_full, _, _, _ = pyscf_to_cas(pyscf_mf)

    # rank-reduced ints do not exist, so create them
    n_orb = h1.shape[0]  # number of orbitals

    # one body contributions
    T = h1 - 0.5 * np.einsum("illj->ij", eri_full) + np.einsum("llij->ij", eri_full)
    e, _ = np.linalg.eigh(T)
    lambda_T = np.sum(np.abs(e))

    # two body contributions
    lambda_F = 0.0
    for vector in range(LR.shape[2]):
        Lij = LR[:,:,vector]
        e, v = np.linalg.eigh(Lij)
        lambda_F += 0.25 * np.sum(np.abs(e))**2

    lambda_tot = lambda_T + lambda_F

    return lambda_tot
