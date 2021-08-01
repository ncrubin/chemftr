""" Compute lambda for double low rank factorization method of von Burg, et al """
from typing import Tuple
import numpy as np
import h5py
from chemftr.rank_reduce import modified_cholesky, eigendecomp
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

    h1, eri, _, _ = read_cas(integral_path, num_alpha=-1, num_beta=-1)

    # rank-reduced ints do not exist, so create them
    n_orb = h1.shape[0]  # number of orbitals

    # rank-reduced ints do not exist, so create them

    if reduction == 'cholesky':
        # FIXME: Is this correct? I get different lambda than with eigendecomp
        L = modified_cholesky(eri.reshape(n_orb**2, n_orb**2),tol=1e-12,verbose=False)

    elif reduction == 'eigendecomp':
        L = eigendecomp(eri.reshape(n_orb**2, n_orb**2),tol=1e-12)

    #FIXME: Add option to save and read rank-reduced integrals

    if verify_eri:
        # Make sure we are reading in the integrals correctly ... don't check for large cases (!)
        eri_new = np.einsum('ik,kj->ij',L,L.T,optimize=True)
        assert np.allclose(eri_new.flatten(),eri.flatten())

    # Reshape for lambda calcs
    L = L.reshape(n_orb, n_orb, -1)

    nchol_max = max(L.shape)

    T = h1 - 0.5 * np.einsum("illj->ij", eri) + np.einsum("llij->ij", eri)
    e, v = np.linalg.eigh(T)
    lambda_T = np.sum(np.abs(e))

    # double factorized Hamiltonian
    H_df = np.zeros_like(eri)

    lambda_F = 0.0

    M = 0 # rolling number of eigenvectors
    for R in range(nchol_max):
        Lij = L[:,:, R]
        e, v = np.linalg.eigh(Lij)
        normSC = np.sum(np.abs(e))

        truncation = normSC * np.abs(e)

        idx = truncation > thresh
        plus  = np.sum(idx)
        M += plus

        if plus == 0:
            #print ("{} out of {}".format(R, nchol_max))
            break

        e_selected = np.diag(e[idx])
        v_selected = v[:,idx]

        Lij_selected = v_selected.dot(e_selected).dot(v_selected.T)

        H_df += np.einsum("ij,kl->ijkl", Lij_selected, Lij_selected, optimize=True)

        normSC = 0.25 * np.sum(np.abs(e_selected))**2
        lambda_F += normSC

    # incoherent error
    # ein = np.sqrt(np.sum(np.abs(eri - H_df)**2))

    lambda_tot = lambda_T + lambda_F

    #print(thresh, M, ein, lambda_T, lambda_F, lambda_tot)

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
