""" Compute lambda for single low rank factorization method of Berry, et al """
from typing import Tuple
import numpy as np
import h5py


def compute_lambda(cholesky_dim: int, integral_path: str, verify_eri: bool = False)\
    -> Tuple[int, float]:
    """ Compute lambda for Hamiltonian using SF method of Berry, et al.

    Args:
        cholesky_dim (int) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        integral_path (str) - path to file which integrals to use; assumes hdf5 with 'h0' and 'eri'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orb (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """

    with h5py.File(integral_path, "r") as f:
        eri = np.asarray(f['eri'][()])
        h0  = np.asarray(f['h0'][()])

    n_orb = len(h0)  # number orbitals
    # Check dims are consistent
    assert [n_orb] * 4 == [*eri.shape]

    # Cholesky factored ints do not exist, so create them
    cholesky_diagonals, cholesky_lower_tri = np.linalg.eigh(eri.reshape(n_orb **2, n_orb **2))

    # Put in descending order
    cholesky_diagonals = cholesky_diagonals[::-1]
    cholesky_lower_tri = cholesky_lower_tri[:,::-1]

    # Truncate
    # FIXME: add user-defined threshold (usually 1E-8 is more than enough)
    idx = np.where(cholesky_diagonals > 1.15E-16)[0]
    cholesky_diagonals, cholesky_lower_tri = cholesky_diagonals[idx], cholesky_lower_tri[:,idx]

    #FIXME: Add option to save and read rank-reduced integrals

    # eliminate diagonals D from Cholesky decomposition LDL^T
    L = np.einsum("ij,j->ij",cholesky_lower_tri,
        np.sqrt(cholesky_diagonals))

    if verify_eri:
        # Make sure we are reading in the integrals correctly ... don't check for large cases (!)
        eri_new = np.einsum('ik,kj->ij',L,L.T,optimize=True)
        assert np.allclose(eri_new.flatten(),eri.flatten())

    # Reshape for lambda calcs
    L = L.reshape(n_orb, n_orb, -1)

    # Effective one electron operator contribution
    T = h0 - 0.5 * np.einsum("pqqs->ps", eri, optimize=True) +\
        np.einsum("pqrr->pq", eri, optimize = True)

    lambda_T = np.sum(np.abs(T))

    # Do rank-reduction of ERIs using cholesky_dim vectors
    LR = L[:,:,:cholesky_dim]

    # Two electron operator contributions
    lambda_W = 0.25 * np.einsum("ijP,klP->",np.abs(LR), np.abs(LR), optimize=True)
    lambda_tot = lambda_T + lambda_W

    return n_orb * 2, lambda_tot  #  return number spin orbitals from spatial and combined lambda


if __name__ == '__main__':

    CHOL_DIM = 200
    NAME = '../integrals/eri_reiher.h5'
    VERIFY=True
    number_orbitals, total_lambda = compute_lambda(cholesky_dim=CHOL_DIM,
        integral_path=NAME,verify_eri=VERIFY)
    print(number_orbitals, total_lambda, CHOL_DIM)
    assert number_orbitals == 108
    assert np.isclose(total_lambda,4258.0)
