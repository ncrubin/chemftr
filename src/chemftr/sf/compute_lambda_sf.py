""" Compute lambda for single low rank factorization method of Berry, et al """
from typing import Tuple
import os
import numpy as np
import h5py


def compute_lambda(cholesky_dim: int, integral_name: str, verify_eri: bool = False)\
    -> Tuple[int, float]:
    """ Compute lambda for Hamiltonian using SF method of Berry, et al.

    Args:
        cholesky_dim (int) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        integral_name (str) - name to indicate which integrals to use (currently only 'reiher' will
            work, and this argument will very likely change in the future.)
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orb (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """

    integral_name = str(integral_name).lower()

    # Import data from disk for now.
    #FIXME: move outside function + refactor after PySCF is integrated into pipeline
    cwd = os.path.dirname(os.path.abspath(__file__))
    with h5py.File(os.path.join(cwd, "..", "integrals",
                                "eri_"+integral_name+".h5"), "r") as f:
        eri = np.asarray(f['eri'][()])
        h0  = np.asarray(f['h0'][()])

    with h5py.File(os.path.join(cwd, "..", "integrals",
                                "eri_"+integral_name+"_cholesky.h5"), "r") as f:
        cholesky_diagonals = np.asarray(f["gval"][()])
        cholesky_lower_tri = np.asarray(f["gvec"][()])

    n_orb = len(h0)  # number orbitals

    # Check dims are consistent
    assert [n_orb] * 4 == [*eri.shape]

    # eliminate diagonals D from Cholesky decomposition LDL^T
    L = np.einsum("ij,j->ji",cholesky_lower_tri,
        np.sqrt(cholesky_diagonals)).reshape(-1, n_orb, n_orb)

    if verify_eri:
        # Make sure we are reading in the integrals correctly ... don't check for large cases (!)
        eri_new = np.einsum("ijP,Pkl->ijkl",L.T,L,optimize=True)
        assert np.allclose(eri_new.flatten(),eri.flatten())

    # Effective one electron operator contribution
    T = h0 - 0.5 * np.einsum("pqqs->ps", eri, optimize=True) +\
        np.einsum("pqrr->pq", eri, optimize = True)

    lambda_T = np.sum(np.abs(T))

    # Do rank-reduction of ERIs using cholesky_dim vectors
    LR = L[:cholesky_dim,:,:]

    # Two electron operator contributions
    lambda_W = 0.25 * np.einsum("Pij,Pkl->",np.abs(LR), np.abs(LR), optimize=True)
    lambda_tot = lambda_T + lambda_W

    return n_orb * 2, lambda_tot  #  return number spin orbitals from spatial and combined lambda


if __name__ == '__main__':

    CHOL_DIM = 200
    NAME = 'reiher'
    VERIFY=True
    number_orbitals, total_lambda = compute_lambda(cholesky_dim=CHOL_DIM,
        integral_name=NAME,verify_eri=VERIFY)
    print(number_orbitals, total_lambda, CHOL_DIM)
    assert number_orbitals == 108
    assert np.isclose(total_lambda,4258.0)
