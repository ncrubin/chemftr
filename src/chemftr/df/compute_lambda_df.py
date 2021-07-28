""" Compute lambda for double low rank factorization method of von Burg, et al """
from typing import Tuple
import os
import numpy as np
import h5py


def compute_lambda(thresh: float, integral_name: str, verify_eri: bool = False)\
     -> Tuple[int, float, float, float]:
    """ Compute lambda for Hamiltonian using DF method of von Burg, et al.

    Args:
        thresh (float) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        integral_name (str) - name to indicate which integrals to use (currently only 'reiher' will
            work, and this argument will very likely change in the future.)
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orbitals (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the double factorized Hamiltonian
        L (int) - the rank of the first decomposition
        Lxi (int) - the total number of eigenvectors
    """

    integral_name = str(integral_name).lower()

    # Import data from disk for now.
    #FIXME: move outside function + refactor after PySCF is integrated into pipeline
    cwd = os.path.dirname(os.path.abspath(__file__))
    with h5py.File(os.path.join(cwd, "..", "integrals",
                                "eri_"+integral_name+".h5"), "r") as f:
        eri = np.asarray(f['eri'][()])
        h0  = np.asarray(f['h0'][()])

    n_orb = len(h0)  # number orbitals
    # Check dims are consistent
    assert [n_orb] * 4 == [*eri.shape]

    try:
        with h5py.File(os.path.join(cwd, "..", "integrals",
                                    "eri_"+integral_name+"_cholesky.h5"), "r") as f:
            cholesky_diagonals = np.asarray(f["gval"][()])
            cholesky_lower_tri = np.asarray(f["gvec"][()])

    except FileNotFoundError:

        # Cholesky factored ints do not exist, so create them
        cholesky_diagonals, cholesky_lower_tri = np.linalg.eigh(eri.reshape(n_orb **2, n_orb **2))

        # Put in descending order
        cholesky_diagonals = cholesky_diagonals[::-1]
        cholesky_lower_tri = cholesky_lower_tri[:,::-1]

        # Truncate
        # FIXME: add user-defined threshold (usually 1E-8 is more than enough)
        idx = np.where(cholesky_diagonals > 1.15E-16)[0]
        cholesky_diagonals, cholesky_lower_tri = cholesky_diagonals[idx], cholesky_lower_tri[:,idx]

        #FIXME: Add option to save rank reduced integrals

    # eliminate diagonals D from Cholesky decomposition LDL^T
    L = np.einsum("ij,j->ij",cholesky_lower_tri,
        np.sqrt(cholesky_diagonals))

    if verify_eri:
        # Make sure we are reading in the integrals correctly ... don't check for large cases (!)
        eri_new = np.einsum('ik,kj->ij',L,L.T,optimize=True)
        assert np.allclose(eri_new.flatten(),eri.flatten())

    # Reshape for lambda calcs
    L = L.reshape(n_orb, n_orb, -1)

    nchol_max = cholesky_diagonals.shape[0]

    T = h0 - 0.5 * np.einsum("illj->ij", eri) + np.einsum("llij->ij", eri)
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
    NAME = 'reiher'
    VERIFY=True
    n_orbital, total_lambda, rank, num_eigen = compute_lambda(thresh=THRESH,
        integral_name=NAME,verify_eri=VERIFY)
    assert n_orbital == 108
    assert rank == 360
    assert num_eigen == 13031
    assert np.isclose(np.round(total_lambda,decimals=1),294.8)
