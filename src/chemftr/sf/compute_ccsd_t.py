""" Compute CCSD(T) for single low rank factorization method of Berry, et al """
from typing import Tuple
import numpy as np
import h5py
from chemftr.util import modified_cholesky, eigendecomp
from chemftr.cc_util import ccsd_t
from pyscf import gto, scf, cc


def compute_ccsd_t(cholesky_dim: int, integral_path: str, nalpha = None, nbeta = None, \
    reduction: str = 'eigendecomp', verify_eri: bool = False) -> Tuple[int, float]:
    """ Compute CCSD(T) energy for Hamiltonian using SF method of Berry, et al.

    Args:
        cholesky_dim (int) - dimension to retain in Cholesky for low-rank reconstruction of ERIs
        integral_path (str) - path to file which integrals to use; assumes hdf5 with 'h0' and 'eri'
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        n_orb (int) - number of spin orbitals
        lambda_tot (float) - lambda value for the single factorized Hamiltonian
    """

    with h5py.File(integral_path, "r") as f:
        eri_full = np.asarray(f['eri'][()])
        try:
            h1  = np.asarray(f['h0'][()])
        except KeyError:
            h1  = np.asarray(f['hcore'][()])
        # ecore sometimes exists, and sometimes as enuc (no frozen electrons) ... set to zero if N/A
        try:
            ecore = float(f['ecore'][()])
        except KeyError:
            try:
                ecore = float(f['enuc'][()])
            except KeyError:
                ecore = 0.0
        try:
            mo_coeff = np.asarray(f['mo_coeff'][()])
        except KeyError:
            mo_coeff = None

    n_orb = len(h1)  # number orbitals
    # Check dims are consistent
    assert [n_orb] * 4 == [*eri_full.shape]

    # rank-reduced ints do not exist, so create them

    if reduction == 'cholesky':
        L = modified_cholesky(eri_full.reshape(n_orb**2, n_orb**2),tol=1e-16,verbose=False)

    elif reduction == 'eigendecomp':
        L = eigendecomp(eri_full.reshape(n_orb**2, n_orb**2),tol=1e-16)

    #FIXME: Add option to save and read rank-reduced integrals

    if verify_eri:
        # Make sure we are reading in the integrals correctly ... don't check for large cases (!)
        eri_rr = np.einsum('ik,kj->ij',L,L.T,optimize=True)
        assert np.allclose(eri_rr.flatten(),eri_full.flatten())

    # Do rank-reduction of ERIs using cholesky_dim vectors
    LR = L[:,:cholesky_dim]
    eri_rr = np.einsum('ik,kj->ij',LR,LR.T,optimize=True)
    eri_rr = eri_rr.reshape(n_orb, n_orb, n_orb, n_orb)
    eri_full = eri_full.reshape(n_orb, n_orb, n_orb, n_orb)
    print("ERI delta = ", np.linalg.norm(eri_rr - eri_full))

    if mo_coeff is not None:
        h1 = mo_coeff.conj().T.dot(h1).dot(mo_coeff)
        eri_full = np.einsum("mp,nq,lr,sk,mnls->pqrk",
                       mo_coeff.conj(),mo_coeff.conj(),mo_coeff,mo_coeff,eri_full,optimize=True)
        eri_rr   = np.einsum("mp,nq,lr,sk,mnls->pqrk",
                       mo_coeff.conj(),mo_coeff.conj(),mo_coeff,mo_coeff,eri_rr,  optimize=True)

    e_scf, e_cor, e_tot = ccsd_t(h1, eri_rr, ecore, nalpha, nbeta, eri_full)

    return e_scf, e_cor, e_tot



if __name__ == '__main__':

    NAME = '../integrals/eri_reiher.h5'
    VERIFY=True
    escf, ecorr, etot = compute_ccsd_t(cholesky_dim=-1,integral_path=NAME,nalpha=27,nbeta=27,
                                       verify_eri=VERIFY)
    exact_energy = ecorr
    appx_energy = []
    ranks = range(50,401,25)
    for CHOL_DIM in ranks:
        escf, ecorr, etot = compute_ccsd_t(cholesky_dim=CHOL_DIM,integral_path=NAME,
                                           nalpha=27,nbeta=27,verify_eri=VERIFY)
        appx_energy.append(ecorr)

    appx_energy = np.asarray(appx_energy)
    error = (appx_energy - exact_energy)*1E3  # mEh

    for i,rank in enumerate(ranks):
        print('{:>12}  {:>12.6f}  {:>12.2f}'.format(rank, appx_energy[i], error[i]))
