""" Compute CCSD(T) for double low rank factorization method of von Burg, et al. """
from typing import Tuple
import numpy as np
from chemftr.rank_reduce import double_factorize
from chemftr.util import read_cas, ccsd_t

# FIXME: remove use_kernel when finished debugging
def compute_ccsd_t(thresh: float, integral_path: str, num_alpha = None, num_beta = None, \
    reduction: str = 'eigendecomp', verify_eri: bool = False, use_kernel = False) -> Tuple[int, float]:
    """ Compute CCSD(T) energy for Hamiltonian using DF method of von Burg, et al.

    Args:
        thresh (float) - threshold for double factorization truncationn strategy 
        integral_path (str) - path to file which integrals to use; assumes hdf5 with 'h0' and 'eri'
        reduction (str) -  method to rank-reduce ERIs. 'cholesky' or 'eigendecomp'
        verify_eri (bool) - check full cholesky integrals and ERIs are equivalent to epsilon

    Returns:
        e_scf (float) - SCF energy
        e_cor (float) - Correlation energy from CCSD(T)
        e_tot (float) - Total energy; i.e. SCF energy + Correlation energy from CCSD(T)
    """
    h1, eri_full, ecore, num_alpha, num_beta = read_cas(integral_path, num_alpha, num_beta)
    eri_rr, _, _, _ = double_factorize(eri_full, thresh, reduction, verify_eri)
    e_scf, e_cor, e_tot = ccsd_t(h1, eri_rr, ecore, num_alpha, num_beta, eri_full, use_kernel)

    return e_scf, e_cor, e_tot

if __name__ == '__main__':

    NAME = '../integrals/eri_reiher.h5'
    VERIFY=True
    escf, ecorr, etot = compute_ccsd_t(thresh=0.0,integral_path=NAME,
                                      num_alpha=27,num_beta=27,verify_eri=VERIFY)
    exact_energy = ecorr
    appx_energy = []
    thresholds = [0.1, 0.00125, 0.00005]
    for thresh in thresholds:
        escf, ecorr, etot = compute_ccsd_t(thresh,integral_path=NAME,
                                           num_alpha=27,num_beta=27,verify_eri=VERIFY)
        appx_energy.append(ecorr)

    appx_energy = np.asarray(appx_energy)
    error = (appx_energy - exact_energy)*1E3  # mEh

    for i,thresh in enumerate(thresholds):
        print('{:>12.4f}  {:>12.6f}  {:>12.2f}'.format(thresh, appx_energy[i], error[i]))
