""" Double factorization rank reduction of ERIs """ 
import sys
import numpy as np
from chemftr.sf import rank_reduce as single_factorize


def double_factorize(eri_full, thresh, reduction='eigendecomp',verify_eri=True):
    """ Do double factorization of the ERI tensor

    Args:
       eri_full (np.ndarray) - 4D (N x N x N x N) full ERI tensor
       thresh (float) - threshold for double factorization
       reduction (str) - type of initial rank reduction on ERI ('cholesky' or 'eigendecomp')
       verify_eri (bool) - verify that initial decomposition can reconstruct the ERI tensor

    Returns:
       eri_rr (np.ndarray) - 4D approximate ERI tensor reconstructed from LR vectors
       LR (np.ndarray) - 3D (N x N x M) tensor containing vectors from rank-reduction
       R (int) - rank retained from initial eigendecomposition 
       M (int) - number of eigenvectors 
    """
    _, L = single_factorize(eri_full, cholesky_dim=None, reduction=reduction, verify_eri=verify_eri)

    n_orb = eri_full.shape[0]
    assert n_orb**4 == len(eri_full.flatten())

    nchol_max = L.shape[2]

    # double factorized eris
    eri_rr = np.zeros_like(eri_full)

    M = 0 # rolling number of eigenvectors
    LR = []  # collect the selected vectors
    for R in range(nchol_max):
        Lij = L[:,:, R]
        e, v = np.linalg.eigh(Lij)
        normSC = np.sum(np.abs(e))

        truncation = normSC * np.abs(e)

        idx = truncation > thresh
        plus = np.sum(idx)
        M += plus

        if plus == 0:
            break

        e_selected = np.diag(e[idx])
        v_selected = v[:,idx]

        Lij_selected = v_selected.dot(e_selected).dot(v_selected.T)
        LR.append(Lij_selected)

    LR = np.asarray(LR).T
    eri_rr = np.einsum('ijP,klP', LR, LR, optimize=True)

    return eri_rr, LR, R, M
