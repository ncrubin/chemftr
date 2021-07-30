""" Utilities for FT costing calculations """
from typing import Tuple
import numpy as np
import time


def QR(L : int, M1 : int) -> Tuple[int, int]:
    """ This gives the optimal k and minimum cost for a QROM over L values of size M.

    Args:
        L (int) -
        M1 (int) -

    Returns:
       k_opt (int) - k that yields minimal (optimal) cost of QROM
       val_opt (int) - minimal (optimal) cost of QROM
    """
    k = 0.5 * np.log2(L/M1)
    assert k >= 0
    value = lambda k: L/np.power(2,k) + M1*(np.power(2,k) - 1)
    k_int = [np.floor(k),np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)

def QR2(L1: int, L2: int, M1: int) -> Tuple[int, int, int]:
    """ This gives the optimal k values and minimum cost for a QROM using two L values of size M,
        e.g. the optimal k values for the QROM on two registers.

    Args:
        L1 (int) -
        L2 (int) -
        M1 (int) -

    Returns:
       k1_opt (int) - k1 that yields minimal (optimal) cost of QROM with two registers
       k2_opt (int) - k2 that yields minimal (optimal) cost of QROM with two registers
       val_opt (int) - minimal (optimal) cost of QROM
    """

    k1_opt, k2_opt = 0, 0 
    val_opt = 1e50
    # Doing this as a stupid loop for now, worth refactoring but cost is quick regardless
    # Biggest concern is if k1 / k2 range is not large enough!
    for k1 in range(1, 17):
        for k2 in range(1, 17):
            value = np.ceil(L1 / np.power(2, k1)) * np.ceil(L2 / np.power(2, k2)) +\
                M1 * (np.power(2, k1 + k2) - 1)
            if value < val_opt:
                val_opt = value
                k1_opt = k1
                k2_opt = k2

    assert val_opt.is_integer()
    return int(np.power(2, k1_opt)), int(np.power(2,k2_opt)), int(val_opt)

def QI(L: int) -> Tuple[int, int]:
    """ This gives the optimal k and minimum cost for an inverse QROM over L values.

    Args:
        L (int) -

    Returns:
       k_opt (int) - k that yiles minimal (optimal) cost of inverse QROM
       val_opt (int) - minimal (optimal) cost of inverse QROM
    """
    k = 0.5 * np.log2(L)
    assert k >= 0
    value = lambda k: L/np.power(2,k) + np.power(2,k)
    k_int = [np.floor(k),np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)

# FIXME: Is this ever used? It's defined in costingsf.nb, but I don't think it was ever called.
def QI2(L1: int, L2: int) -> Tuple[int, int, int]:
    """ This gives the optimal k values and minimum cost for inverse QROM using two L values,
        e.g. the optimal k values for the inverse QROM on two registers.

    Args:
        L1 (int) -
        L2 (int) -

    Returns:
       k1_opt (int) - k1 that yields minimal (optimal) cost of inverse QROM with two registers
       k2_opt (int) - k2 that yields minimal (optimal) cost of inverse QROM with two registers
       val_opt (int) - minimal (optimal) cost of inverse QROM with two registers
    """

    k1_opt, k2_opt = 0, 0 
    val_opt = 1e50
    # Doing this as a stupid loop for now, worth refactoring but cost is quick regardless
    # Biggest concern is if k1 / k2 range is not large enough!
    for k1 in range(1, 17):
        for k2 in range(1, 17):
            value = np.ceil(L1 / np.power(2, k1)) * np.ceil(L2 / np.power(2, k2)) +\
                np.power(2, k1 + k2)
            if value < val_opt:
                val_opt = value
                k1_opt = k1
                k2_opt = k2

    assert val_opt.is_integer()
    return int(np.power(2,k1_opt)), int(np.power(2,k2_opt)), int(val_opt)

def power_two(m: int) -> int:
    """ Return the power of two that is a factor of m """
    assert m >= 0
    if m % 2 == 0:
        count = 0
        while (m > 0) and (m % 2 == 0):
            m = m // 2
            count += 1
        return count
    return 0


def modified_cholesky(M, tol=1e-6, verbose=True, cmax=120):
    # JJG FIXME: taken froom pauxy-qmc
    """Modified cholesky decomposition of matrix.
    See, e.g. [Motta17]_
    Parameters
    ----------
    M : :class:`numpy.ndarray`
        Positive semi-definite, symmetric matrix.
    tol : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors.
    """
    # matrix of residuals.
    assert len(M.shape) == 2
    delta = np.copy(M.diagonal())
    nchol_max = int(cmax*M.shape[0]**0.5)
    # index of largest diagonal element of residual matrix.
    nu = np.argmax(np.abs(delta))
    delta_max = delta[nu]
    if verbose:
        print ("# max number of cholesky vectors = %d"%nchol_max)
        print ("# iteration %d: delta_max = %f"%(0, delta_max.real))
    # Store for current approximation to input matrix.
    Mapprox = np.zeros(M.shape[0], dtype=M.dtype)
    chol_vecs = np.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
    nchol = 0
    chol_vecs[0] = np.copy(M[:,nu])/delta_max**0.5
    while abs(delta_max) > tol:
        # Update cholesky vector
        start = time.time()
        Mapprox += chol_vecs[nchol]*chol_vecs[nchol].conj()
        delta = M.diagonal() - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        nchol += 1
        Munu0 = np.dot(chol_vecs[:nchol,nu].conj(), chol_vecs[:nchol,:])
        chol_vecs[nchol] = (M[:,nu] - Munu0) / (delta_max)**0.5
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %d: delta_max = %13.8e: time = %13.8e"%info)

    return np.array(chol_vecs[:nchol]).T


def eigendecomp(M, tol=1.15E-16):
    """ Decompose matrix M into L.L^T where rank(L) < rank(M) to some threshold 

    Args:
       M (np.ndarray) - (N x N) positive semi-definite matrix to be decomposed
       tol (float) - eigenpairs with eigenvalue above tol will be kept

    Returns:
       L (np.ndarray) - (K x N) array such that K <= N and L.L^T = M 
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Put in descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:,::-1]

    # Truncate
    idx = np.where(eigenvalues > tol)[0]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]

    # eliminate eigenvlaues from eigendecomposition 
    L = np.einsum("ij,j->ij",eigenvectors,
        np.sqrt(eigenvalues))

    return L


