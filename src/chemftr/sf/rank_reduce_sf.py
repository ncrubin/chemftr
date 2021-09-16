""" Single factorization of the ERI tensor """ 
import numpy as np


def single_factorize(eri_full, cholesky_dim, reduction='eigendecomp',verify_eri=True):
    """ Do single factorization of the ERI tensor

    Args:
       eri_full (np.ndarray) - 4D (N x N x N x N) full ERI tensor
       cholesky_dim (int) - number of vectors to retain in ERI rank-reduction procedure
       reduction (str) - type of initial rank reduction on ERI ('cholesky' or 'eigendecomp')
       verify_eri (bool) - verify that initial decomposition can reconstruct the ERI tensor

    Returns:
       eri_rr (np.ndarray) - 4D approximate ERI tensor reconstructed from LR vectors
       LR (np.ndarray) - 3D (N x N x cholesky_dim) tensor containing vectors from rank-reduction
    """
    n_orb = eri_full.shape[0]
    assert n_orb**4 == len(eri_full.flatten())

    if reduction == 'cholesky':
        L = modified_cholesky(eri_full.reshape(n_orb**2, n_orb**2),tol=1e-16,verbose=False)

    elif reduction == 'eigendecomp':
        L = eigendecomp(eri_full.reshape(n_orb**2, n_orb**2),tol=1e-16)

    if verify_eri:
        # Make sure we are reading in the integrals correctly ... don't check for large cases (!)
        eri_rr = np.einsum('ik,kj->ij',L,L.T,optimize=True)
        assert np.allclose(eri_rr.flatten(),eri_full.flatten())

    # Do rank-reduction of ERIs using cholesky_dim vectors

    if cholesky_dim is None:
        LR = L[:,:]
    else:
        LR = L[:,:cholesky_dim]
    eri_rr = np.einsum('ik,kj->ij',LR,LR.T,optimize=True)
    eri_rr = eri_rr.reshape(n_orb, n_orb, n_orb, n_orb)
    LR = LR.reshape(n_orb, n_orb, -1)
    if cholesky_dim is not None:
        try:
            assert LR.shape[2] == cholesky_dim
        except AssertionError:
            sys.exit("LR.shape:     %s\ncholesky_dim: %s\nLR.shape and cholesky_dim are inconsistent" % (LR.shape, cholesky_dim))

    #print("ERI delta = ", np.linalg.norm(eri_rr - eri_full))
    return eri_rr, LR

# JJG FIXME: taken from pauxy-qmc
def modified_cholesky(M, tol=1e-6, verbose=True, cmax=120):
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

    # eliminate eigenvalues from eigendecomposition
    L = np.einsum("ij,j->ij",eigenvectors,
        np.sqrt(eigenvalues))

    return L
