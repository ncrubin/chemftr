""" Utilities for rank reduction of ERIs """
import sys
import time
import numpy as np
import uuid

from chemftr.thc.thc_factorization import lbfgsb_opt_thc, adagrad_opt_thc


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

def double_factorize(eri_full, thresh, reduction='eigendecomp',verify_eri=True):
    """ Do double factorization of the ERI tensor

    Args:
       eri_full (np.ndarray) - 4D (N x N x N x N) full ERI tensor
       thresh (float) - threshold for double factorization
       reduction (str) - type of initial rank reduction on ERI ('cholesky' or 'eigendecomp')
       verify_eri (bool) - verify that initial decomposition can reconstruct the ERI tensor

    Returns:
       eri_rr (np.ndarray) - 4D approximate ERI tensor reconstructed from LR vectors
       LR (np.ndarray) - 3D (N x N x cholesky_dim) tensor containing vectors from rank-reduction
    """
    _, L = single_factorize(eri_full, cholesky_dim=None, reduction=reduction, verify_eri=verify_eri)

    n_orb = eri_full.shape[0]
    assert n_orb**4 == len(eri_full.flatten())

    nchol_max = max(L.shape)

    # double factorized eris
    eri_rr = np.zeros_like(eri_full)

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
            break

        e_selected = np.diag(e[idx])
        v_selected = v[:,idx]

        Lij_selected = v_selected.dot(e_selected).dot(v_selected.T)

        eri_rr += np.einsum("ij,kl->ijkl", Lij_selected, Lij_selected, optimize=True)

        normSC = 0.25 * np.sum(np.abs(e_selected))**2
        lambda_F += normSC

    # incoherent error
    # ein = np.sqrt(np.sum(np.abs(eri - H_df)**2))

    #print("ERI error DF: ", np.linalg.norm(eri_rr - eri_full))

    return eri_rr, lambda_F, R, M


def thc_via_cp3(eri_full, nthc, first_factor_thresh=1.0E-8, perform_bfgs_opt=True, bfgs_chkfile_name=None,
                bfgs_maxiter=1500, random_start_thc=False, verify=False):
    """
    THC-CP3 performs an SVD decomposition of the eri matrix followed by a CP decomposition
    via pybtas.  The CP decomposition is assumes the tensor is symmetric in in the first two
    indices corresponding to a reshaped (and rescaled by the signular value) singular vector.

    Args:
        eri_full - (n x n x n x n) eri tensor in Mulliken (Chemists) ordering
        nthc - number of thc factors to use.
        first_factor_thresh - SVD threshold on eri matrix.  Default 1.0e-8 because
                              square of this is numerical precision.
        perform_bfgs_opt - Perform extra gradient optimization on top of CP3 decomp
        bfgs_maxiter - Maximum bfgs steps to take. Default 1500.
        random_start_thc - Perform random start for CP3.  If false perform HOSVD start.
        verify - check eri properties. Default is False

    returns:
        THC-eri, etaPp, MPQ.  etaPp and MPQ can be used in lambda calculation.
    """
    # fail fast if we don't have the tools to use this routine
    try:
        import pybtas
    except ImportError:
        raise ImportError("pybtas could not be imported. Is it installed and in your PYTHONPATH?")
    
    norb = eri_full.shape[0]
    if verify:
        assert np.allclose(eri_full, eri_full.transpose(1, 0, 2, 3))  # (ij|kl) == (ji|kl)
        assert np.allclose(eri_full, eri_full.transpose(0, 1, 3, 2))  # (ij|kl) == (ij|lk)
        assert np.allclose(eri_full, eri_full.transpose(1, 0, 3, 2))  # (ij|kl) == (ji|lk)
        assert np.allclose(eri_full, eri_full.transpose(2, 3, 0, 1))  # (ij|kl) == (kl|ij)

    eri_mat = eri_full.transpose(0, 1, 3, 2).reshape((norb ** 2, norb ** 2))
    if verify:
        assert np.allclose(eri_mat, eri_mat.T)

    u, sigma, vh = np.linalg.svd(eri_mat)
    
    if verify:
        assert np.allclose(u @ np.diag(sigma) @ vh, eri_mat)

    non_zero_sv = np.where(sigma >= first_factor_thresh)[0]
    u_chol = u[:, non_zero_sv]
    diag_sigma = np.diag(sigma[non_zero_sv])
    u_chol = u_chol @ np.diag(np.sqrt(sigma[non_zero_sv]))

    if verify:
        test_eri_mat_mulliken = u[:, non_zero_sv] @ diag_sigma @ vh[non_zero_sv, :]
        assert np.allclose(test_eri_mat_mulliken, eri_mat)

    start_time = time.time()  # timing results if requested by user
    beta, gamma, scale = pybtas.cp3_from_cholesky(u_chol.copy(), nthc, random_start=random_start_thc)
    cp3_calc_time = time.time() - start_time
    
    if verify:
        u_alpha = np.zeros((norb, norb, len(non_zero_sv)))
        for ii in range(len(non_zero_sv)):
            u_alpha[:, :, ii] = np.sqrt(sigma[ii]) * u[:, ii].reshape((norb, norb))
            assert np.allclose(u_alpha[:, :, ii], u_alpha[:, :, ii].T)  # consequence of working with Mulliken rep

        u_alpha_test = np.einsum("ar,br,xr,r->abx", beta, beta, gamma, scale.ravel())
        print("\tu_alpha l2-norm ", np.linalg.norm(u_alpha_test - u_alpha))

    thc_leaf = beta.T
    thc_gamma = np.einsum('xr,r->xr', gamma, scale.ravel())
    thc_central = thc_gamma.T @ thc_gamma

    if verify:
        eri_thc = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", thc_leaf, thc_leaf, thc_leaf, thc_leaf, thc_central, optimize=True)
        print("\tERI L2 CP3-THC ", np.linalg.norm(eri_thc - eri_full))
        print("\tCP3 timing: ", cp3_calc_time)

    if perform_bfgs_opt:
        if bfgs_chkfile_name is None:
            chkfile = "{}_bfgs_post_cp3.chk".format(str(uuid.uuid4()[:8]))
        else:
            chkfile = bfgs_chkfile_name

        x = np.hstack((thc_leaf.ravel(), thc_central.ravel()))
        lbfgs_start_time = time.time()
        x = lbfgsb_opt_thc(eri_full, nthc, initial_guess=x, chkfile_name=chkfile, maxiter=bfgs_maxiter)
        lbfgs_calc_time =  time.time() - lbfgs_start_time
        thc_leaf = x[:norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
        thc_central = x[norb * nthc:norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    total_calc_time = time.time() - start_time

    eri_thc = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", thc_leaf, thc_leaf, thc_leaf, thc_leaf, thc_central, optimize=True)
    return eri_thc, thc_leaf, thc_central


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