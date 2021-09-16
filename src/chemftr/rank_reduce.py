""" Utilities for rank reduction of ERIs """
import sys
import time
import numpy as np
import uuid

from chemftr.thc.thc_factorization import lbfgsb_opt_thc, adagrad_opt_thc

def thc_via_cp3(eri_full, nthc, first_factor_thresh=1.0E-8, conv_eps=1.0E-4, perform_bfgs_opt=True, bfgs_chkfile_name=None,
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
    beta, gamma, scale = pybtas.cp3_from_cholesky(u_chol.copy(), nthc, random_start=random_start_thc, conv_eps=conv_eps)
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
            chkfile = "{}_bfgs_post_cp3.chk".format(str(uuid.uuid4())[:8])
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
