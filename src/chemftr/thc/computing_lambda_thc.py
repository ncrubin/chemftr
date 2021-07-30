"""
Compute lambdas for THC according to 
PRX QUANTUM 2, 030305 (2021) Section II. D.
"""
import numpy as np


def compute_thc_lambda(oei: np.ndarray, etaPp: np.ndarray, MPQ: np.ndarray, true_eri: np.ndarray,
                       use_eri_reconstruct_for_v=False):
    """
    Compute lambda thc

    :param  oei: one-electron spatial orbitals
    :param etaPp: leaf tensor for THC that is  nthc x norb.  The nthc and norb is
                inferred from this quantity.
    :param MPQ:  central tensor for THC factorization. nthc x nthc
    :param true_eri: true two-electron spatial integrals (11'|22')  ordering--AKA chemist order
    zeta"""
    nthc = etaPp.shape[0]

    # computing Least-squares THC residual
    CprP = np.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    BprQ = np.tensordot(CprP, MPQ, axes=([2], [0]))
    Iapprox = np.tensordot(CprP, np.transpose(BprQ), axes=([2], [0]))
    deri = true_eri - Iapprox
    res = 0.5 * np.sum((deri) ** 2)

    # NOTE: remove in future once we resolve why it was being used in the first place.
    # NOTE: see T construction for details.
    eri_thc = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", etaPp, etaPp, etaPp, etaPp, MPQ, optimize=True)

    # projecting into the THC basis requires each THC factor mu to be  be normalized.
    # we roll the normalization constant into the central tensor zeta
    SPQ = etaPp.dot(etaPp.T)  # (nthc x norb)  x (norb x nthc) -> (nthc  x nthc) metric
    cP = np.diag(np.diag(SPQ))  # grab diagonal elements. equivalent to np.diag(np.diagonal(SPQ))
    # no sqrts because we have two normalized THC vectors (index by mu and nu) on each side.
    MPQ_normalized = cP.dot(MPQ).dot(cP)  # get normalized zeta in Eq. 11 & 12

    lambda_z = np.sum(np.abs(MPQ_normalized)) * 0.5  # Eq. 13
    # NCR: originally Joonho's code add np.einsum('llij->ij', eri_thc)
    # NCR: I don't know how much this matters.
    if use_eri_reconstruct_for_v:
        # use eri_thc for second coulomb contraction.  This was in the original code which is different than what the
        # paper says.
        T = oei - 0.5 * np.einsum("illj->ij", true_eri) + np.einsum("llij->ij", eri_thc)  # Eq. 3 + Eq. 18
    else:
        T = oei - 0.5 * np.einsum("illj->ij", true_eri) + np.einsum("llij->ij", true_eri)  # Eq. 3 + Eq. 18
    e, v = np.linalg.eigh(T)
    lambda_T = np.sum(np.abs(e))  # Eq. 19. NOTE: sum over spin orbitals removes 1/2 factor

    lambda_tot = lambda_z + lambda_T  # Eq. 20

    return nthc, np.sqrt(res), res, lambda_T, lambda_z, lambda_tot

