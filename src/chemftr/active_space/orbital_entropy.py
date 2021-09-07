"""
Tools for computing orbital entropy and mutual information between orbital-density matrices
"""
import numpy as np
from numpy import ndarray


def compute_orbital_reduced_density_matrix(orbital_index: int, opdm_alpha: ndarray, opdm_beta: ndarray,
                                           tpdm_alpha_beta: ndarray) -> ndarray:
    """
    Compute the orbital density matrix defined in Table 1 of arXiv:0508524

    Diagonal elements are given by  [<n_{p,alpha}(1-n_{p,beta}), n_{p,beta}(1 - n_{p,alpha}),
    (1 - n_{p,alpha}), (1 - n_{p,beta}), n_{p,alpha}n_{p,beta}]
    where p is the orbital_index and n is the n is the spin-orbital number operator.

    :param int orbital_index: single spatial orbital label (zero-indexed) to create orbital reduced density matrix for.
    :param ndarray opdm_alpha: one-particle-density matrix of the alpha-spin-orbitals. This is a norb x norb matrix
                                  where norb is the number of spatial orbtials.  orbital_index indexes these orbitals.
    :param ndarray opdm_beta:  one-particle-density matrix of the beta-spin-orbitals. This is a norb x norb matrix
                                  where norb is the number of spatial orbtials.  orbital_index indexes these orbitals.
    :param ndarray tpdm_alpha_beta: alpha-beta sector of tpdm in openfermion ordering <1'2'|21>.  1 and 1' range over
                                    alpha spin orbitals and 2 and 2' range over beta spin-orbitals.
    """
