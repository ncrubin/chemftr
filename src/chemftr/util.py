""" Utilities for FT costing calculations """
from typing import Tuple
import numpy as np


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
