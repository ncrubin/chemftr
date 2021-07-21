"""Test cases for costing_thc.py
"""
from chemftr.thc.costing_thc import cost_thc


def test_reiher_thc():
    """ Reproduce Reiher et al orbital THC FT costs from paper """
    DE = 0.001
    CHI = 10

    # Reiher et al orbitals
    N = 108
    LAM = 306.3
    BETA = 16
    THC_DIM = 350

    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    output = cost_thc(N,LAM,DE,CHI,BETA,THC_DIM,stps=20000)
    stps1 = output[0]
    output = cost_thc(N,LAM,DE,CHI,BETA,THC_DIM,stps1)
    assert output == (10912, 5250145120, 2142)


def test_li_thc():
    """ Reproduce Li et al orbital THC FT costs from paper """
    DE = 0.001
    CHI = 10

    # Li et al orbitals
    N = 152
    LAM = 1201.5
    BETA = 20
    THC_DIM = 450

    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    output = cost_thc(N,LAM,DE,CHI,BETA,THC_DIM,stps=20000)
    stps2 = output[0]
    output = cost_thc(N,LAM,DE,CHI,BETA,THC_DIM,stps2)
    assert output == (16923, 31938980976, 2196)
    print("OUTPUT (Li): ", output)
