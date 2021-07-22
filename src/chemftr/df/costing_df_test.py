"""Test cases for costing_df.py
"""
from chemftr.df.costing_df import cost_df, power_two


def test_reiher_df():
    """ Reproduce Reiher et al orbital DF FT costs from paper """
    DE = 0.001
    CHI = 10

    # Reiher et al orbitals
    N = 108
    LAM = 294.8
    L = 360
    LXI = 13031
    BETA = 16

    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    output = cost_df(N,LAM,DE,L,LXI,CHI,BETA,stps=20000)
    stps1 = output[0]
    output = cost_df(N,LAM,DE,L,LXI,CHI,BETA,stps1)
    assert output == (21753, 10073183463, 3725)
    print("OUTPUT (Reiher): ", output)


def test_li_df():
    """ Reproduce Li et al orbital THC DF costs from paper """
    DE = 0.001
    CHI = 10

    # Li et al orbitals
    N = 152
    LAM = 1171.2
    L = 394
    LXI = 20115
    BETA = 20

    # Here we're using an initial calculation with a very rough estimate of the number of steps
    # to give a more accurate number of steps. Then we input that into the function again.
    output = cost_df(N,LAM,DE,L,LXI,CHI,BETA,stps=20000)
    stps2 = output[0]
    output = cost_df(N,LAM,DE,L,LXI,CHI,BETA,stps2)
    assert output == (35008, 64404812736, 6404)
    print("OUTPUT (Li): ", output)


def test_power_two():
    """ Test for power_two(m) which returns power of 2 that is a factor of m """
    try:
        power_two(-1234)
    except AssertionError:
        pass
    assert power_two(0) == 0
    assert power_two(2) == 1
    assert power_two(3) == 0
    assert power_two(104) == 3  # 2**3 * 13
    assert power_two(128) == 7  # 2**7
    assert power_two(393120) == 5  # 2**5 * 3**3 * 5 * 7 * 13
