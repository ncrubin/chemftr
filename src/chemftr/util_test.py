"""Test cases for util.py
"""
from chemftr.util import QR, QI


def test_QR():
    """ Tests function QR which gives the minimum cost for a QROM over L values of size M. """
    # Tests checked against Mathematica noteboook `costingTHC.nb`
    # Arguments are otherwise random
    assert QR(12341234,5670) == (6,550042)
    assert QR(12201990,520199) == (2,4611095)


def test_QI():
    """ Tests function QI which gives the minimum cost for inverse QROM over L values. """
    # Tests checked against Mathematica noteboook `costingTHC.nb`
    # Arguments are otherwise random
    assert QI(987654) == (10,1989)
    assert QI(8052021) == (11,5980)
