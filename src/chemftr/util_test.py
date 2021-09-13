"""Test cases for util.py
"""
import numpy as np
from pyscf import gto, scf, cc
from chemftr.util import QR, QI, QR2, QI2, power_two, ccsd_t
from chemftr.molecule import pyscf_to_cas


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

def test_QR2():
    """ Tests function QR2 which gives the minimum cost for a QROM with two registers. """
    # Tests checked against Mathematica noteboook `costingsf.nb`
    # Arguments are otherwise random
    assert QR2(12, 34, 81) == (2, 2, 345)
    assert QR2(712, 340111, 72345) == (4, 16, 8341481)

def test_QI2():
    """ Tests function QI which gives the minimum cost for inverse QROM with two registers. """
    # Tests checked against Mathematica noteboook `costingsf.nb`
    # Arguments are otherwise random
    assert QI2(1234,5678) == (32, 64, 5519)
    assert QI2(7120,1340111) == (4, 32768, 204052)

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

def test_full_ccsd_t():
    """ Test chemftr full CCSD(T) from h1/eri/ecore tensors matches regular PySCF CCSD(T) """

    for scf_type in ['rhf', 'rohf']:
        mol = gto.Mole()
        mol.atom = 'H 0 0 0; F 0 0 1.1'
        mol.charge = 0
        if scf_type == 'rhf':
            mol.spin = 0
        elif scf_type == 'rohf':
            mol.spin = 2
        mol.basis = 'ccpvtz'
        mol.symmetry = False
        mol.build()

        if scf_type == 'rhf':
            mf = scf.RHF(mol)
        elif scf_type == 'rohf':
            mf = scf.ROHF(mol) 

        mf.init_guess = 'mindo'
        mf.conv_tol = 1e-10
        mf.kernel()
        mo1 = mf.stability()[0]
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)

        # Do PySCF CCSD(T)
        mycc = cc.CCSD(mf)
        mycc.max_cycle = 500
        mycc.conv_tol = 1E-9
        mycc.conv_tol_normt = 1E-5
        mycc.diis_space = 24
        mycc.diis_start_cycle = 4
        mycc.kernel()
        et = mycc.ccsd_t()

        pyscf_escf = mf.e_tot
        pyscf_ecor = mycc.e_corr + et
        pyscf_etot = pyscf_escf + pyscf_ecor
        pyscf_results = np.array([pyscf_escf, pyscf_ecor, pyscf_etot])

        n_elec = mol.nelectron
        n_orb = mf.mo_coeff[0].shape[-1]

        chemftr_results = ccsd_t(*pyscf_to_cas(mf, n_orb, n_elec))
        chemftr_results = np.asarray(chemftr_results)

        # ignore relative tolerance, we just want absolute tolerance
        assert np.allclose(pyscf_results,chemftr_results,rtol=1E-14)

def test_reduced_ccsd_t():
    """ Test chemftr reduced (2e space) CCSD(T) from tensors matches PySCF CAS(2e,No)"""

    for scf_type in ['rhf','rohf']:
        mol = gto.Mole()
        mol.atom = 'H 0 0 0; F 0 0 1.1'
        mol.charge = 0
        if scf_type == 'rhf':
            mol.spin = 0
        elif scf_type == 'rohf':
            mol.spin = 2
        mol.basis = 'ccpvtz'
        mol.symmetry = False
        mol.build()

        if scf_type == 'rhf':
            mf = scf.RHF(mol)
        elif scf_type == 'rohf':
            mf = scf.ROHF(mol) 

        mf.init_guess = 'mindo'
        mf.conv_tol = 1e-10
        mf.kernel()
        mo1 = mf.stability()[0]
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)

        # Do PySCF CAS(No,2e) -- for 2 electrons CCSD (and so CCSD(T)) is exact
        n_elec = 2 # electrons
        n_orb = mf.mo_coeff[0].shape[-1] - mf.mol.nelectron - n_elec
        mycas = mf.CASCI(n_orb, n_elec).run()

        pyscf_etot = mycas.e_tot

        # Don't do triples (it's zero anyway for 2e) b/c div by zero error for ROHF references
        _, _, chemftr_etot = ccsd_t(*pyscf_to_cas(mf, n_orb, n_elec),no_triples=True)

        # ignore relative tolerance, we just want absolute tolerance
        print("Pyscf:", pyscf_etot, " Chemftr: ", chemftr_etot)
        assert np.isclose(pyscf_etot,chemftr_etot,rtol=1E-14)
