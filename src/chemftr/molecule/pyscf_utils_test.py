"""Test cases for pyscf_utils.py
"""
import numpy as np
from pyscf import gto, scf, cc
from chemftr.util import QR, QI, QR2, QI2, power_two
from chemftr.molecule import pyscf_to_cas, ccsd_t, stability


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
        mf = stability(mf) 

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
        mf = stability(mf)

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
