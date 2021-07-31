""" Routines to grab active space Hamiltonian and do CCSD(T) on the active space """
from typing import Tuple
import numpy as np

from pyscf import gto, scf, mcscf, ao2mo, cc

def get_cas(mf, cas_orbitals: int, cas_electrons: int, avas_orbs=None):
    """ Generate CAS Hamiltonian given a PySCF mean field object

    Args:
        mf (PySCF mean field object) - instantiation of PySCF mean field method class
        cas_orbitals (int) - number of orbitals in CAS space
        cas_electrons (int) - number of electrons in CAS space

    Returns:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        cas_alpha, cas_beta (Tuple(int, int)) - number of spin alpha and spin beta electrons in CAS
    """

    # Only can do RHF or ROHF with mcscf.CASCI
    assert isinstance(mf, scf.rhf.RHF)  # ROHF inherits RHF class (i.e. ROHF == RHF but RHF != ROHF)
    cas = mcscf.CASCI(mf, ncas = cas_orbitals, nelecas = cas_electrons)
    h1, ecore = cas.get_h1eff(mo_coeff = avas_orbs)
    eri = cas.get_h2cas(mo_coeff = avas_orbs)
    eri = ao2mo.restore('s1', eri, h1.shape[0])  # chemist convention (11|22)
    ecore = float(ecore)

    # Sanity checks and active space info
    total_electrons = mf.mol.nelectron
    frozen_electrons  = total_electrons - cas_electrons
    assert frozen_electrons % 2 == 0

    # Again, recall ROHF == RHF but RHF != ROHF, and we only do either RHF or ROHF
    if isinstance(mf, scf.rohf.ROHF):
        frozen_alpha = frozen_electrons // 2
        frozen_beta  = frozen_electrons // 2
        cas_alpha = mf.nelec[0] - frozen_alpha
        cas_beta  = mf.nelec[1] - frozen_beta
        assert np.isclose(cas_beta + cas_alpha, cas_electrons)

    else:
        assert cas_electrons % 2 == 0
        cas_alpha = cas_electrons // 2
        cas_beta  = cas_electrons // 2

    return h1, eri, ecore, (cas_alpha, cas_beta)


def ccsd_t(h1, eri, ecore, num_alpha: int, num_beta: int, eri_full = None) \
    -> Tuple[float, float, float]:
    """ Do CCSD(T) on set of one- and two-body Hamiltonian elements

    Args:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis), may be rank-reduced
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin alpha electrons in Hamiltonian
        num_beta (int) - number of spin beta electrons in Hamiltonian
        eri_full (ndarray) - optional 4D tensor containing full (i.e. not rank-reduced) two-body
            terms (MO basis) for the SCF procedure only

    Returns:
        e_scf (float) - SCF energy
        e_cor (float) - Correlation energy from CCSD(T)
        e_tot (float) - Total energy; i.e. SCF energy + Correlation energy from CCSD(T)
    """

    mol = gto.M()
    mol.nelectron = num_alpha + num_beta
    mol.incore_anyway = True

    # If eri_full not provided, use (possibly rank-reduced) ERIs for SCF and SCF energy check
    if eri_full is None:
        eri_full = eri

    # Assumes Hamiltonian is either RHF or ROHF ... should be OK since UHF will have two h1s, etc.
    if num_alpha == num_beta:
        mf = scf.RHF(mol)
        scf_energy = ecore + \
                     2*np.einsum('ii',h1[:num_alpha,:num_alpha]) + \
                     2*np.einsum('iijj',eri_full[:num_alpha,:num_alpha,:num_alpha,:num_alpha]) - \
                       np.einsum('ijji',eri_full[:num_alpha,:num_alpha,:num_alpha,:num_alpha])

    else:
        mf = scf.ROHF(mol)
        mf.nelec = (num_alpha, num_beta)
        # grab singly and doubly occupied orbitals (assumes high-spin open shell)
        docc = slice(None                    , min(num_alpha, num_beta))
        socc = slice(min(num_alpha, num_beta), max(num_alpha, num_beta))
        scf_energy = ecore + \
                     2.0*np.einsum('ii',h1[docc, docc]) + \
                         np.einsum('ii',h1[socc, socc]) + \
                     2.0*np.einsum('iijj',eri_full[docc, docc, docc, docc]) - \
                         np.einsum('ijji',eri_full[docc, docc, docc, docc]) + \
                         np.einsum('iijj',eri_full[socc, socc, docc, docc]) - \
                     0.5*np.einsum('ijji',eri_full[socc, docc, docc, socc]) + \
                         np.einsum('iijj',eri_full[docc, docc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri_full[docc, socc, socc, docc]) + \
                     0.5*np.einsum('iijj',eri_full[socc, socc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri_full[socc, socc, socc, socc])

    mf.get_hcore  = lambda *args: np.asarray(h1)
    mf.get_ovlp   = lambda *args: np.eye(h1.shape[0])
    mf.energy_nuc = lambda *args: ecore
    mf._eri = eri_full # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)

    mf.conv_tol = 1e-10
    mf.init_guess = '1e'
    mf.kernel()
    mol = mf.stability()[0]
    dm = mf.make_rdm1(mol, mf.mo_occ)
    mf = mf.run(dm)
    mol = mf.stability()[0]
    dm = mf.make_rdm1(mol, mf.mo_occ)
    mf = mf.run(dm)

    # Now replace with possibly rank-reduced ERIs for the coupled cluster calculation
    mf._eri = eri

    # Check SCF has not changed by doing restart!
    #print(scf_energy, mf.e_tot)
    try:
        assert np.isclose(scf_energy, mf.e_tot,rtol=1e-14)
    except AssertionError:
        print("WARNING: SCF energy from input integrals does not match SCF energy from mf.kernel()")
        print("E(SCF, ints) = {:12.6f} whereas E(SCF) = {:12.6f}".format(scf_energy,mf.e_tot))

    mycc = cc.CCSD(mf)
    mycc.max_cycle = 300
    mycc.verbose = 4
    mycc.kernel()
    et = mycc.ccsd_t()

    e_scf = mf.e_tot
    e_cor = mycc.e_corr + et
    e_tot = e_scf + e_cor

    print("E(SCF, ints): ", scf_energy)
    print("E(SCF):       ", e_scf)
    print("E(cor):       ", e_cor)
    print("Total energy: ", e_tot)
    return e_scf, e_cor, e_tot


if __name__ == '__main__':

    print('Doing: RHF')
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.charge = 0
    mol.spin = 0
    mol.basis = 'ccpvtz'
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.init_guess = 'mindo'
    mf.conv_tol = 1e-10
    mf.kernel()
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)

    mycc = cc.CCSD(mf)
    mycc.max_cycle = 300
    mycc.kernel()
    et = mycc.ccsd_t()

    print("E(SCF):       ", mf.e_tot)
    print("E(cor):       ", mycc.e_corr)
    print("E(T):         ", et)
    print("Total energy: ", mycc.e_corr + mf.e_tot + et)

    n_elec = mol.nelectron
    n_orb = mf.mo_coeff[0].shape[-1]

    # Now repeat, but freeze two core electrons
    frozen = 2
    n_elec -= frozen
    n_orb -= frozen//2
    h1, eri, ecore, (num_alpha, num_beta) = get_cas(mf, n_orb, n_elec)

    ccsd_t(h1, eri, ecore, num_alpha, num_beta)

    print('\nDoing: ROHF')
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.charge = 0
    mol.spin = 2
    mol.basis = 'ccpvtz'
    mol.symmetry = False
    mol.build()

    mf = scf.ROHF(mol)
    mf.init_guess = 'mindo'
    mf.conv_tol = 1e-10
    mf.kernel()
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)

    mycc = cc.CCSD(mf)
    mycc.max_cycle = 300
    mycc.kernel()
    et = mycc.ccsd_t()

    print("E(SCF):       ", mf.e_tot)
    print("E(cor):       ", mycc.e_corr)
    print("E(T):         ", et)
    print("Total energy: ", mycc.e_corr + mf.e_tot + et)

    n_elec = mol.nelectron
    n_orb = mf.mo_coeff[0].shape[-1]

    # Now repeat, but freeze two core electrons
    frozen = 2
    n_elec -= frozen
    n_orb -= frozen//2
    h1, eri, ecore, (num_alpha, num_beta) = get_cas(mf, n_orb, n_elec)

    ccsd_t(h1, eri, ecore, num_alpha, num_beta)
