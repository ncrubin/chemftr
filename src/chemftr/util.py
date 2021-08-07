""" Utilities for FT costing calculations """
from typing import Tuple, Optional
import sys
import os
import h5py
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, cc


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
    try:
        assert k >= 0
    except AssertionError:
        sys.exit("In function QR: \
        \n  L is smaller than M: increase RANK or lower THRESH (or alternatively decrease CHI)")
    value = lambda k: L/np.power(2,k) + M1*(np.power(2,k) - 1)
    k_int = [np.floor(k),np.ceil(k)]  # restrict optimal k to integers
    k_opt = k_int[np.argmin(value(k_int))]  # obtain optimal k
    val_opt = np.ceil(value(k_opt))  # obtain ceiling of optimal value given k
    assert k_opt.is_integer()
    assert val_opt.is_integer()
    return int(k_opt), int(val_opt)

def QR2(L1: int, L2: int, M1: int) -> Tuple[int, int, int]:
    """ This gives the optimal k values and minimum cost for a QROM using two L values of size M,
        e.g. the optimal k values for the QROM on two registers.
    Args:
        L1 (int) -
        L2 (int) -
        M1 (int) -

    Returns:
       k1_opt (int) - k1 that yields minimal (optimal) cost of QROM with two registers
       k2_opt (int) - k2 that yields minimal (optimal) cost of QROM with two registers
       val_opt (int) - minimal (optimal) cost of QROM
    """

    k1_opt, k2_opt = 0, 0
    val_opt = 1e50
    # Doing this as a stupid loop for now, worth refactoring but cost is quick regardless
    # Biggest concern is if k1 / k2 range is not large enough!
    for k1 in range(1, 17):
        for k2 in range(1, 17):
            value = np.ceil(L1 / np.power(2, k1)) * np.ceil(L2 / np.power(2, k2)) +\
                M1 * (np.power(2, k1 + k2) - 1)
            if value < val_opt:
                val_opt = value
                k1_opt = k1
                k2_opt = k2

    assert val_opt.is_integer()
    return int(np.power(2, k1_opt)), int(np.power(2,k2_opt)), int(val_opt)

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

# FIXME: Is this ever used? It's defined in costingsf.nb, but I don't think it was ever called.
def QI2(L1: int, L2: int) -> Tuple[int, int, int]:
    """ This gives the optimal k values and minimum cost for inverse QROM using two L values,
        e.g. the optimal k values for the inverse QROM on two registers.

    Args:
        L1 (int) -
        L2 (int) -

    Returns:
       k1_opt (int) - k1 that yields minimal (optimal) cost of inverse QROM with two registers
       k2_opt (int) - k2 that yields minimal (optimal) cost of inverse QROM with two registers
       val_opt (int) - minimal (optimal) cost of inverse QROM with two registers
    """

    k1_opt, k2_opt = 0, 0
    val_opt = 1e50
    # Doing this as a stupid loop for now, worth refactoring but cost is quick regardless
    # Biggest concern is if k1 / k2 range is not large enough!
    for k1 in range(1, 17):
        for k2 in range(1, 17):
            value = np.ceil(L1 / np.power(2, k1)) * np.ceil(L2 / np.power(2, k2)) +\
                np.power(2, k1 + k2)
            if value < val_opt:
                val_opt = value
                k1_opt = k1
                k2_opt = k2

    assert val_opt.is_integer()
    return int(np.power(2,k1_opt)), int(np.power(2,k2_opt)), int(val_opt)

def power_two(m: int) -> int:
    """ Return the power of two that is a factor of m """
    assert m >= 0
    if m % 2 == 0:
        count = 0
        while (m > 0) and (m % 2 == 0):
            m = m // 2
            count += 1
        return count
    return 0

def read_cas(integral_path, num_alpha: Optional[int] = None, num_beta: Optional[int] = None):
    """ Read CAS Hamiltonian from pre-computed HD5 file.

    Args:
        integral_path (str) - path to hd5 file containing CAS one and two body terms 
        num_alpha (int, optional) - number of spin up electrons in CAS space
        num_beta (int, optional) - number of spin down electrons in CAS space

    Returns:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin up electrons in CAS space
        num_beta (int) - number of spin down electrons in CAS space
    """

    with h5py.File(integral_path, "r") as f:
        eri = np.asarray(f['eri'][()])
        #FIXME: h1 is sometimes called different things. We should make this consistent
        try:
            h1  = np.asarray(f['h0'][()])
        except KeyError:
            h1  = np.asarray(f['hcore'][()])
        # ecore sometimes exists, and sometimes as enuc (no frozen electrons) ... set to zero if N/A
        try:
            ecore = float(f['ecore'][()])
        except KeyError:
            try:
                ecore = float(f['enuc'][()])
            except KeyError:
                ecore = 0.0
        # attempt to read the number of spin up and spin down electrons if not input directly
        # FIXME: make hd5 key convention consistent
        if (num_alpha is None) or (num_beta is None):
            try:
                num_alpha = int(f['active_nalpha'][()])
            except KeyError:
                sys.exit("No values found on file for num_alpha (key: 'active_nalpha' in h5). " + \
                         "Try passing in a value for num_alpha, or re-check integral file.")
            try:
                num_beta = int(f['active_nbeta'][()])
            except KeyError:
                sys.exit("No values found on file for num_beta (key: 'active_nbeta' in h5). " + \
                         "Try passing in a value for num_beta, or re-check integral file.")

    n_orb = len(h1)  # number orbitals
    assert [n_orb] * 4 == [*eri.shape]  # check dims are consistent

    return h1, eri, ecore, num_alpha, num_beta

def gen_cas(mf, cas_orbitals: int, cas_electrons: int, avas_orbs=None):
    """ Generate CAS Hamiltonian given a PySCF mean field object

    Args:
        mf (PySCF mean field object) - instantiation of PySCF mean field method class
        cas_orbitals (int) - number of orbitals in CAS space
        cas_electrons (int) - number of electrons in CAS space

    Returns:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin up electrons in CAS space
        num_beta (int) - number of spin down electrons in CAS space
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
        num_alpha = mf.nelec[0] - frozen_alpha
        num_beta  = mf.nelec[1] - frozen_beta
        assert np.isclose(num_beta + num_alpha, cas_electrons)

    else:
        assert cas_electrons % 2 == 0
        num_alpha = cas_electrons // 2
        num_beta  = cas_electrons // 2

    return h1, eri, ecore, num_alpha, num_beta

def save_cas(fname, h1, eri, ecore, num_alpha, num_beta):
    """ Save CAS Hamiltonian to HD5 file.

    Args:
        fname (str) - path to hd5 file to be created containing CAS one and two body terms 
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin alpha electrons in Hamiltonian
        num_beta (int) - number of spin beta electrons in Hamiltonian
    """
    with h5py.File(fname, 'w') as fid:
        fid.create_dataset('ecore', data=float(ecore), dtype=float)
        fid.create_dataset('h0', data=h1)  # note the name change to be consistent with THC paper
        fid.create_dataset('eri', data=eri)
        fid.create_dataset('active_nalpha', data=int(num_alpha), dtype=int)
        fid.create_dataset('active_nbeta', data=int(num_beta), dtype=int)

def ccsd_t(h1, eri, ecore, num_alpha: int, num_beta: int, eri_full = None, use_kernel=True, \
    no_triples=False) -> Tuple[float, float, float]:
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
    n_orb = h1.shape[0]
    alpha_diag = [1] * num_alpha + [0] * (n_orb - num_alpha)
    beta_diag  = [1] * num_beta  + [0] * (n_orb - num_beta)

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

    mf.init_guess = '1e'
    mf.mo_coeff = np.eye(n_orb)
    mf.mo_occ = np.array(alpha_diag) + np.array(beta_diag)
    w, v = np.linalg.eigh(mf.get_fock())
    mf.mo_energy = w

    # This option should not be used other than for debugging, as it will rotate the interaction
    # tensors into a new basis. However, sometimes it is useful to see if the "converged SCF" that 
    # is read in is really converged or not ...
    if use_kernel: 
        mf.conv_tol = 1e-9
        mf.init_guess = '1e'
        mf.verbose=4
        mf.diis_space = 24
        mf.diis_start_cycle = 4
        mf.level_shift = 0.25
        mf.max_cycle = 500 
        mf.kernel()
        mol = mf.stability()[0]
        dm = mf.make_rdm1(mol, mf.mo_occ)
        mf = mf.run(dm)
        mol = mf.stability()[0]
        dm = mf.make_rdm1(mol, mf.mo_occ)
        mf = mf.run(dm)
        #save_cas('eri_reiher_newscf.h5',*gen_cas(mf,h1.shape[0],num_alpha + num_beta))
        mf._eri = eri # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)

        # Check SCF has not changed by doing restart!
        print(scf_energy, mf.e_tot)
        try:
            assert np.isclose(scf_energy, mf.e_tot,rtol=1e-14)
        except AssertionError:
            print("WARNING: SCF energy from input integrals does not match SCF energy from mf.kernel()")
            print("E(SCF, ints) = {:12.6f} whereas E(SCF) = {:12.6f}".format(scf_energy,mf.e_tot))

    # Now set the eri's to the (possibly rank-reduced) ERIs
    mf._eri = eri # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)
    mf.mol.incore_anyway = True

    mycc = cc.CCSD(mf)
    mycc.max_cycle = 800
    mycc.conv_tol = 1E-8
    mycc.conv_tol_normt = 1E-4
    mycc.diis_space = 24
    mycc.diis_start_cycle = 4
    mycc.verbose = 4
    mycc.kernel()

    if no_triples:
        et = 0.0
    else:
        et = mycc.ccsd_t()

    e_scf = scf_energy 
    e_cor = mycc.e_corr + et
    e_tot = e_scf + e_cor

    print("E(SCF, ints): ", scf_energy)
    print("E(SCF):       ", e_scf)
    print("E(cor):       ", e_cor)
    print("Total energy: ", e_tot)
    return e_scf, e_cor, e_tot

class RunSilent(object):
    """ Context manager to prevent function from writing anything to stdout/stderr 
        e.g. for noisy_function(), wrap it like so

        with RunSilent():
            noisy_function()

        ... and your terminal will no longer be littered with prints
    """
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


if __name__ == '__main__':

    print("chemftr full CCSD(T) from h1/eri/ecore tensors vs regular PySCF CCSD(T) ")
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
        mycc.conv_tol = 1E-7
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

        chemftr_results = ccsd_t(*gen_cas(mf, n_orb, n_elec))
        chemftr_results = np.asarray(chemftr_results)

        # ignore relative tolerance, we just want absolute tolerance
        print("Pyscf:", pyscf_etot, " Chemftr: ", chemftr_results[-1])
        assert np.allclose(pyscf_results,chemftr_results,rtol=1E-14)


    print("\nReduced CCSD(T) vs PySCF CAS(No,2e) ")
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
        _, _, chemftr_etot = ccsd_t(*gen_cas(mf, n_orb, n_elec),no_triples=True)

        # ignore relative tolerance, we just want absolute tolerance
        print("Pyscf:", pyscf_etot, " Chemftr: ", chemftr_etot)
        assert np.isclose(pyscf_etot,chemftr_etot,rtol=1E-14)
