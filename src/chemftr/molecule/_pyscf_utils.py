""" Drivers for various PySCF electronic structure routines """

import numpy as np
import h5py
import sys
from typing import Optional
from pyscf import gto, scf, ao2mo, ci, cc, fci, mp, mcscf, lo, tools
from pyscf.mcscf import avas

def stability(pyscf_mf):
    """
    Test wave function stability and re-optimize SCF.

    Args:
        pyscf_mf: PySCF mean field object (e.g. `scf.RHF()`)

    Returns:
        pyscf_mf: Updated PySCF mean field object
    """
    new_orbitals = pyscf_mf.stability()[0]
    new_1rdm = pyscf_mf.make_rdm1(new_orbitals, pyscf_mf.mo_occ)
    pyscf_mf = pyscf_mf.run(new_1rdm)

    return pyscf_mf

def localize(pyscf_mf, loc_type='pm', verbose=0):
    """ Localize orbitals given a PySCF mean-field object 

    Args:
        pyscf_mf:  PySCF mean field object
        loc_type (str): localization type; Pipek-Mezey ('pm') or Edmiston-Rudenberg ('er') 
        verbose (int): print level during localization

    Returns:
        pyscf_mf:  Updated PySCF mean field object with localized orbitals
    """

    # Split-localization (localize doubly-occupied, singly-occupied, and virtual separately)
    docc_idx = np.where(np.isclose(pyscf_mf.mo_occ, 2.))[0]
    socc_idx = np.where(np.isclose(pyscf_mf.mo_occ, 1.))[0]
    virt_idx = np.where(np.isclose(pyscf_mf.mo_occ, 0.))[0]

    # Pipek-Mezey
    if loc_type.lower() == 'pm':
        print("Localizing doubly occupied ... ", end="")
        loc_docc_mo = lo.PM(pyscf_mf.mol, pyscf_mf.mo_coeff[:, docc_idx]).kernel(verbose=verbose)
        print("singly occupied ... ", end="")
        loc_socc_mo = lo.PM(pyscf_mf.mol, pyscf_mf.mo_coeff[:, socc_idx]).kernel(verbose=verbose)
        print("virtual ... ", end="")
        loc_virt_mo = lo.PM(pyscf_mf.mol, pyscf_mf.mo_coeff[:, virt_idx]).kernel(verbose=verbose)
        print("DONE")

    # Edmiston-Rudenberg
    elif loc_type.lower() == 'er':
        print("Localizing doubly occupied ... ", end="")
        loc_docc_mo = lo.ER(pyscf_mf.mol, pyscf_mf.mo_coeff[:, docc_idx]).kernel(verbose=verbose)
        print("singly occupied ... ", end="")
        loc_socc_mo = lo.ER(pyscf_mf.mol, pyscf_mf.mo_coeff[:, socc_idx]).kernel(verbose=verbose)
        print("virtual ... ", end="")
        loc_virt_mo = lo.ER(pyscf_mf.mol, pyscf_mf.mo_coeff[:, virt_idx]).kernel(verbose=verbose)
        print("DONE")

    # overwrite orbitals with localized orbitals 
    pyscf_mf.mo_coeff[:,docc_idx] = loc_docc_mo.copy()
    pyscf_mf.mo_coeff[:,socc_idx] = loc_socc_mo.copy()
    pyscf_mf.mo_coeff[:,virt_idx] = loc_virt_mo.copy()

    return pyscf_mf

def cas_from_avas(pyscf_mf, ao_list=None, molden_fname='avas_localized_orbitals', **kwargs): 
    """ Return active space and re-ordered orbitals from AVAS 

    Args:
        pyscf_mf:  PySCF mean field object
        ao_list: list of strings of AOs (print mol.ao_labels() to see options)
                 Example: ao_list = ['H 1s', 'O 2p', 'O 2s'] for water
        verbose (bool): do additional print
        molden_fname (str): MOLDEN filename to save AVAS active space orbitals. Default is to save
                            to 'avas_localized_orbitals.molden' 
        **kwargs: other keyworded arguments to pass into avas.avas()

    Returns:
        active_norb (int): number of active orbitals selected by AVAS
        active_ne (int): number of active electrons selected by AVAS
        reordered_orbitals: orbital initial guess for CAS
    """

    # Note: requires openshell_option = 3 for this to work
    # we also require canonicalize = False so that we don't destroy local orbitals
    avas_output = avas.avas(pyscf_mf, ao_list, canonicalize=False, openshell_option=3,**kwargs)
    active_norb, active_ne, reordered_orbitals = avas_output

    active_alpha, active_beta = get_num_active_alpha_beta(pyscf_mf, active_ne)

    if molden_fname is not None:
        # save set of localized orbitals for active space
        if isinstance(pyscf_mf, scf.rohf.ROHF):
            frozen_alpha = pyscf_mf.nelec[0] - active_alpha
            assert frozen_alpha >= 0
        else:
            frozen_alpha = pyscf_mf.mol.nelectron // 2  - active_alpha
            assert frozen_alpha >= 0

        active_space_idx = slice(frozen_alpha, frozen_alpha + active_norb)
        active_mos = reordered_orbitals[:,active_space_idx]
        tools.molden.from_mo(pyscf_mf.mol, molden_fname+'.molden', mo_coeff=active_mos)

    return active_norb, active_ne, reordered_orbitals


def load_cas(fname, num_alpha: Optional[int] = None, num_beta: Optional[int] = None):
    """ Load CAS Hamiltonian from pre-computed HD5 file into a PySCF mean-field object

    Args:
        fname (str): path to hd5 file to be created containing CAS one and two body terms 
        num_alpha (int, optional): number of spin up electrons in CAS space
        num_beta (int, optional):  number of spin down electrons in CAS space

    Returns:
        pyscf_mol: PySCF molecule object
        pyscf_mf:  PySCF mean field object
    """

    with h5py.File(fname, "r") as f:
        eri = np.asarray(f['eri'][()])
        #FIXME: h1 is sometimes called different things. We should make this consistent
        try:
            h1  = np.asarray(f['h0'][()])
        except KeyError:
            try:
                h1  = np.asarray(f['hcore'][()])
            except KeyError:
                try:
                    h1 = np.asarray(f['h1'][()])
                except KeyError:
                    raise KeyError("Could not find 1-electron Hamiltonian")
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
                sys.exit("In `load_cas`: \n" + \
                         " No values found on file for num_alpha (key: 'active_nalpha' in h5). " + \
                         " Try passing in a value for num_alpha, or re-check integral file.")
            try:
                num_beta = int(f['active_nbeta'][()])
            except KeyError:
                sys.exit("In `load_cas`: \n" + \
                         " No values found on file for num_beta (key: 'active_nbeta' in h5). " + \
                         " Try passing in a value for num_beta, or re-check integral file.")

    n_orb = len(h1)  # number orbitals
    assert [n_orb] * 4 == [*eri.shape]  # check dims are consistent

    pyscf_mol = gto.M()
    pyscf_mol.nelectron = num_alpha + num_beta
    n_orb = h1.shape[0]
    alpha_diag = [1] * num_alpha + [0] * (n_orb - num_alpha)
    beta_diag  = [1] * num_beta  + [0] * (n_orb - num_beta)


    # Assumes Hamiltonian is either RHF or ROHF ... should be OK since UHF will have two h1s, etc.
    if num_alpha == num_beta:
        pyscf_mf = scf.RHF(pyscf_mol)
        scf_energy = ecore + \
                     2*np.einsum('ii',  h1[:num_alpha,:num_alpha]) + \
                     2*np.einsum('iijj',eri[:num_alpha,:num_alpha,:num_alpha,:num_alpha]) - \
                       np.einsum('ijji',eri[:num_alpha,:num_alpha,:num_alpha,:num_alpha])

    else:
        pyscf_mf = scf.ROHF(pyscf_mol)
        pyscf_mf.nelec = (num_alpha, num_beta)
        # grab singly and doubly occupied orbitals (assumes high-spin open shell)
        docc = slice(None                    , min(num_alpha, num_beta))
        socc = slice(min(num_alpha, num_beta), max(num_alpha, num_beta))
        scf_energy = ecore + \
                     2.0*np.einsum('ii',h1[docc, docc]) + \
                         np.einsum('ii',h1[socc, socc]) + \
                     2.0*np.einsum('iijj',eri[docc, docc, docc, docc]) - \
                         np.einsum('ijji',eri[docc, docc, docc, docc]) + \
                         np.einsum('iijj',eri[socc, socc, docc, docc]) - \
                     0.5*np.einsum('ijji',eri[socc, docc, docc, socc]) + \
                         np.einsum('iijj',eri[docc, docc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri[docc, socc, socc, docc]) + \
                     0.5*np.einsum('iijj',eri[socc, socc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri[socc, socc, socc, socc])

    pyscf_mf.get_hcore  = lambda *args: np.asarray(h1)
    pyscf_mf.get_ovlp   = lambda *args: np.eye(h1.shape[0])
    pyscf_mf.energy_nuc = lambda *args: ecore
    pyscf_mf._eri = eri # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)
    pyscf_mf.e_tot = scf_energy

    pyscf_mf.init_guess = '1e'
    pyscf_mf.mo_coeff = np.eye(n_orb)
    pyscf_mf.mo_occ = np.array(alpha_diag) + np.array(beta_diag)
    w, v = np.linalg.eigh(pyscf_mf.get_fock())
    pyscf_mf.mo_energy = w

    return pyscf_mol, pyscf_mf

def gen_cas(pyscf_mf, cas_orbitals: Optional[int] = None,
             cas_electrons: Optional[int] = None, avas_orbs=None):
    """ Return CAS Hamiltonian tensors from a PySCF mean-field object 

    Args:
        pyscf_mf: PySCF mean field object 
        cas_orbitals (int, optional):  number of orbitals in CAS space, default all orbitals
        cas_electrons (int, optional): number of electrons in CAS space, default all electrons
        avas_orbs (ndarray, optional): orbitals selected by AVAS in PySCF

    Returns:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin up electrons in CAS space
        num_beta (int) - number of spin down electrons in CAS space
    """

    # Only RHF or ROHF possible with mcscf.CASCI
    assert isinstance(pyscf_mf, scf.rhf.RHF)  # ROHF is child of RHF class

    if cas_orbitals is None:
        cas_orbitals = len(pyscf_mf.mo_coeff)
    if cas_electrons is None:
        cas_electrons = pyscf_mf.mol.nelectron

    cas = mcscf.CASCI(pyscf_mf, ncas = cas_orbitals, nelecas = cas_electrons)
    h1, ecore = cas.get_h1eff(mo_coeff = avas_orbs)
    eri = cas.get_h2cas(mo_coeff = avas_orbs)
    eri = ao2mo.restore('s1', eri, h1.shape[0])  # chemist convention (11|22)
    ecore = float(ecore)

    num_alpha, num_beta = get_num_active_alpha_beta(pyscf_mf, cas_electrons) 

    return h1, eri, ecore, num_alpha, num_beta 

def get_num_active_alpha_beta(pyscf_mf, cas_electrons):
    # Sanity checks and active space info
    total_electrons = pyscf_mf.mol.nelectron
    frozen_electrons  = total_electrons - cas_electrons
    assert frozen_electrons % 2 == 0

    # Again, recall ROHF == RHF but RHF != ROHF, and we only do either RHF or ROHF
    if isinstance(pyscf_mf, scf.rohf.ROHF):
        frozen_alpha = frozen_electrons // 2
        frozen_beta  = frozen_electrons // 2
        num_alpha = pyscf_mf.nelec[0] - frozen_alpha
        num_beta  = pyscf_mf.nelec[1] - frozen_beta
        assert np.isclose(num_beta + num_alpha, cas_electrons)

    else:
        assert cas_electrons % 2 == 0
        num_alpha = cas_electrons // 2
        num_beta  = cas_electrons // 2

    return num_alpha, num_beta

def save_cas(fname, pyscf_mf, cas_orbitals: Optional[int] = None, 
             cas_electrons: Optional[int] = None, avas_orbs=None):
    """ Save CAS Hamiltonian from a PySCF mean-field object to an HD5 file 

    Args:
        fname (str): path to hd5 file to be created containing CAS one and two body terms
        pyscf_mf: PySCF mean field object 
        cas_orbitals (int, optional):  number of orbitals in CAS space, default all orbitals
        cas_electrons (int, optional): number of electrons in CAS space, default all electrons
        avas_orbs (ndarray, optional): orbitals selected by AVAS in PySCF
    """
    h1, eri, ecore, num_alpha, num_beta = gen_cas(pyscf_mf, cas_orbitals, cas_electrons, avas_orbs)

    with h5py.File(fname, 'w') as fid:
        fid.create_dataset('ecore', data=float(ecore), dtype=float)
        fid.create_dataset('h0', data=h1)  # note the name change to be consistent with THC paper
        fid.create_dataset('eri', data=eri)
        fid.create_dataset('active_nalpha', data=int(num_alpha), dtype=int)
        fid.create_dataset('active_nbeta', data=int(num_beta), dtype=int)

