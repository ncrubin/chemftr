from pyscf import gto, scf, lo, tools, mcscf, ao2mo, cc
import h5py
from chemftr.util import gen_cas, RunSilent
from chemftr.molecule import *
from chemftr import sf, df
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

mol = gto.Mole()
mol.atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587'''
mol.basis = 'ccpvdz'
mol.symmetry = False
mol.build()

mf = scf.RHF(mol)
mf.verbose = 0
mf.max_cycle = 500
mf.diis_space = 24
mf.level_shift = 0.25
mf.kernel()
mf = stability(mf)
print("E(SCF, canon) ", mf.e_tot)  # After SCF, energy is true HF variational energy 

# Try CCSD(T) on re-loaded canonical Reiher Hamiltonian
mycc = cc.CCSD(mf)
mycc.verbose = 0
mycc.max_cycle = 500
mycc.conv_tol = 1E-8
mycc.conv_tol_normt = 1E-4
mycc.diis_space = 24
mycc.diis_start_cycle = 4
mycc.kernel()
et = mycc.ccsd_t()

pyscf_escf = mf.e_tot
pyscf_ecor = mycc.e_corr + et
pyscf_etot = pyscf_escf + pyscf_ecor
pyscf_results = np.array([pyscf_escf, pyscf_ecor, pyscf_etot])

print("E(SCF):       ", pyscf_escf)
print("E(cor):       ", pyscf_ecor)
print("E(CCSD(T)):   ", pyscf_etot)

mf = localize(mf)

# Try CCSD(T) on localized water 
mycc = cc.CCSD(mf)
mycc.verbose = 0
mycc.max_cycle = 500
mycc.conv_tol = 1E-8
mycc.conv_tol_normt = 1E-4
mycc.diis_space = 24
mycc.diis_start_cycle = 4
mycc.kernel()
et = mycc.ccsd_t()

pyscf_escf = mf.e_tot
pyscf_ecor = mycc.e_corr + et
pyscf_etot = pyscf_escf + pyscf_ecor
pyscf_results = np.array([pyscf_escf, pyscf_ecor, pyscf_etot])

print("E(cor, local):       ", pyscf_ecor)
print("E(CCSD(T), local):   ", pyscf_etot)

norb, ne, avas_orbs = get_avas_active_space(mf,ao_list=['H 1s', 'O 2s', 'O 2p'])

