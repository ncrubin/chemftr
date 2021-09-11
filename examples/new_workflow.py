from pyscf import gto, scf, lo, tools, mcscf, ao2mo, cc
from pyscf.mcscf import avas
import h5py
from chemftr.util import gen_cas, RunSilent
from chemftr.molecule import *
from chemftr import sf, df
import openfermion as of
from openfermion.chem.molecular_data import spinorb_from_spatial

mol, mf = load_cas('../src/chemftr/integrals/eri_reiher.h5',num_alpha=27, num_beta=27)
print("E(SCF, local) ", mf.e_tot)  # Hamiiltonian uses localized orbitals
mf.verbose = 0
mf.max_cycle = 500
mf.diis_space = 24
mf.level_shift = 0.25
mf.kernel()
mf = stability(mf)
print("E(SCF, canon) ", mf.e_tot)  # After SCF, energy is true HF variational energy 

# Saving and loading is pretty easy now 
save_cas('reiher.h5',mf)
mol, mf = load_cas('reiher.h5')

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
