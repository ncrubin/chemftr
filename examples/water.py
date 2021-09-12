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
mol.unit = 'bohr'
mol.symmetry = False
mol.build()

mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
print("E(SCF, canon) ", mf.e_tot)  # After SCF, energy is true HF variational energy 

mf = localize(mf, verbose=4)
mycc = cc.CCSD(mf)
mycc.verbose = 3
mycc.kernel()

save_cas('water.h5',mf)
_ , mf = load_cas('water.h5')
mf.mol = mol.copy()
mf.verbose = 0
#mf.kernel()
print("E(SCF, reload) ", mf.e_tot)  # After SCF, energy is true HF variational energy 

mf = localize(mf, verbose=4)
mycc = cc.CCSD(mf)
mycc.verbose = 3
mycc.kernel()


