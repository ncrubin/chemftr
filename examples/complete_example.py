from pyscf import gto, scf
from chemftr.molecule import stability, localize, avas_active_space
from chemftr.sf import single_factorization
from chemftr.df import double_factorization

mol = gto.M(
    atom = 'Fe 0.0 0.0 0.0',
    basis = 'ccpvtz',
    symmetry = False,
    charge = 3,
    spin = 5
)
mf = scf.ROHF(mol)
mf.kernel()

mf = stability(mf)
mf = localize(mf)

# use larger basis for minao to select non-valence...here select 4d as well for double-shell effect
mol, mf = avas_active_space(mf, ao_list=['Fe 2s', 'Fe 2p', 'Fe 3d', 'Fe 4d'],minao='ccpvdz') 
#mf.mol = mol

# make pretty SF costing table
single_factorization(mf, rank_range=[20,25,30,35,40,45,50])

# make pretty DF costing table
double_factorization(mf, thresh_range=[4e-3,3e-3,2e-3,1e-3,9e-4,8e-4,7e-4])
