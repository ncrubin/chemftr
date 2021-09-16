""" Do the costing estimated for a water molecule """

from pyscf import gto, scf
from chemftr import sf, df, thc
from chemftr.molecule import stability, localize, avas_active_space

# input is just like any other PySCF script
mol = gto.M(
    atom = '''O    0.000000      -0.075791844    0.000000
              H    0.866811829    0.601435779    0.000000
              H   -0.866811829    0.601435779    0.000000
           ''',
    basis = 'ccpvtz',
    symmetry = False,
    charge = 1,
    spin = 1
)
mf = scf.ROHF(mol)
mf.verbose = 4
mf.kernel()

# make sure wave function is stable before we proceed
mf = stability(mf)

# localize before automatically selecting active space with AVAS
mf = localize(mf, loc_type='pm')  # default is loc_type ='pm' (Pipek-Mezey)
# you can use larger basis for `minao` to select non-valence...here select O 3s and 3p as well 
mol, mf = avas_active_space(mf, ao_list=['H 1s', 'O 2s', 'O 2p', 'O 3s', 'O 3p'], minao='ccpvtz') 

# make pretty SF costing table
sf.generate_costing_table(mf, name='water', rank_range=[20,25,30,35,40,45,50])

# make pretty DF costing table
df.generate_costing_table(mf, name='water', thresh_range=[1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5]) 

# make pretty THC costing table
thc.generate_costing_table(mf, name='water', nthc_range=[20,25,30,35,40,45,50]) 
