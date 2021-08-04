from chemftr.avas import AVAS

# AVAS class is a manager for the pipeline. Most functions utilize saved intermediates, so feel free
# to comment out routines already run. If a function doesn't have enough information, it won't work!

# This is a very small example, so kind of finicky with thresholds and rank-reductions
chem = AVAS('Fe 0.0 0.0 0.0',charge=3,multiplicity=6,basis='ccpvtz')  # Fe(III) high-spin d5
chem.do_scf()
chem.localize(loc_type='pm')
#print(chem.mf.mol.ao_labels())  # see labels of the AO basis to choose from
chem.do_avas(ao_list=['Fe 3s', 'Fe 3p', 'Fe 3d', 'Fe 4s'])

# make pretty SF costing table
chem.do_single_factorization(rank_range=[20,25,30,35,40,45,50])

# make pretty DF costing table
chem.do_double_factorization(thresh_range=[4e-3,3e-3,2e-3,1e-3,9e-4,8e-4,7e-4])
